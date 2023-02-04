import gzip
import numpy
import random
import tensorflow as tf
import json
import itertools
import operator
from statistics import mean
from sklearn import linear_model

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)
    
f = gzip.open("modcloth_final_data.json.gz")
dataset1 = []
for l in f:
    dataset1.append(json.loads(l))

f = gzip.open("renttherunway_final_data.json.gz")
dataset2 = []
for l in f:
    dataset2.append(json.loads(l))    
    
ratingsTrain = dataset2[:150000]
ratingsValid = dataset2[150000:]    


ages = []
for d in ratingsTrain:
    try:  
        ages.append(int(d['age']))
    except:
        pass
avg_age = sum(ages) / len(ages)

UserItemRatingTrain = []
UserItemRatingValid = []
for d in ratingsTrain:
    try:
        UserItemRatingTrain.append((d['user_id'], d['item_id'], int(d['rating'])))
    except:
        UserItemRatingTrain.append((d['user_id'], d['item_id'], 0))
    
for d in ratingsValid:
    try:
        UserItemRatingValid.append((d['user_id'], d['item_id'], int(d['rating'])))
    except:
        UserItemRatingValid.append((d['user_id'], d['item_id'], 0))

c = 0
for i in ratingsTrain:
    try:
        UserItemRatingTrain[c] = UserItemRatingTrain[c] + (int(i['age']),)
    except:
        UserItemRatingTrain[c] = UserItemRatingTrain[c] + (avg_age,)
    c += 1
     
c = 0
for i in ratingsValid:
    try:
        UserItemRatingValid[c] = UserItemRatingValid[c] + (int(i['age']),)
    except:
        UserItemRatingValid[c] = UserItemRatingValid[c] + (avg_age,)
    c += 1



def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)

userIDs = {}
itemIDs = {}
c = 0
for u in UserItemRatingTrain:
    c += 1
    print(c)
    if u[0] not in userIDs:
        userIDs[u[0]] = len(userIDs)
    if u[1] not in itemIDs:
        itemIDs[u[1]] = len(itemIDs)

itemAvg_age = {}
c = 0 
for i in itemIDs.keys():
    l = [d[3] for d in UserItemRatingTrain if d[1] == i]
    itemAvg_age[i] = sum(l) / len(l)
    c += 1
    print(c)


c = 0
for i in UserItemRatingTrain:
    try:
        c += float(i[2])
    except:
        c += 0

mu = c / len([d for d in UserItemRatingTrain])

#always pred average
always_mu = []
for i in range(len(UserItemRatingTrain)):
    always_mu.append(mu)        

UserItemAgeTrain = []
UserItemAgeVali = []
for d in UserItemRatingTrain:
    UserItemAgeTrain.append((d[0], d[1], d[2], itemAvg_age[d[1]] - d[3]))
for d in UserItemRatingValid:
    try:
        UserItemAgeVali.append((d[0], d[1], d[2], itemAvg_age[d[1]] - d[3]))
    except:
        UserItemAgeVali.append((d[0], d[1], d[2], avg_age - d[3]))


optimizer = tf.keras.optimizers.Adam(0.05)


class LatentFactorModel(tf.keras.Model):
    def __init__(self, mu, K, lamb, lamb2):
        super(LatentFactorModel, self).__init__()
        # Initialize to average
        self.alpha = tf.Variable(mu, trainable=False)
        # Initialize to small random values
        self.agecoef = tf.Variable(0.0001)
        self.betaU = tf.Variable(tf.random.normal([len(userIDs)],stddev=0.001))
        self.betaI = tf.Variable(tf.random.normal([len(itemIDs)],stddev=0.001))
        self.gammaU = tf.Variable(tf.random.normal([len(userIDs),K],stddev=0.001))
        self.gammaI = tf.Variable(tf.random.normal([len(itemIDs),K],stddev=0.001))
        self.lamb = lamb
        self.lamb2 = lamb2

    # Prediction for a single instance (useful for evaluation)
    def predict(self, u, i):
       alpha = self.alpha
       betaU = self.betaU[u]
       betaI = self.betaI[i]
       gammaU = self.gammaU[u]
       gammaI = self.gammaI[i]
       agecoef = self.agecoef
       return alpha, betaU, betaI, gammaU, gammaI, agecoef
   
        
   
    # Regularizer
    def reg(self):
        return self.lamb * (tf.reduce_sum(self.betaU**2) +\
                            tf.reduce_sum(self.betaI**2)) +\
                            self.lamb2 * (tf.reduce_sum(self.gammaU**2) +\
                            tf.reduce_sum(self.gammaI**2))  
    
    # Prediction for a sample of instances
    def predictSample(self, sampleU, sampleI, sampleA):
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        a = tf.convert_to_tensor(sampleA)
        beta_u = tf.nn.embedding_lookup(self.betaU, u)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        gamma_u = tf.nn.embedding_lookup(self.gammaU, u)
        gamma_i = tf.nn.embedding_lookup(self.gammaI, i)
        pred = self.alpha + beta_u + beta_i +\
               tf.reduce_sum(tf.multiply(gamma_u, gamma_i), 1) + a * self.agecoef
        return pred
    
    # Loss
    def call(self, sampleU, sampleI, sampleR, sampleA):
        pred = self.predictSample(sampleU, sampleI, sampleA)
        r = tf.convert_to_tensor(sampleR, dtype=tf.float32)
        return tf.nn.l2_loss(pred - r)*2 / len(sampleR)



modelLFM = LatentFactorModel(mu, 2, 0.0001, 0.01)


def trainingStep(model, interactions):
    Nsamples = 3000
    with tf.GradientTape() as tape:
        sampleU, sampleI, sampleR, sampleA = [], [], [], []
        for _ in range(Nsamples):
            u,i,r,a = random.choice(interactions)
            sampleU.append(userIDs[u])
            sampleI.append(itemIDs[i])
            sampleR.append(r)
            sampleA.append(a)
        
        loss = model(sampleU,sampleI,sampleR, sampleA)
        loss += model.reg()
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients((grad, var) for
                              (grad, var) in zip(gradients, model.trainable_variables)
                              if grad is not None)
    return loss.numpy()
        
for i in range(300):
    obj = trainingStep(modelLFM, UserItemAgeTrain)
    if (i % 10 == 9): print("iteration " + str(i+1) + ", objective = " + str(obj))        

# In[1]: get trainable variables and pred 
    
train_prediction = []
pred_temp = []
betaU_list = [float(d) for d in modelLFM.trainable_variables[1]]
betaI_list = [float(d) for d in modelLFM.trainable_variables[2]]
gammaU_list = [d for d in modelLFM.trainable_variables[3]]
gammaI_list = [d for d in modelLFM.trainable_variables[4]]
agecoef_list = [float(modelLFM.trainable_variables[0])]
betaU = {}
betaI = {}
gammaU = {}
gammaI = {}
agecoefI = {}
c = 0 
d = 0
e = 0
f = 0
g = 0
for u in UserItemRatingTrain:
    if u[0] not in betaU:
        betaU[u[0]] = betaU_list[c]
        c += 1
    if u[1] not in betaI:
        betaI[u[1]] = betaI_list[d]
        d += 1
    if u[0] not in gammaU:
        gammaU[u[0]] = gammaU_list[e]
        e += 1
    if u[1] not in gammaI:
        gammaI[u[1]] = gammaI_list[f]
        f += 1
    if u[1] not in agecoefI:
        agecoefI[u[1]] = agecoef_list[0]
for d in UserItemAgeTrain:
    u,i,a = d[0], d[1], d[3]
    p = mu + betaU[u] + betaI[i] +\
                numpy.dot(gammaU[u], gammaI[i]) + a * agecoefI[i] 
    train_prediction.append(p)  

  

    

# In[2]:    

#always mean
print('always_mu in training: ' + str(MSE(always_mu, [d[2] for d in UserItemRatingTrain])))
#LF               
print('training MSE: ' + str(MSE(train_prediction, [d[2] for d in UserItemRatingTrain])))

# In[3]: Valid MSE
agecoefI = agecoef_list[0]
vali_prediction = []
for d in UserItemAgeVali:
    alpha_ = mu
    betaU_ = 0
    betaI_ = 0
    gammaI_ = 0
    gammaU_ = 0
    u, i, a = d[0], d[1], d[3]
    if u in userIDs.keys():
        betaU_ = betaU[u]
        gammaU_ = gammaU[u]
    if i in itemIDs.keys():
        betaI_ = betaI[i]
        gammaI_ = gammaI[i]
    p = alpha_ + betaU_ + betaI_ +\
                numpy.dot(gammaU_, gammaI_) + a * agecoefI
    try:
        p = p[0]
    except:
        pass
    if p > 10:
        p = 10
    if p < 0:
        p = 0
    
    vali_prediction.append(p)


print('always_mu in vali: ' + str(MSE(always_mu, [d[2] for d in UserItemRatingValid])))
print('vali MSE: ' + str(MSE(vali_prediction, [d[2] for d in UserItemRatingValid])))   
