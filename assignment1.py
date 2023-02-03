#!/usr/bin/env python
# coding: utf-8

# In[155]:


import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model
from operator import itemgetter


# In[156]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)
def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r
answers = {}


# In[157]:


allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)


# In[158]:


ratingDict = {}
ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))
    ratingDict[(u,b)] = r
    usersPerItem[b].add(u)
    itemsPerUser[u].add(b)


# In[159]:


trainRatings = [r[2] for r in ratingsTrain]
globalAverage = sum(trainRatings) * 1.0 / len(trainRatings)


# In[160]:


validMSE = 0
for u,b,r in ratingsValid:
  se = (r - globalAverage)**2
  validMSE += se

validMSE /= len(ratingsValid)

print("Validation MSE (average only) = " + str(validMSE))


# In[161]:


##################################################
# Read prediction                                #
##################################################


# In[162]:


# From baseline code
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead/2: break


# In[163]:


# Generate a negative set

userSet = set()
bookSet = set()
readSet = set()

for u,b,r in allRatings:
    userSet.add(u)
    bookSet.add(b)
    readSet.add((u,b))

lUserSet = list(userSet)
lBookSet = list(bookSet)

notRead = set()
for u,b,r in ratingsValid:
    #u = random.choice(lUserSet)
    b = random.choice(lBookSet)
    while ((u,b) in readSet or (u,b) in notRead):
        b = random.choice(lBookSet)
    notRead.add((u,b))

readValid = set()
for u,b,r in ratingsValid:
    readValid.add((u,b))


# In[164]:


userAverages = {}
itemAverages = {}

for u in itemsPerUser:
    rs = [ratingDict[(u,i)] for i in itemsPerUser[u]]
    userAverages[u] = sum(rs) / len(rs)
    
for i in usersPerItem:
    rs = [ratingDict[(u,i)] for u in usersPerItem[i]]
    if len(rs) == 0 :
        itemAverages[i] = 0
    else:
        itemAverages[i] = sum(rs) / len(rs)


# In[165]:


def accuracy(predictions, y):
    correct = predictions == y
    return sum(correct) / len(correct)
    


# In[166]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom > 0:
        return numer/denom 
    return 0


# In[167]:


def Cosine(i1, i2):
    # Between two items
    inter = set(ratingsPerItem[i1]).intersection(set(ratingsPerItem[i2]))
    numer = 0
    denom1 = 0
    denom2 = 0
    for u in inter:
        numer += ratingDict[(u[0],i1)]*ratingDict[(u[0],i2)]
    for u in usersPerItem[i1]:
        denom1 += ratingDict[(u,i1)]**2
    for u in usersPerItem[i2]:
        denom2 += ratingDict[(u,i2)]**2
    denom = math.sqrt(denom1) * math.sqrt(denom2)
    if denom == 0: return 0
    return numer / denom


# In[168]:


def Cosine_u(u1, u2):
    # Between two items
    inter = set(ratingsPerUser[u1]).intersection(set(ratingsPerUser[u2]))
    numer = 0
    denom1 = 0
    denom2 = 0
    for i in inter:
        numer += ratingDict[(u1,i[0])]*ratingDict[(u2,i[0])]
    for i in itemsPerUser[u1]:
        denom1 += ratingDict[(u1,i)]**2
    for i in itemsPerUser[u2]:
        denom2 += ratingDict[(u2,i)]**2
    denom = math.sqrt(denom1) * math.sqrt(denom2)
    if denom == 0: return 0
    return numer / denom


# In[169]:


def Pearson(i1, i2):
    # Between two items
    iBar1 = itemAverages[i1]
    iBar2 = itemAverages[i2]
    inter = set(ratingsPerItem[i1]).intersection(set(ratingsPerItem[i2]))
    numer = 0
    denom1 = 0
    denom2 = 0
    for u in inter:
        numer += (ratingDict[(u[0],i1)] - iBar1)*(ratingDict[(u[0],i2)] - iBar2)
    for u in inter: #usersPerItem[i1]:
        denom1 += (ratingDict[(u[0],i1)] - iBar1)**2
    #for u in usersPerItem[i2]:
        denom2 += (ratingDict[(u[0],i2)] - iBar2)**2
    denom = math.sqrt(denom1) * math.sqrt(denom2)
    if denom == 0: return 0
    return numer / denom


# In[170]:


def Pearson_u(u1, u2):
    # Between two items
    global_avg = sum([userAverages[u] for u in userAverages]) / len([userAverages[u] for u in userAverages])
    ulist = [u for u in userAverages]
    if u1 not in ulist:
        iBar1 = global_avg
    else:
        iBar1 = userAverages[u1]
    if u2 not in ulist:
        iBar2 = global_avg
    else:
        iBar2 = userAverages[u2]
    inter = set(ratingsPerUser[u1]).intersection(set(ratingsPerUser[u2]))
    numer = 0
    denom1 = 0
    denom2 = 0
    for i in inter:
        numer += (ratingDict[(u1,i[0])] - iBar1)*(ratingDict[(u2,i[0])] - iBar2)
    for i in inter: #usersPerItem[i1]:
        denom1 += (ratingDict[(u1,i[0])] - iBar1)**2
    #for u in usersPerItem[i2]:
        denom2 += (ratingDict[(u2,i[0])] - iBar2)**2
    denom = math.sqrt(denom1) * math.sqrt(denom2)
    if denom == 0: return 0
    return numer / denom


# In[171]:


correct = 0
for (label,sample) in [(1, readValid), (0, notRead)]:
    for (u,b) in sample:
        maxSim = 0
        users = set(ratingsPerItem[b])
        for b2,_ in ratingsPerUser[u]:
            sim = Jaccard(users,set(ratingsPerItem[b2]))
            if sim > maxSim:
                maxSim = sim
        pred = 0
        if maxSim > 0.013 or len(ratingsPerItem[b]) > 40:
            pred = 1
        if pred == label:
            correct += 1


# In[172]:


#new_method
vector = {}
y = []
for (label,sample) in [(1, readValid), (0, notRead)]:
    for (u,b) in sample:
        maxC, minC, maxC_u = 0, 0, 0
        maxJ, maxJ_u = 0, 0
        maxP, maxP_u = 0, 0
        users = set(ratingsPerItem[b])
        books = set(ratingsPerUser[u])
        for b2,_ in ratingsPerUser[u]:
            Cossim = Cosine(b,b2)
            if Cossim > maxC:
                maxC = Cossim
            if Cossim < maxC:
                minC = Cossim
            sim = Jaccard(users,set(ratingsPerItem[b2]))
            if sim > maxJ:
                maxJ = sim
            #psim = Pearson(b,b2)
            #if psim > maxP:
                #maxP = psim
        
        
        for u2,_ in ratingsPerItem[b]:
            Cossim_u = Cosine_u(u,u2)
            if Cossim_u > maxC_u:
                maxC_u = Cossim_u
            sim_u = Jaccard(books,set(ratingsPerUser[u2]))
            if sim_u > maxJ_u:
                maxJ_u = sim_u
            
                
        vector[(u,b)] = [u , b, maxJ, maxC, len(ratingsPerItem[b])]
        if label == 1:
            y.append(True)
        else:
            y.append(False)


# In[173]:


acc1 = accuracy(predictions, y)
acc1
if v[1] > 0.18   or v[2] > 28 :


# In[ ]:


def tune(t1, t2):
    correct = 0
    s = []
    for (label,sample) in [(1, readValid), (0, notRead)]:
        for (u,b) in sample:
            v = vector[u,b]
            pred = 0
            if v[3] > t1   or v[4] > t2 :
                pred = 1
            if pred == label:
                correct += 1
            s.append([u, b, label, pred, v[3], v[4]])
    acc = correct / (len(readValid) + len(notRead))
    return acc, s


# In[ ]:


acc, s = tune(0.18, 28)


# In[ ]:


#Test
read_len = len([d for d in s if d[3] == 1])
read = [d for d in s if d[3] == 1]
not_read_len = len([d for d in s if d[3] == 0])
not_read = [d for d in s if d[3] == 0]

if read_len > 10000:
    read = sorted(read, key = itemgetter(5))
    l = read_len - 10000
    for i in read[0:l]:
        i[3] = 0
elif not_read_len > 10000:
    not_read = sorted(not_read, key = itemgetter(5), reverse=True)
    l = not_read_len - 10000
    for i in not_read[0:l]:
        i[3] = 1
new_pred = read + not_read

c = 0  
for a, b in zip([d[3] for d in new_pred], [d[2] for d in new_pred]):
    if a == b:
        c+= 1
new_acc = c / len(new_pred)
new_acc


# In[ ]:


predictions = open("predictions_Read.csv", 'w')
s = []
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    maxC = 0
    users = set(ratingsPerItem[b])
    for b2,_ in ratingsPerUser[u]:
        Cossim = Cosine(b,b2)
        if Cossim > maxC:
            maxC = Cossim
    pred = 0
    if maxC > 0.18 or len(ratingsPerItem[b]) > 28:
        pred = 1
    _ = predictions.write(u + ',' + b + ',' + str(pred) + '\n')

predictions.close()


# In[ ]:


##################################################
# Rating prediction (CSE258 only)                #
##################################################


# In[ ]:


betaU = {}
betaI = {}
nUsers = len(ratingsPerUser)
nItems = len(ratingsPerItem)
for u in ratingsPerUser:
    betaU[u] = 0

for b in ratingsPerItem:
    betaI[b] = 0
alpha = globalAverage # Could initialize anywhere, this is a guess


# In[ ]:


users = list(ratingsPerUser.keys())
items = list(ratingsPerItem.keys())
K = 2
userGamma = {}
itemGamma = {}
for u in users:
    userGamma[u] = [random.random() * 0.1 - 0.05 for k in range(K)]
for i in items:
    itemGamma[i] = [random.random() * 0.1 - 0.05 for k in range(K)]


# In[ ]:


def prediction(user, book):
    return alpha + userBiases[user] + itemBiases[book]


# In[ ]:


def iterate(lamb):
    newAlpha = 0
    for u,b,r in ratingsTrain:
        newAlpha += r - (betaU[u] + betaI[b])
    alpha = newAlpha / len(ratingsTrain)
    for u in ratingsPerUser:
        newBetaU = 0
        for b,r in ratingsPerUser[u]:
            newBetaU += r - (alpha + betaI[b])
        betaU[u] = newBetaU / (lamb + len(ratingsPerUser[u]))
    for b in ratingsPerItem:
        newBetaI = 0
        for u,r in ratingsPerItem[b]:
            newBetaI += r - (alpha + betaU[u])
        betaI[b] = newBetaI / (lamb + len(ratingsPerItem[b]))
    mse = 0
    for u,b,r in ratingsTrain:
        prediction = alpha + betaU[u] + betaI[b]
        mse += (r - prediction)**2
    regularizer = 0
    for u in betaU:
        regularizer += betaU[u]**2
    for b in betaI:
        regularizer += betaI[b]**2
    mse /= len(ratingsTrain)
    return mse, mse + lamb*regularizer


# In[ ]:


mse,objective = iterate(1)
newMSE,newObjective = iterate(1)
iterations = 2


# In[ ]:


while iterations < 100 or objective - newObjective > 0.0001:
    mse, objective = newMSE, newObjective
    newMSE, newObjective = iterate(1)
    iterations += 1
    print("Objective after "
        + str(iterations) + " iterations = " + str(newObjective))
    print("MSE after "
        + str(iterations) + " iterations = " + str(newMSE))


# In[ ]:


validMSE = 0
for u,b,r in ratingsValid:
    bu = 0
    bi = 0
    if u in betaU:
        bu = betaU[u]
    if b in betaI:
        bi = betaI[b]
    prediction = alpha + bu + bi
    validMSE += (r - prediction)**2

validMSE /= len(ratingsValid)
print("Validation MSE = " + str(validMSE))


# In[ ]:


iterations = 1
while iterations < 10 or objective - newObjective > 0.0001:
    mse, objective = newMSE, newObjective
    newMSE, newObjective = iterate(4.5)
    iterations += 1
    print("Objective after " + str(iterations) + " iterations = " + str(newObjective))
    print("MSE after " + str(iterations) + " iterations = " + str(newMSE))


# In[ ]:


validMSE = 0
for u,b,r in ratingsValid:
    bu = 0
    bi = 0
    if u in betaU:
        bu = betaU[u]
    if b in betaI:
        bi = betaI[b]
    prediction = alpha + bu + bi
    validMSE += (r - prediction)**2

validMSE /= len(ratingsValid)
print("Validation MSE = " + str(validMSE))


# In[ ]:


predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    bu = 0
    bi = 0
    if u in betaU:
        bu = betaU[u]
    if b in betaI:
        bi = betaI[b]
    _ = predictions.write(u + ',' + b + ',' + str(alpha + bu + bi) + '\n')

predictions.close()


# In[ ]:




