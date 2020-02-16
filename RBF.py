import sklearn.cluster as sk
import numpy as np
import math 
from numpy.linalg import inv
import matplotlib.pyplot as plt
def DistanceCalculator(Centers,n):
    D=0
    for i in range (n):
        for j in range (n):
            Distance=np.abs(Centers[i]-Centers[j])
            if(Distance>D):
                D=Distance
    sig=D/(n*np.sqrt(2))
    return sig        
X = np.linspace(-3,3,100)
X = X.reshape((100,1))
#print(X.shape)
#print (X)
#X = 

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

n = 4
b = np.ones((100,1))
kmeans = sk.KMeans(n_clusters=n , random_state=0).fit(X)
centers = kmeans.cluster_centers_
#print(centers)
#self.M = self.centers.shape(0)
#
#self.dm = max([abs(self.centers[i]-self.centers]])
sig = DistanceCalculator(centers,n)
GG = np.zeros((5,1))
for i in range(100):
    f1 = gaussian(X[i,0],centers[0],sig)
    f2 = gaussian(X[i,0],centers[1],sig)
    f3 = gaussian(X[i,0],centers[2],sig)
    f4 = gaussian(X[i,0],centers[3],sig)
    G= np.array(([1],f1,f2,f3,f4))
    GG = np.concatenate((GG,G), axis=1)
    
GG = GG[0:5,1:101]    

#aa = np.ones((5, 1))
b = np.sin(X)
GG = GG.T
b1 = b + np.random.normal(0, 0.1 , b.shape)
#A = np.concatenate((aa,GG), axis = 1)
A = GG
print('salam',A.shape)
W = np.dot(np.dot(inv(np.dot(A.T, A)), A.T),b1)
#print(W)
        
def predict(x):
    f1 = gaussian(x,centers[0],sig)
    f2 = gaussian(x,centers[1],sig)
    f3 = gaussian(x,centers[2],sig)
    f4 = gaussian(x,centers[3],sig)
    G= np.array(([1],f1,f2,f3,f4))
    return float(np.dot(W.T , G))
print(predict(1.7))   
plt.figure()
#plt.y_lim[-1.1]

XX = np.linspace(-3,3,600)
yy=[]
for i in range(600):
  yy.append(predict(XX[i]))


plt.plot(XX,yy,'ro',X,b1,'bx') 
#print(yy)   
plt.show()
    