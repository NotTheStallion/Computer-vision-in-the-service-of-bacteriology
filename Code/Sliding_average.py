import numpy as np
import matplotlib.pyplot as plt

def MoyGlissante(x,n):
    if len(x)<n:
        return str(n)+"est une valeur invalide"
    L=np.ones(len(x))
    s=0
    for i in range(len(x)):
        if i<n-1:
            for j in range(i+1):
                s+=x[j]
            L=np.append(L,s/(i+1))
            s=0
        else:
            for j in range(i-n-1,i+1):
                s+=x[j]
            L=np.append(L,s/n)
            s=0
    return L

x=np.array([ 1., 1.,  1.,  1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1.,  1.,  1,  1.,1.,  1.,  1.,  1.,  1.,  1.,  1.,  1. , 1. , 1. , 1. , 1.,  1.,  1. , 1. , 1. , 1., -0.,-1. , 3.,  3.,  3.,  1.,  0. , 1. , 0. , 1. , 0.,  0., -1.,  0. , 0. , 0. ,-1. , 3. ,-1.,3. , 3. , 3.,  3.,  1.,  1.,  1., -0., -1. ,-1. , 3., -1., -0., -0., -0., -0.])


plt.plot(MoyGlissante(x,5))
plt.plot(MoyGlissante(x,10))
plt.plot(MoyGlissante(x,15))
plt.plot(x)
plt.show()
