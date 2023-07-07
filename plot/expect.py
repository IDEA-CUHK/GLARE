import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb

zerows = []
thirtytwos = []


def E1(m, n, arr4, L):
    E_sum = 0
    for i in range(len(arr4)-1):
        sub_sum = 0
        for k in range(int(n/m)+1):
            sub_sum += comb(int(n/m), k)*(1-arr4[i]**m)**k*arr4[i]**(n-m*k)*k
            sub_sum += comb(int(n/m), k)*(1-arr4[i+1]**m)**k*arr4[i+1]**(n-m*k)*k
        E_sum += (2*int(n/m)+m*sub_sum)
    for i in range(120, L):
        E_sum += 2*int(n/m)
    return E_sum

def E2(m, n, arr4, L):
    E_sum = 0
    for i in range(len(arr4)-1):
        sub_sum = 0
        for k in range(int(n/m)+1):
            sub_sum += comb(int(n/m), k)*(1-arr4[i]**m)**k*arr4[i]**(n-m*k)*k
            sub_sum += (1-arr4[i+1]**m)
        E_sum += (1+int(n/m)+m*sub_sum)
    for i in range(120, L):
        E_sum += 1+int(n/m)
    return E_sum

with open('../log/SDGC/Row0-1024.txt') as f:
    lines = f.readlines()
    for line in lines:
        arr = line.split(' ')
        zerows.append(float(arr[1]))

with open('../log/SDGC/Num32-1024.txt') as f:
    lines = f.readlines()
    for line in lines:
        arr = line.split(' ')
        thirtytwos.append(float(arr[1]))

layer = np.linspace(0, 120, 120)
zerows = np.array(zerows) / (60000)
thirtytwos = np.array(thirtytwos) / (1024*60000)
arr4 = np.array(thirtytwos / (1-zerows))

n = 65536
mlist0 = [64, 128, 256, 512, 1024, 2048, 4096, 4096*2, 16384, 16384*2, 65536]

mlist = []

for m in mlist0:
    if m <= n and m >= min(n, 8096):
        mlist.append(m)

Elist1 = []
L = 480
for m in mlist:
    Elist1.append(E1(m, n, arr4, L))
Elist2 = []
for m in mlist:
    Elist2.append(E2(m, n, arr4, L))
print(Elist1)
print(Elist2)

