import matplotlib.pyplot as plt
import numpy as np

BF = []
BF_GLARE = []
SNIG = []
SNIG_GLARE = []
XY = []
XY_GLARE = []
SNICIT = []
SNICIT_GLARE = []

with open('../log/SDGC/BF-1024.txt') as f:
    lines = f.readlines()
    for line in lines:
        arr = line.split(' ')
        BF.append(float(arr[1]))

with open('../log/SDGC/BF_GLARE-1024.txt') as f:
    lines = f.readlines()
    for line in lines:
        arr = line.split(' ')
        BF_GLARE.append(float(arr[1]))

with open('../log/SDGC/SNIG-1024.txt') as f:
    lines = f.readlines()
    for line in lines:
        arr = line.split(' ')
        SNIG.append(float(arr[1]))

with open('../log/SDGC/SNIG_GLARE-1024.txt') as f:
    lines = f.readlines()
    for line in lines:
        arr = line.split(' ')
        SNIG_GLARE.append(float(arr[1]))

with open('../log/SDGC/XY-1024.txt') as f:
    lines = f.readlines()
    for line in lines:
        arr = line.split(' ')
        XY.append(float(arr[1]))

with open('../log/SDGC/XY_GLARE-1024.txt') as f:
    lines = f.readlines()
    for line in lines:
        arr = line.split(' ')
        XY_GLARE.append(float(arr[1]))

with open('../log/SDGC/SNICIT-1024.txt') as f:
    lines = f.readlines()
    for line in lines:
        arr = line.split(' ')
        SNICIT.append(float(arr[1]))

with open('../log/SDGC/SNICIT_GLARE-1024.txt') as f:
    lines = f.readlines()
    for line in lines:
        arr = line.split(' ')
        SNICIT_GLARE.append(float(arr[1]))

plt.figure(figsize=(12, 8))
layer1 = np.linspace(0, 120, 120)
layer2 = np.linspace(22, 120, 120-22)
layer3 = np.linspace(30, 120, 120-30)

p1 = plt.plot(layer1, (np.array(BF)-np.array(BF_GLARE))/np.array(BF), linewidth = 5)
p2 = plt.plot(layer1, (np.array(SNIG)-np.array(SNIG_GLARE))/np.array(SNIG), '-.', linewidth = 5)
p3 = plt.plot(layer2, (np.array(XY)-np.array(XY_GLARE))/np.array(XY), ':', linewidth = 5)
p4 = plt.plot(layer3, (np.array(SNICIT)-np.array(SNICIT_GLARE))/np.array(SNICIT),'--', linewidth = 5)

plt.legend(['BF', 'SNIG', 'XY', 'SNICIT'], fontsize=30) #, 'XY', 'SNICIT'
plt.xlabel('Layer', fontsize=24)
plt.ylabel('Reduction rate', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.grid()
# fig = plt.figure()
# fig.subplots_adjust(right = 0.3)
# plt.show()
plt.savefig("figs/exp.pdf")