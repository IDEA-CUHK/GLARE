import matplotlib.pyplot as plt
import numpy as np

zeros = []
zerows = []
thirtytwos = []

with open('../log/SDGC/Num0-1024.txt') as f:
    lines = f.readlines()
    for line in lines:
        arr = line.split(' ')
        zeros.append(float(arr[1]))

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
zeros = np.array(zeros) / (1024*60000)
zerows = np.array(zerows) / (60000)
thirtytwos = np.array(thirtytwos) / (1024*60000)
arr4 = np.array(thirtytwos / (1-zerows))
plt.figure(figsize=(7, 6))
p1 = plt.plot(layer, zeros, linewidth = 3)
p2 = plt.plot(layer, zerows, '-.',linewidth = 3)
p3 = plt.plot(layer, thirtytwos, ':', linewidth = 3)
p4 = plt.plot(layer, arr4, '--',linewidth = 3)
plt.legend(['0s', 'all-zero entries', '32s', '32 in non-zero entries'], fontsize=22)
plt.xlabel('Layer', fontsize=19)
plt.ylabel('Ratio', fontsize=19)
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
plt.grid()
# plt.show()
plt.savefig("figs/ratios.pdf")