import matplotlib.pyplot as plt
import numpy as np

BF = []
BF_GLARE = []
SNIG = []
SNIG_GLARE = []

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