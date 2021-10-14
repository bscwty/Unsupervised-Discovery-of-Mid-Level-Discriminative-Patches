import numpy as np
from utils import load_patches

folder1 = r'/mnt/data1/output/midpatch/officehome/realworld/centroids/centroids.npy'
folder1 = r'/mnt/data1/output/midpatch/officehome/art/centroids/centroids.npy'

centroids1 = np.load(folder1)
centroids2 = np.load(folder2)

print(centroids1.shape)

print(centroids1)
print(centroids2)
