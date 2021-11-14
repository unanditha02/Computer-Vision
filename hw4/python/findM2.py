'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
import submission as sub
import matplotlib.pyplot as plt
import helper

data = np.load('../data/some_corresp.npz')
dataK = np.load('../data/intrinsics.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

x1, y1, _ = im1.shape
x2, y2, _ = im2.shape
K1 = dataK['K1']
K2 = dataK['K2']
M = max(x1, y1)
F = sub.eightpoint(data['pts1'], data['pts2'], M)
E = sub.essentialMatrix(F, K1, K2)
M2s = helper.camera2(E)
M1 = np.concatenate([np.eye(3, 3), np.zeros([3, 1])], axis=1)
C1 = K1 @ M1

for n in range(4):
    M2_1 = M2s[:,:,n]
    C2 = K2 @ M2_1
    P, err = sub.triangulate(C1, data['pts1'], C2, data['pts2'])
    print((np.any(P[:,2]<0)))
    if not (np.any(P[:,2]<0)):
        
        M2 = M2_1
        break
        
print("Error: ",err)
np.savez("q3_3.npz", M2=M2, C2=C2, P=P)
