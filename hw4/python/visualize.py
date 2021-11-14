'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper

# read data 
data = np.load('../data/some_corresp.npz')
dataK = np.load('../data/intrinsics.npz')
dataPts = np.load('../data/templeCoords.npz')

im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

im1w, im1h, _ = im1.shape
M = max(im1w, im1h)

corres1 = data['pts1'] # for eight point
corres2 = data['pts2']

x1 = dataPts['x1'] # for eppipolar correspondences
y1 = dataPts['y1']
totalCorres = len(x1)


K1 = dataK['K1']
K2 = dataK['K2']

# camera 1 M1, C1
M1 = np.concatenate([np.eye(3, 3), np.zeros([3, 1])], axis=1)
C1 = K1 @ M1


F = sub.eightpoint(corres1, corres2, M)
E = sub.essentialMatrix(F, K1, K2)

x2 = np.zeros(shape=(totalCorres,1))
y2 = np.zeros(shape=(totalCorres,1))

for i in range(totalCorres):
    x2[i], y2[i] = sub.epipolarCorrespondence(im1, im2, F, x1[i].item(), y1[i].item())

pts1 = np.concatenate((x1, y1), axis=1)
pts2 = np.concatenate((x2, y2), axis=1)

# camera 2 M2, C2
M2s = helper.camera2(E)
for n in range(4):
    M2_1 = M2s[:,:,n]
    C2 = K2 @ M2_1
    P, err = sub.triangulate(C1, pts1, C2, pts2)
    print((np.any(P[:,2]<0)))
    if not (np.any(P[:,2]<0)):
        M2 = M2_1
        print("Error: ", err)
        break
print('F: ', F)

np.savez('q4_2.npz', F=F, M1=M1, M2=M2, C1=C1, C2=C2)
fig = plt.figure()
ax = plt.axes(projection ="3d")
 
# Creating plot
ax.scatter3D(P[:,0], P[:,1], P[:,2])
plt.title("Temple - 3D Visualization")
 
# show plot
plt.show()

