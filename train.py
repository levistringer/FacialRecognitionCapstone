from PIL import Image
from scipy.linalg import solve
from numpy import *
from pylab import *
import pca
import os
import math
import pickle

IM_DIR = 'eigenFaceData_train'
EIGEN_FACES = 3
imlist = []

for filename in os.listdir(IM_DIR):
    path = os.path.join(IM_DIR,filename)
    imlist.append(path)

classes = []
for name in imlist:
  front = name.split('.')[0]
  front = front[20:]
  classes.append(front)

im = array(Image.open(imlist[0])) # open one image to get size
m,n = im.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

# create matrix to store all flattened images
immatrix = array([array(Image.open(im)).flatten()
              for im in imlist],'f')

# perform PCA
V,S,immean = pca.pca(immatrix)

transformed = dot(V[:EIGEN_FACES],immatrix.T).T

trainedData = []
for i in range(len(transformed)):
  trainedData.append([classes[i],transformed[i]])

with open("faces.txt", "wb") as fp:   #Pickling
  pickle.dump(trainedData, fp)

with open("model.txt", "wb") as fp:   #Pickling
  pickle.dump(V[:EIGEN_FACES], fp)
  

