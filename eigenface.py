from PIL import Image
from scipy.linalg import solve
from numpy import *
from pylab import *
import pca
import os
import math
import pickle

def calculateDistance(xlist,ylist):  
  total = 0
  for i in range(len(xlist)):
    total = total + (xlist[i]-ylist[i])**2
  dist = math.sqrt(total)  
  return dist

with open("model.txt", "rb") as fp:   # Unpickling
    V = pickle.load(fp)

with open("faces.txt", "rb") as fp:   # Unpickling
    trainedData = pickle.load(fp)

IM_DIR = 'eigenFaceData_test'
imlist = []

for filename in os.listdir(IM_DIR):
    path = os.path.join(IM_DIR,filename)
    imlist.append(path)

classes = []
for name in imlist:
  front = name.split('.')[0]
  front = front[19:]
  classes.append(front)

im = array(Image.open(imlist[0])) # open one image to get size
m,n = im.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

# create matrix to store all flattened images
immatrix = array([array(Image.open(im)).flatten()
              for im in imlist],'f')

transformed = dot(V,immatrix.T).T

truth = 0
count = 0
for i in range(len(transformed)):
  min = 0
  value = float('inf')
  for j in range(len(trainedData)):
    if calculateDistance(trainedData[j][1],transformed[i]) < value:
      value = calculateDistance(trainedData[j][1],transformed[i])
      min = j
  if trainedData[min][0] == classes[i]:
    truth = truth + 1
  count = count + 1
print(truth/count)
