from PIL import Image
from scipy.linalg import solve
from numpy import *
from pylab import *
import pca
import os
import math
import pickle
import numpy as np
import collections
import matplotlib
import matplotlib.pyplot as plt

def calculateDistance(instance1, instance2):
  instance1 = np.array(instance1) 
  instance2 = np.array(instance2)
  return np.linalg.norm(instance1 - instance2)

def vote_distance_weights(neighbors):
  class_counter = collections.Counter()
  number_of_neighbors = len(neighbors)
  for index in range(number_of_neighbors):
    dist = neighbors[index][1]
    label = neighbors[index][2]
    class_counter[label] += 1 / (dist**2 + 1)
  winner = class_counter.most_common(1)[0][0]
  return winner
  
def getNeighbours(trainingData, testData, num_neighbours):
  distances = []

  for index in range(len(trainingData)):
    dist = calculateDistance(testData, trainingData[index][1])
    distances.append((trainingData[index][1], dist,trainingData[index][0]))
    
  distances.sort(key= lambda x: x[1])
  neighbours = distances[:num_neighbours]
  return neighbours

def eigenface(k_neighbours = 6):
  with open("model.txt", "rb") as fp:   # Unpickling
    V = pickle.load(fp)

  with open("faces.txt", "rb") as fp:   # Unpickling
    trainedData = pickle.load(fp)
  
  IM_DIR = 'eigenFaceData_test'
  imlist = []

  for filename in os.listdir(IM_DIR):
      if '.DS_Store' in filename:
        continue
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
  immatrix = array([array(Image.open(im)).flatten() for im in imlist],'f')

  transformed = dot(V,immatrix.T).T
  neighboursDistance = [] 
  truth = 0
  for i in range(len(transformed)):
    neigh = getNeighbours(trainedData,transformed[i], k_neighbours)
    predicted = vote_distance_weights(neigh)
    if(predicted == classes[i]):
      truth +=1
  return truth/len(transformed)

# def varyNeighbours():
#   kVal = list(range(1,25))
#   accuracy = []
#   for k in kVal:
#     acc = eigenface(k)
#     accuracy.append(acc)
#     print(str(k) + ' and '+ str(acc))
#   plt.plot(kVal, accuracy, color='g')
#   plt.xlabel('K Neighbouts')
#   plt.ylabel('Accuracy')
#   plt.show()

