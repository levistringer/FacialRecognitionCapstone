import cv2
import os 

imageDIR = 'data/dataset_1'
newDIR = 'data/dataset_1_resized'
files = os.listdir(imageDIR)
files.sort()
for filename in files:
    personTags = filename.split('.')
    if len(personTags) == 3:
        personTag = int(personTags[1])
        path = os.path.join(imageDIR,filename)
        im = cv2.imread(path,0)
        im = cv2.resize(im,(100,100))
        #personClass = personTags[0]
        newPath = os.path.join(newDIR, personTags[0],personTags[1] + '.' + personTags[2])
        if not os.path.exists(newDIR + '/' + personTags[0]): #Checks if person folder exists, if not makes a person folder
            os.mkdir(newDIR + '/' + personTags[0])
        cv2.imwrite(newPath,im)
        