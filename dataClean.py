import cv2
import os 

imageDIR = 'faces-data-new/images'
newDIR = 'eigenFaceData_train'
newerDIR = 'eigenFaceData_test'

MAX_PEOPLE = 50

count = 1
flag = True

for filename in os.listdir(imageDIR):
    if count > MAX_PEOPLE:
        break
    personTags = filename.split('.')
    if len(personTags) == 3:
        personTag = int(personTags[1])
        if personTag in [1,10,11,12,13,14,15,16,17,18]:
            flag = True
            path = os.path.join(imageDIR,filename)
            im = cv2.imread(path,0)
            im = cv2.resize(im,(100,100))
            newPath = os.path.join(newDIR,filename)
            cv2.imwrite(newPath,im)
        elif personTag in [19]:
            flag = True
            path = os.path.join(imageDIR,filename)
            im = cv2.imread(path,0)
            im = cv2.resize(im,(100,100))
            newPath = os.path.join(newerDIR,filename)
            cv2.imwrite(newPath,im)
        else:
            if flag == True:
                flag = False
                count = count + 1

