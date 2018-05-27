import cv2
import numpy as np
from PIL import Image
import os  #accessing the filesystem
# Path for face image database
path = 'C:\Users\Youness Haraki\Documents\Python\Face_recog\Data_Training\DataSet'
recognizer = cv2.face.LBPHFaceRecognizer_create()
faceCascade = cv2.CascadeClassifier('C:/Users/Youness Haraki/Documents/Python/Face_recog/haarcascade_frontalface_default.xml')
# function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')  #convert a PIL Image into a numpy array
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = faceCascade.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
# Save the model into trainer/trainer.yml
recognizer.write('C:/Users/Youness Haraki/Documents/Python/Face_recog/Data_Training/Trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi
# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
#if too many same id are sorted it makes the as one
