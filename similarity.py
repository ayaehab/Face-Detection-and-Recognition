import numpy as np
import math 
from FaceRecog import FaceRecognizer

def similarity(self, zeroMean_trainImages, subtracted_testImage, percent, testImage_name):
        threshold = 3000
        label = 0
        distances=[]
        #  start distance with 0
        smallestDistance = 0
        #  iterate over all image vectors until fit input image
        for i in range(len(self.total_images)):
            # projecting each image in face space
            trainImage_wights = self.EigenFaces.dot(zeroMean_trainImages[i])
            print(trainImage_wights.shape)
            # calculating euclidean distance between vectors
            feautersDiffrence = subtracted_testImage[:int(percent*len(self.total_images))] - trainImage_wights[:int(percent*len(self.total_images))]
            euclidean_distances = math.sqrt(feautersDiffrence.dot(feautersDiffrence))
            distances.append(euclidean_distances)
        # get smallest distance index
        minDistance_index = np.argmin(distances)    
        # comparing smallest distance with threshold
        if minDistance_index[min] < threshold:
            print("the input image fit :", self.total_images[minDistance_index])
        else:
            print("unknown Face")

        #########################################################################################################
        testImage_name='Face_Recognition_Dataset/test/s1/1.pgm'
        splitName = testImage_name.split("/")
        if splitName[1] in self.classesNames:
