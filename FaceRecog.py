from pydoc import classname
import numpy as np
import glob
import cv2
import time
from sklearn import preprocessing
import math
import os
from matplotlib import pyplot as plt



class FaceRecognizer:
    def __init__(self):
        self.dataset_path = 'images2/train'                   
        self.img_shape = (100,100)                       
        self.classes_num = 0                         
        self.all_images_Matrix = np.array([])                       
        self.names = list()                          
        self.mean_vector = None                    
        self.A_tilde = None                        
        self.eigenfaces = np.array([])
        self.mean_subtracted_test_img = None                                        
        self.classesNames = list()
        self.total_imagesNo=0

    def create_image_matrix(self):
        """ Retrieve all files in a dir into a list """
        self.imgs = os.listdir(self.dataset_path)
        self.total_imagesNo = len(self.imgs)
        face_matrix = []
        for i in range(self.total_imagesNo):
            # read img as greyscale
            img = cv2.imread(self.dataset_path + "/" + self.imgs[i], cv2.IMREAD_GRAYSCALE)
            # resize img
            img = cv2.resize(img, (100, 100))
            # convert img into vector
            shape = img.shape[0] * img.shape[1]
            img = np.reshape(img, shape)
            # add to face matrix
            face_matrix.append(img)
        self.all_images_Matrix = np.array(face_matrix)
        # print(self.all_images_Matrix.shape)
        


    def fit(self,  threshold=0.9):
        
        self.mean_matrix = np.mean(self.all_images_Matrix, axis=0)
        self.A_tilde = np.subtract(self.all_images_Matrix, self.mean_matrix)

        L = (self.A_tilde.dot(self.A_tilde.T)) / (self.total_imagesNo-1)

        eigenvalues, eigenvectors = np.linalg.eig(L)
        
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        eigenvectors_c = self.A_tilde.T @ eigenvectors
        self.eigenfaces = preprocessing.normalize(eigenvectors_c.T,axis=1)
        
        commulated_variance = np.cumsum(idx)/sum(idx)
        self.eigenfaces_num = len(commulated_variance[commulated_variance<=threshold])

        return self.eigenfaces_num



    def project_newFace_oneigenfaces(self,img_path):      

        test_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        test_img = cv2.resize(test_img, (self.img_shape[1], self.img_shape[0]))

        # subtract the mean
        self.mean_subtracted_test_img = np.reshape(test_img, (test_img.shape[0] * test_img.shape[1])) - self.mean_matrix
        # the vector that represents the image with respect to the eigenfaces.
        w = self.eigenfaces.dot(self.mean_subtracted_test_img) #weights of testing image

        return  w


    def similarity(self, subtracted_testImage, percent=0.9):
        threshold = 3000
        distances=[]
        #  iterate over all image vectors until fit input image
        for i in range(self.total_imagesNo):
            # projecting each image in face space
            trainImage_wights = self.eigenfaces[:int(percent*self.total_imagesNo)].dot(self.A_tilde[i])
            # print(trainImage_wights.shape)
            # calculating euclidean distance between vectors
            feautersDiffrence = subtracted_testImage[:int(percent*self.total_imagesNo)] - trainImage_wights[:int(percent*self.total_imagesNo)]
            euclidean_distances = math.sqrt(feautersDiffrence.dot(feautersDiffrence))
            distances.append(euclidean_distances)
        self.minDistance_index = np.argmin(distances)
        # comparing smallest distance with threshold
        if distances[self.minDistance_index] < threshold:
            # print("the input image similar to this image from train_dataset :", self.imgs[self.minDistance_index])
            txt = "the input image similar to this image from train_dataset :"+str(self.imgs[self.minDistance_index]) 
            label=1
            return self.imgs[self.minDistance_index], label, txt
        else:
            # print("unknown Face")
            txt=''
            label=0
            return "unknown Face", label,txt


        #########################################################################################################


    def ROC(self, folder_path,percent):
        imgs = os.listdir(folder_path)
        output_images =[]
        labels = []
        self.true_positive  = 0
        self.false_negative = 0
        self.false_positive = 0
        self.true_negative  = 0
        for img in imgs:
            # print(img)
            test_img = cv2.imread(str(folder_path) + '/' + str(img) , cv2.IMREAD_GRAYSCALE)
            test_img = cv2.resize(test_img, (self.img_shape[0], self.img_shape[1]))
            vector_of_mean_subtracted_test_img = self.project_newFace_oneigenfaces(str(folder_path) + '/' + str(img))
            result, label, _ = self.similarity(vector_of_mean_subtracted_test_img,percent)
            # print(result)
            output_images.append(result)
            labels.append(label)
                        
        roc_values=self.compare(imgs, output_images, labels)
        return roc_values

    def compare(self, test_imgs, matched_imgs, labels):
        print (test_imgs, matched_imgs, labels)
        roc=[]
        names=['yaleB01', 'yaleB02', 'yaleB03', 'yaleB04', 'yaleB05','yaleB06','yaleB07','yaleB08']
        for i in range(len(test_imgs)):
            testName_split = test_imgs[i].split("_")
            matchedName_split = matched_imgs[i].split("_")
            print(testName_split, matchedName_split)
            # print(labels)
            if testName_split[0] in names:
                if testName_split[0] == matchedName_split[0]:
                    self.true_positive +=1
                else:
                    self.false_negative +=1
            else :
                if labels[i] == 1:
                    self.false_positive +=1
                else:
                    self.true_negative +=1
        print(self.true_negative, self.true_positive, self.false_negative, self.false_positive)
        true_positive_rate= self.true_positive / (self.true_positive + self.false_negative)
        false_positive_rate = self.true_negative / (self.true_negative + self.false_positive)
        roc.extend([true_positive_rate,false_positive_rate])
        return roc
    
    def ROC_plot(self,result):
        y = [result[0],result[2],result[4],result[6],result[8]]
        x = [result[1], result[3],result[5],result[7],result[9]]
        plt.plot(x, y)
        # naming the x axis
        plt.xlabel('x - axis')
        # naming the y axis
        plt.ylabel('y - axis')
        # giving a title to my graph
        plt.title('roc')
        # function to show the plot
        plt.savefig("roc.png")
    
    def total_ROC(self,pathFile):
        thresholds=[0.9, 0.7,0.5,0.1,0.001,0.0001,0.00002]
        points=[]
        for i in thresholds:
            res = self.ROC(pathFile, i)
            points.append(res)
        points=np.array(points)
        self.ROC_plot(points.flatten())




# rec = FaceRecognizer()
# rec.create_image_matrix()
# rec.fit()
# w = rec.project_newFace_oneigenfaces('images2/test/yaleB01_ (1).jpg')

# matching_image = rec.similarity(w, 0.9)
# rec.total_ROC('images2/test')

# im = cv2.imread('images2/test/Cartoon_Mickey.jpg')
# print('images2/train/'+str(matching_image))
# # im2=cv2.imread('images2/train/'+str(matching_image))
# plt.subplot(1,2,1)
# plt.imshow(im)
# plt.title('tested image')
# # plt.subplot(1,2,2)
# # plt.imshow(im2)
# # plt.title('predicted image')
# plt.show()


