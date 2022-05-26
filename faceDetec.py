import cv2
from cv2 import CascadeClassifier


# image = cv2.imread('images/1.jpg')
# g_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def faceDetection(image):
    g_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faceCascade = CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

    # The function that detects the faces in the image
    faces = faceCascade.detectMultiScale(
        g_image,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE )


    # This will draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return len(faces),image     

# num,img=faceDetection(image)
# img = cv2.resize(img, (1000,600))
# print(str(num)+ " faces are detected")
# cv2.imshow("Face Detection", img)
# cv2.waitKey(0)