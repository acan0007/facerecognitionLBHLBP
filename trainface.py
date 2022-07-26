import cv2
import numpy as np

#import lbp cascade classifier for training data 

cascade_classifier = cv2.CascadeClassifier('C:/Users/ASUS/Desktop/xml/lbpcascade_frontalface_improved.xml')

#function detect the face

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(gray, scaleFactor=1.2,minNeighbors=5)

    if faces is ():
        return None
    
    for (x,y,w,h) in faces:
        crop = img[y:y+h, x:x+w]
    
    return crop #return the cropped face img in the set faces 

cap = cv2.VideoCapture(0)
count = 0
while True:
    ret, frame = cap.read()
    if detect_face(frame) is not None:
        count += 1

        #convert the img into a grayscale and resize it
        face = cv2.resize(detect_face(frame), (300, 300))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        #write all the count of face train set into directory
        path = 'C:/Users/ASUS/Desktop/Mini Project MV/train-data/'+str(count)+'.jpg'
        cv2.imwrite(path, face)
        
        cv2.putText(face, "Count: " + str(count), (50,150),cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 1)
        cv2.imshow('Trainface', face)

    else:
        #if face not found prints an error warning

        print("None faces is found!")
        pass

    if cv2.waitKey(20) & 0xFF == ord('q') or count == 200: #after collecting of 100 samples or can be pressed q = exit
        break

print("collecting samples, please wait..")
print("Complete.")

cap.release()
cv2.destroyAllWindows


