import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

#get train data from training model

path = 'C:/Users/ASUS/Desktop/Mini Project MV/train-data/'
file = [f for f in listdir(path) if isfile(join(path, f))]

#generate training data and its label

training_data, labels = [], []

for i, files in enumerate(file):
    train_image_path = path + file[i]
    images = cv2.imread(train_image_path, cv2.IMREAD_GRAYSCALE)
    training_data.append(np.asarray(images, dtype=np.uint8))
    labels.append(i)

labels = np.asarray(labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(training_data), np.asarray(labels))

print("model training complete")

cascade_classifier = cv2.CascadeClassifier('C:/Users/ASUS/Desktop/xml/lbpcascade_frontalface_improved.xml')

def detect(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(gray, 1.2 , 5)

    if faces is():
        return img,[]
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (300, 300))
    
    return img, roi

accuracy = []
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    image, face = detect(frame)

    try:
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        res = model.predict(face)

    #get the accuracy numbers
        if res[1] < 500:
            confidence = float("{0:.2f}".format((100*(1-(res[1])/300))))
            accuracy.append(confidence)
            dis = str(confidence) + '% Match!'

        cv2.putText(image, dis, (100,120), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 0, 128), 2)

        conf = 80

        if confidence > conf:
            cv2.putText(image, "Face is match!", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face', image)
        else:
            cv2.putText(image, "Face doesn't match!", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face', image)

    except:
        #if face is not detected
        cv2.putText(image, "Cannot Detect Face!", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face', image)
        pass
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

print("The Highest Accuracy that were generated is " + str((np.max(accuracy)))) 

plt.plot(accuracy)
plt.ylabel("Accuracy")
plt.xlabel("Run Time")
plt.title("Confidence Plot Graph")
plt.show()

cap.release()
cv2.destroyAllWindows()
