import cv2
import numpy as np
import mediapipe as mp
import glob
import create_user_dataset
import os

def create_model(image_path):
    Training_Data, Labels = [], []
    for i, files in enumerate(image_path):
        images = cv2.imread(image_path[i], cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)

    Labels = np.asarray(Labels, dtype=np.int32)
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    return model

def face_detector(img):
    with faceModule.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        shape = img.shape

        if result.multi_face_landmarks:
            for res in result.multi_face_landmarks:
                rectangle_data_l = [int(res.landmark[123].x * shape[1]), int(res.landmark[352].x * shape[1]),
                                    int(res.landmark[10].y * shape[0]), int(res.landmark[152].y * shape[0])]
                cv2.rectangle(img, (rectangle_data_l[0]-10, rectangle_data_l[2] - 10), (rectangle_data_l[1] + 10, rectangle_data_l[3] + 10 ), color=(255, 0, 0), thickness=2)

                face_img = img[rectangle_data_l[2]:rectangle_data_l[3], rectangle_data_l[0]:rectangle_data_l[1]]
                face_img = cv2.resize(face_img, dsize=IMG_SIZE)
    return face_img

image_path = glob.glob("faces/*.png")
if len(image_path) == 0:
    create_user_dataset.create_dataset()
    image_path = glob.glob("faces/*.png")

model = create_model(image_path)

print("Model Training Complete!!!!!")

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
IMG_SIZE = (200, 200)
faceModule = mp.solutions.face_mesh
face_check = 0
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    try:
        face = face_detector(frame)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300))
            display_string = str(confidence)+'% Confidence it is user'
        cv2.putText(frame,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)

        if confidence > 75:
            cv2.putText(frame, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', frame)
            face_check = 1
        else:
            cv2.putText(frame, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', frame)
            face_check = 0
    except:
        cv2.putText(frame, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', frame)
        face_check = 0
        pass

    check = cv2.waitKey(1)
    if check == ord('r'):
        cap.release()
        cv2.destroyAllWindows()
        for i in image_path:
            os.remove(i)
        create_user_dataset.create_dataset()
        image_path = glob.glob("faces/*.png")
        model = create_model(image_path)
        cap = cv2.VideoCapture(0)
    if check == ord('a'):
        cap.release()
        cv2.destroyAllWindows()
        create_user_dataset.create_dataset(len(image_path))
        image_path = glob.glob("faces/*.png")
        model = create_model(image_path)
        cap = cv2.VideoCapture(0)
    if check == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
