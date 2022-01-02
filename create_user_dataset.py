import cv2
import mediapipe as mp

def create_dataset(file_size = 1):
    print("start run.py")

    IMG_SIZE = (200, 200)
    faceModule = mp.solutions.face_mesh

    cap = cv2.VideoCapture(0)

    with faceModule.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face:
        while cap.isOpened():
            ret, img = cap.read()
            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            shape = img.shape
            try:
                if result.multi_face_landmarks:
                    for res in result.multi_face_landmarks:
                        rectangle_data_l = [int(res.landmark[123].x * shape[1]), int(res.landmark[352].x * shape[1]), int(res.landmark[10].y * shape[0]), int(res.landmark[152].y * shape[0])]
                        face_img = img[rectangle_data_l[2]:rectangle_data_l[3], rectangle_data_l[0]:rectangle_data_l[1]]
                        face_img = cv2.resize(face_img, dsize=IMG_SIZE)
                        cv2.imshow('face_img', face_img)

                cv2.putText(img, "Face Data Save", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imwrite(f"faces/{file_size}.png", face_img)
                file_size += 1
                print(file_size)
            except:
                cv2.putText(img, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('img', img)
            check = cv2.waitKey(1)
            if check == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break