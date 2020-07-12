

"""""""""""""""""""""""""""""""""""""""""""""IMPORTING THE REQUIREMENTS"""""""""""""""""""""""""""""""""""""""""""""

from PIL import Image
import cv2
from keras.models import load_model
import numpy as np
import collections

""""""""""""""""""""""""""""""""""""""""""""""""""" FACE DETECTION """""""""""""""""""""""""""""""""""""""""""""""""""

model = load_model('D:\Object_detection\model.h5')  # LOADING THE MODEL

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # FACE_CASCADE FOR CROPPING FACES


def face_extractor(img):

    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return None

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cropped_face = img[y:y + h, x:x + w]

    return cropped_face


def detection(classes):

    cap = cv2.VideoCapture('D:\Object_detection\detections.mp4')
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                  cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('D:\Object_detection/final_detections.avi', fourcc, fps,
                          (int(frame_size[0]), int(frame_size[1])))
    while True:
        _, frame = cap.read()
        face = face_extractor(frame)
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        if type(face) is np.ndarray:
            face = cv2.resize(face, (160, 160))  # INPUT SHAPE FOR THE MODEL
            im = Image.fromarray(face, 'RGB')
            img_array = np.array(im)
            img_array = np.expand_dims(img_array, axis=0)
            pred = model.predict(img_array)
            value = np.argmax(pred)
            max = np.max(pred)
            print(value)
            print(max)

            lists = []
            print(classes)
            for values in collections.OrderedDict(classes):
                print(values)
                lists.append(values)

            name = "None matching"

            if (max > 0.6):  # PREDICTION THRESHOLD
                name = lists[int(value)]
            for (x, y, w, h) in faces:
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        else:
            for (x, y, w, h) in faces:
                cv2.putText(frame, "No face found", (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

        cv2.imshow('IDENTIFYING FACES.', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out.write(frame)

    cap.release()
    cv2.destroyAllWindows()
