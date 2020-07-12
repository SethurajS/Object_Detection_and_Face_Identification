
"""""""""""""""""""""""""""""""""""""""""""""IMPORTING THE REQUIREMENTS"""""""""""""""""""""""""""""""""""""""""""""

import cv2
import os

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


"""""""""""""""""""""""""""""""""""""""""""CROPPING FACES FROM THE IMAGES"""""""""""""""""""""""""""""""""""""""""""

def face_extractor(img):

    faces = face_classifier.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return None

    for (x, y, w, h) in faces:
        x = x - 10
        y = y - 10
        cropped_face = img[y:y + h + 50, x:x + w + 50]

    return cropped_face


""""""""""""""""""""""""""""""""""""""""CAPTURING TRAINING INPUTS FOR THE MODEL"""""""""""""""""""""""""""""""""""""""


def capturing_training_images(name):

    cap = cv2.VideoCapture(0)
    count = 0

    if not os.path.exists('D:\Object_detection\Datasets\Train/'+name):
        os.mkdir('D:\Object_detection\Datasets\Train/'+name)  # MAKING DIRECTORY FOR TRAINING DATA

    while True:
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count += 1
            face = cv2.resize(face_extractor(frame), (400, 400))

            file_name_path = 'D:\Object_detection\Datasets\Train/'+name+'/'+name + str(count) + '.jpg'
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', face)
        else:
            print("Face not found")
            pass

        if cv2.waitKey(1) == 13 or count == 200:
            break

    cap.release()
    cv2.destroyAllWindows()


""""""""""""""""""""""""""""""""""""""""CAPTURING TESTING INPUTS FOR THE MODEL"""""""""""""""""""""""""""""""""""""""


def capturing_testing_images(name):

    cap = cv2.VideoCapture(0)
    count = 0
    if not os.path.exists('D:\Object_detection\Datasets\Test/'+name):
        os.mkdir('D:\Object_detection\Datasets\Test/'+name)  # MAKING DIRECTORY FOR TESTING DATA


    while True:
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count += 1
            face = cv2.resize(face_extractor(frame), (400, 400))

            file_name_path = 'D:\Object_detection\Datasets\Test/'+name+'/'+name + str(count) + '.jpg'
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', face)
        else:
            print("Face not found")
            pass

        if cv2.waitKey(1) == 13 or count == 100:
            break

    cap.release()
    cv2.destroyAllWindows()
