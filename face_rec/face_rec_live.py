import os
import cv2 as cv
import numpy as np
import time
from tqdm import tqdm
import shutil
import random

modelFile = r"face_rec\res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = r"face_rec\deploy.prototxt"
os.chdir(r"C:\GitHub\AutoSpotCV")


def constrain(corner1, corner2, end_corner):
    corner3 = min(end_corner[0], max(0, corner1[0])), min(end_corner[1], max(0, corner1[1]))
    corner4 = min(end_corner[0], max(0, corner2[0])), min(end_corner[1], max(0, corner2[1]))
    return corner3, corner4


def bound_check(x1, y1, x2, y2, w, h):
    if x1 < 0 or y1 < 0 or y2 > h or x2 > w:
        return False
    return True


def find_people(location):
    people = os.listdir(location)  # Ex: inside face recognition/faces->['Yujin', 'Wonyoung',...]
    return people


def face_detect_dnn(net, img, threshold=0.7):
    height = img.shape[0]
    width = img.shape[1]
    blob = cv.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            x1 = int(detections[0, 0, i, 3] * width)
            y1 = int(detections[0, 0, i, 4] * height)
            x2 = int(detections[0, 0, i, 5] * width)
            y2 = int(detections[0, 0, i, 6] * height)
            face_boxes.append([x1, y1, x2, y2])
    return face_boxes


def live(read_location=r'C:\User\mukch\face_trained.yml', people_location=r'face_rec\faces'):
    people = find_people(people_location)
    net = cv.dnn.readNetFromCaffe(configFile, modelFile)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.read(read_location)
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face_boxes = face_detect_dnn(net, frame, 0.4)
        for (x1, y1, x2, y2) in face_boxes:
            if bound_check(x1, y1, x2, y2, frame.shape[1], frame.shape[0]):
                sub = cv.cvtColor(frame[y1:y2, x1:x2], cv.COLOR_BGR2GRAY)
                faces_roi = cv.resize(sub, (128, 128))
                predicted_label, confidence = face_recognizer.predict(faces_roi)
                label_text = f"{people[predicted_label]}: {confidence:.2f}"
                cv.putText(frame, label_text, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.imshow('Live Face Recognition', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


def menu():
    print("Face Recognition Live, Muk Chunpongtong 2024/7")
    print(f"Working Directory: {os.getcwd()}")
    read_location = str(input(".yml Read Location: "))
    live(read_location=read_location)


menu()
