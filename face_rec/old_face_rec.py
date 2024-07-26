# heavily modified from tutorial, using a different face detector system and different dataset
import os
import cv2 as cv
import numpy as np
import time
from tqdm import tqdm


people_list = ['Yujin', 'Gaeul', 'Rei', 'Wonyoung', 'Liz', 'Leeseo', 'Naheed Nenshi']
modelFile = r"face_rec\res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = r"face_rec\deploy.prototxt"
# the program's directory is the big folder which contains the folder of stuff on github
os.chdir(r"D:\GitHub\AutoSpotCV")


def constrain(corner1, corner2, end_corner):
    corner3 = min(end_corner[0], max(0, corner1[0])), min(end_corner[1], max(0, corner1[1]))
    corner4 = min(end_corner[0], max(0, corner2[0])), min(end_corner[1], max(0, corner2[1]))
    return corner3, corner4


def bound_check(x1, y1, x2, y2, w, h):
    if x1 < 0 or y1 < 0 or y2 > h or x2 > w:
        return False
    return True


def face_detect_dnn(net, img, threshold=0.7):
    height = img.shape[0]
    width = img.shape[1]
    blob = cv.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False, )
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


def preprocess(people, location):
    features = []
    labels = []
    time_adder = 0
    images_count = 0
    net = cv.dnn.readNetFromCaffe(configFile, modelFile)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    for person in people:
        path = os.path.join(location, person)
        label = people.index(person)
        print(f"Loading {person}")
        progress_bar = tqdm(os.listdir(path), ascii=False, colour='#00ff00', ncols=100)
        for i, img_subpath in enumerate(progress_bar):
            start_time = time.time()
            img_path = os.path.join(path, img_subpath)
            img = cv.imread(img_path)
            if img is None:
                # print(f"Could not read image: {img_path}")
                continue
            face_boxes = face_detect_dnn(net, img, 0.5)
            images_count += 1
            height = img.shape[0]
            width = img.shape[1]
            for (x1, y1, x2, y2) in face_boxes:
                if bound_check(x1, y1, x2, y2, width, height):
                    sub = cv.cvtColor(img[y1:y2, x1:x2], cv.COLOR_BGR2GRAY)
                    faces_roi = cv.resize(sub, (128, 128))
                    features.append(faces_roi)
                    labels.append(label)

                else:
                    # print(f"Face out of bounds: {img_path}")
                    pass
            time_adder += time.time()-start_time
    print(time_adder/images_count)
    features = np.array(features)
    labels = np.array(labels)
    return features, labels


def train(train_location=r'face_rec\faces'):
    people = people_list
    features, labels = preprocess(people, train_location)
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(features, labels)
    face_recognizer.save(r'D:\face_trained.yml')


def test(test_location=r'face_rec\tests'):
    people = people_list
    net = cv.dnn.readNetFromCaffe(configFile, modelFile)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    features, labels = preprocess(people, test_location)
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.read(r'D:\face_trained.yml')
    counters = {person: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for person in people}
    n_labels = len(labels)
    for feature, label in zip(features, labels):
        predicted_label, confidence = face_recognizer.predict(feature)
        cv.putText(feature,
                   str(people[predicted_label]),
                   (20, 20),
                   cv.FONT_HERSHEY_SIMPLEX,
                   1.0,
                   (0, 0, 255),
                   thickness=2)
        # cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
        # cv.imshow(f"{people[predicted_label]} vs {people[label]}", cv.resize(feature, (128, 128)))
        for person in people:
            if person == people[label]:
                if person == people[predicted_label]:
                    counters[person]['TP'] += 1  # True positive
                else:
                    counters[person]['FN'] += 1  # False negative
            else:
                if person == people[predicted_label]:
                    counters[person]['FP'] += 1  # False positive
                else:
                    counters[person]['TN'] += 1  # True negative
    for person, counter in counters.items():
        sensitivity = counter['TP'] / (counter['TP'] + counter['FN'])
        specificity = counter['TN'] / (counter['TN'] + counter['FP'])
        print(f"{person}: sensitivity = {sensitivity * 100}% and specificity = {specificity * 100}%")

train()
test()

cv.waitKey(0)
