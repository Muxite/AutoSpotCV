# blob settings taken Bruno Capuano
# the testing set is pictures of IVE members (6 members) and Naheed Nenshi
import os
import cv2 as cv
import numpy as np
import time
from tqdm import tqdm
import shutil
import random


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


def preprocess(location):
    features = []  # faces extracted from an image
    labels = []  # the number assigned to a person
    time_adder = 0
    images_count = 0
    net = cv.dnn.readNetFromCaffe(configFile, modelFile)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    people = os.listdir(location)  # Ex: inside face recognition/faces->['Yujin', 'Wonyoung',...]
    print(enumerate(people))
    for label, person in enumerate(people):
        path = os.path.join(location, person)
        print(f"Loading {person}")
        for i, img_subpath in enumerate(tqdm(os.listdir(path), colour='#00ff00', ncols=100)):
            start_time = time.time()
            img_path = os.path.join(path, img_subpath)
            img = cv.imread(img_path)
            if img is None:
                # print(f"Could not read image: {img_path}")
                continue
            face_boxes = face_detect_dnn(net, img, 0.4)
            images_count += 1
            height = img.shape[0]
            width = img.shape[1]
            for (x1, y1, x2, y2) in face_boxes:
                # add the face only if the face is in bounds
                if bound_check(x1, y1, x2, y2, width, height):
                    sub = cv.cvtColor(img[y1:y2, x1:x2], cv.COLOR_BGR2GRAY)
                    faces_roi = cv.resize(sub, (128, 128))
                    features.append(faces_roi)
                    labels.append(label)
                else:
                    # print(f"Face out of bounds: {img_path}")
                    pass
            time_adder += time.time()-start_time
    print(f"Average Time Per Image: {time_adder/images_count}s")
    features = np.array(features)
    labels = np.array(labels)
    return features, labels, people


def train(train_location=r'face_rec\faces', save_location=r'D:\face_trained.yml'):
    features, labels, people = preprocess(train_location)
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(features, labels)
    face_recognizer.save(save_location)
    # np.save('face recognition/features.npy', features)
    # np.save('face recognition/labels.npy', labels)


def test(test_location=r'face_rec\tests', read_location=r'D:\face_trained.yml'):
    net = cv.dnn.readNetFromCaffe(configFile, modelFile)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    features, labels, people = preprocess(test_location)  # extract faces, their real label(number id), and the name
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.read(read_location)
    counters = {person: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for person in people}
    debug_counter = 0
    for feature, label in zip(features, labels):
        predicted_label, confidence = face_recognizer.predict(feature)

        if debug_counter in range(0, 3):
            cv.putText(feature,
                       str(people[predicted_label]),
                       (20, 20),
                       cv.FONT_HERSHEY_SIMPLEX,
                       1.0,
                       (255, 255, 255),
                       thickness=2)
            cv.putText(feature,
                       str(people[predicted_label]),
                       (20, 20),
                       cv.FONT_HERSHEY_SIMPLEX,
                       1.0,
                       (0, 0, 0),
                       thickness=1)
            # cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
            cv.imshow(f"{people[predicted_label]} vs {people[label]}", cv.resize(feature, (256, 256)))
        debug_counter += 1
        for person in people:
            if person == people[label]:
                if person == people[predicted_label]:
                    counters[person]['TP'] += 1  # true positive
                else:
                    counters[person]['FN'] += 1  # false negative
            else:
                if person == people[predicted_label]:
                    counters[person]['FP'] += 1  # false positive
                else:
                    counters[person]['TN'] += 1  # true negative
    for person, counter in counters.items():
        sensitivity = counter['TP'] / (counter['TP'] + counter['FN'])
        specificity = counter['TN'] / (counter['TN'] + counter['FP'])
        print(f"{person}: sensitivity = {sensitivity * 100}% and specificity = {specificity * 100}%")
    cv.waitKey(0)


def split_data(train_location=r"face_rec\faces", tests_location=r'face_rec\tests', split_ratio=0.1):
    sub_folders = os.listdir(train_location)
    for sub_folder in sub_folders:
        train_sub_location = os.path.join(train_location, sub_folder)
        test_sub_location = os.path.join(tests_location, sub_folder)
        if not os.path.exists(test_sub_location):  # make sure the sub location destination exists
            os.makedirs(test_sub_location)
        files = os.listdir(train_sub_location)
        random.shuffle(files)  # Shuffle files
        split_index = int(len(files) * split_ratio)
        test_partition = files[:split_index]  # get files before but not including the index

        # move the partition to the test location
        for image in test_partition:
            shutil.move(os.path.join(train_sub_location, image), os.path.join(test_sub_location, image))


def menu():
    print("Face Recognition for Files, Muk Chunpongtong 2024/7")
    print(f"Working Directory: {os. getcwd()}")
    while True:
        train_or_validate = str(input("Training or Validating? (T/V): "))
        if train_or_validate == "T" or train_or_validate == "t":
            train_location = str(input("Training Dataset Location: "))
            save_location = str(input(".yml Save Location: "))
            train(train_location=train_location, save_location=save_location)
        elif train_or_validate == "V" or train_or_validate == "v":
            test_location = str(input("Test: "))
            read_location = str(input(".yml Read Location: "))
            test(test_location=test_location, read_location=read_location)
        else:
            print("Exiting. Thank You.")
            break


menu()
