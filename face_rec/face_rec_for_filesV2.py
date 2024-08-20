# trained on a variety of faces (mostly celebrities) of different ethnicities. Images are straight from Google.


import os
import cv2 as cv
import numpy as np
import time
import random
import gc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

modelFile = r"face_rec\res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = r"face_rec\deploy.prototxt"
os.chdir(r"C:\GitHub\AutoSpotCV")


class FaceRecognizer:
    def __init__(self, autosave=True, autosave_directory=r"face_rec"):
        self.train_features = None  # must be loaded/inflated using preprocess
        self.train_labels = None
        self.test_features = None
        self.test_labels = None
        self.people = None  # string names for people (for human readability)
        self.recognizer_LBPH = None
        self.recognizer_LBPH_path = None
        self.autosave = autosave
        self.autosave_directory = autosave_directory

    def preprocess(self, load_directory, train_ratio=1.0, reloading=False, test_size=0.1):
        if reloading:  # if the items are already npy saved
            try:
                train_features = np.load(os.path.join(self.autosave_directory, 'train_features.npy'))
                train_labels = np.load(os.path.join(self.autosave_directory, 'train_labels.npy'))
                test_features = np.load(os.path.join(self.autosave_directory, 'test_features.npy'))
                test_labels = np.load(os.path.join(self.autosave_directory, 'test_labels.npy'))
                people = np.load(os.path.join(self.autosave_directory, 'people.npy'))
                # Recombine the features and labels
                features = np.concatenate((train_features, test_features), axis=0)
                labels = np.concatenate((train_labels, test_labels), axis=0)

                # Split them into new train and test sets
                self.train_features, self.test_features, self.train_labels, self.test_labels = split_data(
                    features, labels, test_size)
                self.people = people
                print("*reloaded and recombined previously saved features and labels*")
                return
            except FileNotFoundError:
                print("couldn't reload, will preprocess")

        features = []  # faces extracted from an image
        labels = []  # the number assigned to a person
        time_adder = 0
        images_count = 0
        net = cv.dnn.readNetFromCaffe(configFile, modelFile)
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        inflater = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
        )
        people = os.listdir(load_directory)  # Ex: inside face recognition/faces->['Yujin', 'Wonyoung',...]
        for label, person in enumerate(people):
            path = os.path.join(load_directory, person)
            print(f"Loading {person}")
            img_paths = os.listdir(path)
            random.shuffle(img_paths)  # shuffle the image paths
            if train_ratio < 1.0:  # if the dataset is to be made smaller
                train_size = int(len(img_paths) * train_ratio)  # number of images to use for training
                img_paths = img_paths[:train_size]  # Select only the specified ratio of images
            for i, img_subpath in enumerate(img_paths):
                start_time = time.time()  # set a timer
                img_path = os.path.join(path, img_subpath)
                img = cv.imread(img_path)  # get an image
                if img is None:
                    continue
                fluffed_images = inflate(inflater, img, fluff_amount=train_ratio)  # artificially expand dataset
                for fluff in fluffed_images:  # for every image created
                    face_boxes = face_detect_dnn(net, fluff, 0.4)  # get the location of faces
                    height = fluff.shape[0]
                    width = fluff.shape[1]
                    for (x1, y1, x2, y2) in face_boxes:  # for each location with a face
                        if bound_check(x1, y1, x2, y2, width, height):  # add the face only if the face is in bounds
                            images_count += 1  # count it as a trained-upon face
                            sub = cv.cvtColor(img[y1:y2, x1:x2], cv.COLOR_BGR2GRAY)
                            gray_face = cv.resize(sub, (128, 128)) / 255.0
                            features.append(gray_face)
                            labels.append(label)  # label it appropriately (as a number)
                    time_adder += time.time() - start_time
                del img, fluffed_images, face_boxes
                gc.collect()  # manually trigger garbage collection
        print(f"*Loaded {images_count} images*")
        print(f"Average Time Per Image: {time_adder / images_count}s")

        # Convert lists to numpy arrays
        features = np.array(features)
        labels = np.array(labels)

        # Split the data into training and testing sets
        self.train_features, self.test_features, self.train_labels, self.test_labels = split_data(
            features, labels, test_size)

        self.people = np.array(people)
        if self.autosave:
            np.save(os.path.join(self.autosave_directory, 'train_features.npy'), self.train_features)
            np.save(os.path.join(self.autosave_directory, 'train_labels.npy'), self.train_labels)
            np.save(os.path.join(self.autosave_directory, 'test_features.npy'), self.test_features)
            np.save(os.path.join(self.autosave_directory, 'test_labels.npy'), self.test_labels)
            np.save(os.path.join(self.autosave_directory, 'people.npy'), self.people)

    def train_LBPH(self, directory=None, retrain=False):
        if self.recognizer_LBPH is None:
            self.recognizer_LBPH = cv.face.LBPHFaceRecognizer_create()

        read_directory = directory or self.autosave_directory
        if read_directory and not retrain:
            try:
                print("loading...")
                self.recognizer_LBPH.read(os.path.join(read_directory, "recognizer_LBPH.yml"))
                print("loading complete")
                return
            except FileNotFoundError:
                print("loading LBPH failed, training")

        self.recognizer_LBPH.train(self.train_features, self.train_labels)
        print("trained LBPH recognizer")
        save_directory = directory or self.autosave_directory
        if save_directory:
            print("saving...")
            self.recognizer_LBPH.save(os.path.join(save_directory, "recognizer_LBPH.yml"))
            print("saving complete")

    def test_LBPH(self, read_location=None):
        net = cv.dnn.readNetFromCaffe(configFile, modelFile)
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        features, labels, people = self.test_features, self.test_labels, self.people
        counters = {person: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for person in people}
        predictions = {person: [] for person in people}  # Track predictions for each person
        # debug_counter = 0
        for feature, label in zip(features, labels):
            predicted_label, confidence = self.recognizer_LBPH.predict(feature)
            predictions[people[label]].append(people[predicted_label])  # Track the prediction
            # if debug_counter in range(0, 3):
            # cv.putText(feature,
            #            str(people[predicted_label]),
            #            (20, 20),
            #            cv.FONT_HERSHEY_SIMPLEX,
            #            1.0,
            #            (255, 255, 255),
            #            thickness=2)
            # cv.putText(feature,
            #            str(people[predicted_label]),
            #            (20, 20),
            #            cv.FONT_HERSHEY_SIMPLEX,
            #            1.0,
            #            (0, 0, 0),
            #            thickness=1)
            # cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
            # cv.imshow(f"{people[predicted_label]} vs {people[label]}", cv.resize(feature, (256, 256)))
            # debug_counter += 1
        for person, counter in counters.items():
            sensitivity = counter['TP'] / (counter['TP'] + counter['FN'])
            print(f"{person}: sensitivity = {sensitivity:.4f}")

        # Generate histograms
        for person, preds in predictions.items():
            plt.figure(figsize=(10, 6))
            plt.hist(preds, bins=len(people), edgecolor='black')
            plt.title(f'Predictions for {person}')
            plt.xlabel('Predicted Person')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
            plt.show()
        cv.waitKey(0)


def constrain(corner1, corner2, end_corner):
    corner3 = min(end_corner[0], max(0, corner1[0])), min(end_corner[1], max(0, corner1[1]))
    corner4 = min(end_corner[0], max(0, corner2[0])), min(end_corner[1], max(0, corner2[1]))
    return corner3, corner4


def bound_check(x1, y1, x2, y2, w, h):
    if x1 < 0 or y1 < 0 or y2 > h or x2 > w:
        return False
    return True


# face_detect_dnn is modified from Bruno Capuano's live face recognition tutorial
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


def inflate(inflater, img, fluff_amount):
    # ensure the image is in BGR format
    fluff_images = [img]  # these are the augmented images (make sure to add original)
    img_array = np.expand_dims(img, axis=0)  # add batch dimension
    if fluff_amount > 1:
        for i, batch in enumerate(inflater.flow(img_array, batch_size=1)):
            if i > fluff_amount - 1:  # account for the original
                break
            bgr = (batch[0]*255).astype(np.uint8)
            fluff_images.append(bgr)
    return fluff_images


def split_data(features, labels, test_size, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)  # random seed, otherwise set a seed for reproducibility

    combined = list(zip(features, labels))  # zip makes a tuple, so make it a list
    np.random.shuffle(combined)
    features, labels = zip(*combined)  # * unpacks into new list
    features = np.array(features)
    labels = np.array(labels)
    split_index = int(len(features) * (1 - test_size))

    train_features = features[:split_index]
    test_features = features[split_index:]
    train_labels = labels[:split_index]
    test_labels = labels[split_index:]

    return train_features, test_features, train_labels, test_labels


def debug_menu():
    print("Face Recognition for Files V2, Muk Chunpongtong 2024/8")
    face_rec = FaceRecognizer(autosave=True, autosave_directory=r"face_rec")
    # face_rec.preprocess(load_directory=r"face_rec/faces", reloading=True, train_ratio=2.0)
    face_rec.train_LBPH()
    face_rec.test_LBPH()
