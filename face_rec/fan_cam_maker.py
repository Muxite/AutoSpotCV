import os
import cv2 as cv
import numpy as np
import math
import time
import random
import gc


modelFile = r"res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = r"deploy.prototxt"


class FanCamMaker:
    def __init__(self, prime_directory=r"C:\GitHub\AutoSpotCV\face_rec",
                 video_path=None,
                 recognizer_path=None,
                 people_path=None,
                 threshold=0.5,
                 model_file=modelFile,
                 config_file=configFile,
                 skips=12,
                 autosave_directory=None):
        try:
            os.chdir(prime_directory)
            self.skips = skips  # analyse 1 in every X frames
            self.video_path = video_path
            self.stripped_video = []
            self.autosave_directory = autosave_directory
            self.recognizer_LBPH = cv.face.LBPHFaceRecognizer_create()
            self.recognizer_LBPH.read(recognizer_path)
            self.people = np.load(people_path)
            self.threshold = threshold
            self.net = cv.dnn.readNetFromCaffe(config_file, model_file)
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        except FileNotFoundError:
            print("Missing important file. CANCELED")
            return
        self.face_locations = []  # list of dictionary of tracked face locations (placeholders)
        self.placeholder_to_name = []  # list of dictionary

        # strip the video
        frame_count = 0
        raw_video = cv.VideoCapture(self.video_path)
        while raw_video.isOpened():
            ret, frame = raw_video.read()
            if not ret:
                break
            # append 1 every self.skips frames.
            if frame_count % self.skips == 0:
                self.stripped_video.append(frame)
            frame_count += 1
        raw_video.release()

    def recognize(self, face):
        gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
        resized_face = constrain_img(gray, 128)
        padded_face = pad_image(resized_face, 128)
        predicted_label, confidence = self.recognizer_LBPH.predict(padded_face)
        return predicted_label

    # face_detect_dnn is modified from Bruno Capuano's live face recognition tutorial
    def face_detect_dnn(self, img, threshold=0.7):
        height = img.shape[0]
        width = img.shape[1]
        blob = cv.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False, )
        self.net.setInput(blob)
        detections = self.net.forward()
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

    def find_face_positions(self, queue_length):
        queue = []
        for i, frame in enumerate(self.stripped_video):
            face_boxes = self.face_detect_dnn(frame)
            prev_box = face_boxes[0]
            if i > queue_length:
                # sort face_boxes by proximity to previous box (cheaper than running recognize())
                sorted_face_boxes = sorted(face_boxes,
                                           key=lambda box: get_distance(get_middle(*prev_box), get_middle(*box)))
                for face_box in sorted_face_boxes:
                    # have the previous recognition results and boxes stored
                    # if this chain has a lot of the target recognition result
                    # it is likely to be the correct chain
                    pass



def constrain(corner1, corner2, end_corner):
    corner3 = min(end_corner[0], max(0, corner1[0])), min(end_corner[1], max(0, corner1[1]))
    corner4 = min(end_corner[0], max(0, corner2[0])), min(end_corner[1], max(0, corner2[1]))
    return corner3, corner4


def bound_check(x1, y1, x2, y2, w, h):
    if x1 < 0 or y1 < 0 or y2 > h or x2 > w:
        return False
    return True


def constrain_img(img, size):
    height, width = img.shape[:2]
    factor = min(size / width, size / height)
    if factor >= 1:
        return img
    else:
        new_width = int(width * factor)
        new_height = int(height * factor)
        resized_img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_AREA)
        return resized_img


def pad_image(img, target_size):
    height, width = img.shape[:2]
    delta_w = target_size - width
    delta_h = target_size - height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    new_img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)
    return new_img


def get_middle(x1, y1, x2, y2):
    return (x1 + x2) // 2, (y1 + y2) // 2


def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def menu():
    maker = FanCamMaker(prime_directory=r"C:\GitHub\AutoSpotCV\face_rec",
                        video_path=r"videos\IVE Baddie.mp4",
                        recognizer_path="recognizer_LBPH.yml",
                        people_path="people.npy",
                        threshold=0.5,
                        model_file=modelFile,
                        config_file=configFile,
                        skips=12,
                        autosave_directory=None)
    maker.find_face_positions()


menu()
