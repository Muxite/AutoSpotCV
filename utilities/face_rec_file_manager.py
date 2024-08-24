# a few simple programs to manage training and testing files.
# constrain_img forces all files in a testing/training directory to be under a size
# split_data randomly splits a percent part of a big directory into a little testing directory
import os
import cv2 as cv
import numpy as np
import time
from tqdm import tqdm
import shutil
import random
os.chdir(r"C:\GitHub\AutoSpotCV")


def constrain_img(img, size):
    height, width = img.shape[:2]  # get the first 2 dims
    factor = min(size/width, size / height)  # whichever is smaller
    if factor >= 1:
        return img
    else:  # don't need an else here but it looks better
        new_width = int(width * factor)
        new_height = int(height * factor)
        resized_img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_AREA)
        return resized_img


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


def delete(location):
    if os.path.exists(location):
        os.remove(location)


def simplify(size, location=r"face_rec\faces"):  # convert a directory to a constrained size of image
    sub_folders = os.listdir(location)
    for sub_folder in sub_folders:
        sub_location = os.path.join(location, sub_folder)
        print(sub_location)
        for img_name in tqdm(os.listdir(sub_location)):  # for each image in a set
            old_img_path = os.path.join(sub_location, img_name)
            new_name = img_name.split(".", 9)[0] + ".jpg"  # the part before the dot + .jpg
            new_img_path = os.path.join(sub_location, new_name)
            try:
                old_img = cv.imread(old_img_path)
                new_img = constrain_img(old_img, size)
                # now delete the old one
                delete(old_img_path)
                cv.imwrite(new_img_path, new_img)
            except AttributeError:
                delete(old_img_path)  # its broken, remove it


def menu():
    print("Face Recognition File Manager, Muk Chunpongtong 2024/7")
    print(f"Working Directory: {os. getcwd()}")
    print(f"Enter invalid to exit.")
    while True:
        train_or_validate = str(input("Constrain Directory or Split Data? (C/S): "))
        if train_or_validate == "C" or train_or_validate == "c":
            print("**Constrain directory will open a folder, and change the image files within the subfolders**")
            size = int(input("Pixel Size to Constrain to: "))
            location = str(input("Location to Constrain: "))
            simplify(size, location=location)
        elif train_or_validate == "S" or train_or_validate == "s":
            print("**Will split a ratio of training dataset to become a testing dataset**")
            train_location = str(input("Training Dataset Location: "))
            test_location = str(input("Testing Target Location: "))
            ratio = float(input("Ratio to Testing: "))
            split_data(train_location=train_location, tests_location=test_location, split_ratio=ratio)
        else:
            print("Exiting. Thank You.")
            break


menu()
