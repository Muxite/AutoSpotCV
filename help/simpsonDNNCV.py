import numpy as np
import pandas as pd
import os
import caer
import cv2 as cv
import tensorflow
import canaro
import gc

from keras.src.callbacks import LearningRateScheduler
from keras.src.utils import to_categorical

os.chdir(r"D:\Github\AutoSpotCV")
img_size = (80, 80)
test_size = 0.2
batch_size = 32
epochs = 10
channels = 1  # grayscale
dataset_location = r'simpsons_dataset'
character_dict = {}  # data structure for how many images per character
for character in os.listdir(dataset_location):
    character_dict[character] = len(os.listdir(os.path.join(dataset_location, character)))

character_dict = caer.sort_dict(character_dict, descending=True)

characters = []
for count, i in enumerate(character_dict):
    characters.append(i[0])
    #  count >= 10:
    #    break
print(characters)  # print just 10 or less

# train is a big list which contains sub lists. These sub lists contain the image array, and the label
train = caer.preprocess_from_dir(dataset_location, characters, channels=channels, IMG_SIZE=img_size, isShuffle=True)
# split train into the features and the labels
features, labels = caer.sep_train(train, IMG_SIZE=img_size)  # featureSet is a 4d tensor

# normalize the data for quicker learning
features = caer.normalize(features)
labels = to_categorical(labels, len(characters))

# split the data into testing and training
features_train, features_validate, labels_train, labels_validate = \
    caer.train_val_split(features, labels, val_ratio=test_size)

# remove things to free up memory
del train
del features
del labels
gc.collect()

data_gen = canaro.generators.imageDataGenerator()  # add shifts and flips to increase dataset size
train_gen = data_gen.flow(features_train, labels_train, batch_size=batch_size)  # make batches of imgs and labels


# loss function models error, optimization seeks to minimize the loss function
# decay reduces the rate of learning to increase convergence
# learning_rate is the initial learning rate
# momentum is a measure of how much the previous gradient is used in the current gradient.
# Nesterov uses NAG to find future gradient rather than purely using momentum, tries to get faster convergence.
model = canaro.models.createSimpsonsModel(IMG_SIZE=img_size, channels=channels, output_dim=len(characters),
                                          loss='binary_crossentropy', decay=1e-6, learning_rate=0.001,
                                          momentum=0.9, nesterov=True)
model.summary()
callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]
training = model.fit(train_gen,
                     steps_per_epoch=len(features_train)//batch_size,
                     epochs=epochs,
                     validation_data=(features_validate, labels_validate),
                     validation_steps=len(labels_validate)//batch_size,
                     callbacks=callbacks_list)


def prepare(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, img_size)
    img = caer.reshape(img, img_size, 1)
    return img


def test():
    counters = {char: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for char in characters}
    for feature, label in zip(features_validate, labels_validate):
        predicted_label, confidence = model.predict(feature)
        for char in characters:
            if char == characters[label]:
                if char == characters[predicted_label]:
                    counters[char]['TP'] += 1  # True positive
                else:
                    counters[char]['FN'] += 1  # False negative
            else:
                if char == characters[predicted_label]:
                    counters[char]['FP'] += 1  # False positive
                else:
                    counters[character]['TN'] += 1  # True negative
    for char, counter in counters.items():
        sensitivity = counter['TP'] / (counter['TP'] + counter['FN'])
        specificity = counter['TN'] / (counter['TN'] + counter['FP'])
        print(f"{character}: sensitivity = {sensitivity:.4f}")
