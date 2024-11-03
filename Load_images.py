import os
import pickle
import cv2
from glob import glob
import numpy as np
import random
from sklearn.utils import shuffle
from collections import defaultdict


def images_labels_pickle():
    class_images = defaultdict(list)
    images = glob("gestures/*/processed_images/*.jpg")
    images.sort()
    
    # Grouping images by class based on folder name
    for image in images:
        folder_name = image[image.find(os.sep) + 1: image.rfind(os.sep)]  
        g_id = int(folder_name.split('_')[1])  # Extracting the gesture ID
        img = cv2.imread(image, 0)
        class_images[g_id].append((np.array(img, dtype=np.uint8), g_id))  # Storing images by class
    
    # Shuffling images within each class and then combining
    balanced_images_labels = []
    for g_id, imgs in class_images.items():
        shuffled_imgs = shuffle(imgs)
        balanced_images_labels.extend(shuffled_imgs)

    # Shuffling the entire dataset to mix classes
    balanced_images_labels = shuffle(balanced_images_labels)
    return balanced_images_labels


images_labels = images_labels_pickle()
images, labels = zip(*images_labels)
print("Length of images_labels:", len(images_labels))

train_images = images[:int(5/6 * len(images))]
print("Length of train_images:", len(train_images))
with open("train_images", "wb") as f:
    pickle.dump(train_images, f)
del train_images

train_labels = labels[:int(5/6*len(labels))]
print("Length of train_labels", len(train_labels))
with open("train_labels", "wb") as f:
    pickle.dump(train_labels, f)
del train_labels

test_images = images[int(5/6*len(images)):int(11/12*len(images))]
print("Length of test_images", len(test_images))
with open("test_images", "wb") as f:
    pickle.dump(test_images, f)
del test_images

test_labels = labels[int(5/6*len(labels)):int(11/12*len(images))]
print("Length of test_labels", len(test_labels))
with open("test_labels", "wb") as f:
    pickle.dump(test_labels, f)
del test_labels

val_images = images[int(11/12*len(images)):]
print("Length of val_images", len(val_images))
with open("val_images", "wb") as f:
    pickle.dump(val_images, f)
del val_images

val_labels = labels[int(11/12*len(labels)):]
print("Length of val_labels", len(val_labels))
with open("val_labels", "wb") as f:
    pickle.dump(val_labels, f)
del val_labels
