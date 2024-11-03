import numpy as np
import pickle
import cv2, os
from glob import glob
import tensorflow as tf
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam  
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras import backend as K
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_image_size():
    img = cv2.imread('gestures/gesture_1_hello/processed_images/processed_hello1.jpg', 0)
    return img.shape

def get_num_of_classes():
    return len(glob('gestures/*'))

image_x, image_y = get_image_size()

def cnn_model():
    num_of_classes = get_num_of_classes()
    model = Sequential()
    model.add(Conv2D(16, (2, 2), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_of_classes, activation='softmax'))
    adam = Adam(learning_rate=1e-4)  # Adams optimizer
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    filepath = "cnn_model_keras2.keras"  
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')  
    callbacks_list = [checkpoint1]
    return model, callbacks_list


def train():
    with open("train_images", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)
        
    print(f"Original shape of train_images: {train_images.shape}")

    with open("val_images", "rb") as f:
        val_images = np.array(pickle.load(f))
    with open("val_labels", "rb") as f:
        val_labels = np.array(pickle.load(f), dtype=np.int32)
        
    print(f"Original shape of val_images: {val_images.shape}")
    print("Unique labels in train_labels:", np.unique(train_labels))
    print("Number of classes:", get_num_of_classes())
    
    train_labels -= 1  # To make labels start from 0
    val_labels -= 1    # To make labels start from 0

    num_train_images = train_images.shape[0]
    train_images = np.reshape(train_images, (num_train_images, image_x, image_y, 1))
    val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))
    
    train_labels = to_categorical(train_labels, num_classes=get_num_of_classes())
    val_labels = to_categorical(val_labels, num_classes=get_num_of_classes())

    print(val_labels.shape)
    
    val_true_classes = np.argmax(val_labels, axis=1)
    unique_classes = np.unique(val_true_classes)
    print("Unique classes in validation labels:", unique_classes)
    print("Number of unique classes:", len(unique_classes))

    model, callbacks_list = cnn_model()
    model.summary()
    
    history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=10, batch_size=4, callbacks=callbacks_list, verbose=1)
    
    scores = model.evaluate(val_images, val_labels, verbose=0)
    print("Error: %.2f%%" % (100 - scores[1] * 100))
    
    # Confusion Matriices
    
    train_predictions = model.predict(train_images)
    train_pred_classes = np.argmax(train_predictions, axis=1)
    train_true_classes = np.argmax(train_labels, axis=1)
    train_cm = confusion_matrix(train_true_classes+1, train_pred_classes+1)
    num_classes = len(np.unique(train_true_classes))
    disp_train = ConfusionMatrixDisplay(confusion_matrix=train_cm, display_labels=np.arange(num_classes))
    disp_train.plot(cmap=plt.cm.Blues)
    plt.title("Training Confusion Matrix")
    plt.show()
    
    val_predictions = model.predict(val_images)
    val_pred_classes = np.argmax(val_predictions, axis=1)
    val_true_classes = np.argmax(val_labels, axis=1)
    val_cm = confusion_matrix(val_true_classes+1, val_pred_classes+1)
    num_classes = num_classes = len(unique_classes)
    disp_val = ConfusionMatrixDisplay(confusion_matrix=val_cm, display_labels=np.arange(num_classes))
    disp_val.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
    
    
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid()
    plt.show()

train()
K.clear_session()
