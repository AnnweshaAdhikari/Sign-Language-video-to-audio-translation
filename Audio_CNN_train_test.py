import numpy as np
import pickle
import cv2, os
from glob import glob
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pygame  
import seaborn as sns
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

AUDIO_PATH = "audio/"

# Mapping of label indices to gesture names, including background
gesture_labels = {
    0: "hello",
    1: "help",
    2: "iloveyou",
    3: "no",
    4: "thanks",
    5: "why",
    6: "yes"
}

pygame.mixer.init()

def get_image_size():
    img = cv2.imread('gestures/gesture_1_hello/processed_images/processed_hello1.jpg', 0)
    return img.shape

def get_num_of_classes():
    return len(gesture_labels)

image_x, image_y = get_image_size()

def flatten_list(nested_list):
    """Flatten a nested list."""
    return [item for sublist in nested_list for item in sublist]

def cnn_model():
    num_of_classes = get_num_of_classes()
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.6))  # Increased dropout for better regularization
    model.add(Dense(num_of_classes, activation='softmax'))
    adam = Adam(learning_rate=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    filepath = "cnn_model_audio_keras.keras"  
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')  
    callbacks_list = [checkpoint1]
    return model, callbacks_list

def extract_label_from_audio_path(audio_path):
    """Extract label index from audio file path."""
    gesture_name = os.path.basename(audio_path).split('_')[0]
    for label_index, name in gesture_labels.items():
        if name == gesture_name:
            return label_index
    return 0  

def play_audio(label_index):
    """Play a random audio file associated with the predicted gesture class."""
    gesture_name = gesture_labels.get(label_index)
    if gesture_name:
        audio_files = glob(f"{AUDIO_PATH}{gesture_name}_*.wav")
        if audio_files:
            audio_file = random.choice(audio_files)
            try:
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():  
                    pygame.time.Clock().tick(10)
            except pygame.error as e:
                print(f"Error playing audio {audio_file}: {e}")
        else:
            print(f"No audio files found for gesture: {gesture_name}")

def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def predict_with_threshold(model, image, threshold=0.6):
    """Predict with a confidence threshold."""
    image = np.reshape(image, (1, image_x, image_y, 1))
    predictions = model.predict(image)
    max_prob = np.max(predictions)
    predicted_class = np.argmax(predictions)
    
    if max_prob < threshold:
        return 0  
    return predicted_class

def train():
    with open("train_images", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_audio", "rb") as f:
        train_audio_paths = pickle.load(f)
        if isinstance(train_audio_paths[0], list):
            train_audio_paths = flatten_list(train_audio_paths)
        train_labels = np.array([extract_label_from_audio_path(path) for path in train_audio_paths], dtype=np.int32)

    with open("val_images", "rb") as f:
        val_images = np.array(pickle.load(f))
    with open("val_audio", "rb") as f:
        val_audio_paths = pickle.load(f)
        if isinstance(val_audio_paths[0], list):
            val_audio_paths = flatten_list(val_audio_paths)
        val_labels = np.array([extract_label_from_audio_path(path) for path in val_audio_paths], dtype=np.int32)

    train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
    val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))
    
    train_labels_one_hot = to_categorical(train_labels, num_classes=get_num_of_classes())
    val_labels_one_hot = to_categorical(val_labels, num_classes=get_num_of_classes())

    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_dict = dict(enumerate(class_weights))

    hello_class_weight = class_weights_dict[1]
    scale_factor = 1.25  
    class_weights_dict[1] = hello_class_weight / scale_factor

    for i in range(len(class_weights_dict)):
        if i != 1:  
            class_weights_dict[i] *= scale_factor

    print("Adjusted class weights:", class_weights_dict)

    model, callbacks_list = cnn_model()
    model.summary()
    
    history = model.fit(
        train_images, train_labels_one_hot,
        validation_data=(val_images, val_labels_one_hot),
        epochs=20,
        batch_size=4,
        callbacks=callbacks_list,
        verbose=1,
        class_weight=class_weights_dict
    )
    
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    scores = model.evaluate(val_images, val_labels_one_hot, verbose=0)
    print("Validation Accuracy: %.2f%%" % (scores[1] * 100))
    
    val_predictions = [
        predict_with_threshold(model, img) for img in val_images
    ]

    plot_confusion_matrix(val_labels, val_predictions, list(gesture_labels.values()), "Validation Confusion Matrix")
    
    print("Validation Classification Report:")
    print(classification_report(val_labels, val_predictions, target_names=list(gesture_labels.values())))

train()

