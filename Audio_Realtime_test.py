import cv2
import pickle
import numpy as np
import tensorflow as tf
import os
import sqlite3
from keras.models import load_model
from threading import Thread
import pygame  

pygame.mixer.init()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = load_model('cnn_model_audio_keras.keras')


def get_histogram():
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    return hist


def get_image_size():
    img = cv2.imread('gestures/gesture_1_hello/processed_images/processed_hello1.jpg', 0)
    return img.shape


image_x, image_y = get_image_size()


def preprocess_image(img):
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (1, image_x, image_y, 1))
    return img


def keras_predict(model, image):
    processed = preprocess_image(image)
    pred_probab = model.predict(processed, verbose=0)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


def get_from_database(pred_class):
    conn = sqlite3.connect("gesture_db.db")
    cmd = "SELECT g_name FROM gesture WHERE g_id=" + str(pred_class)
    cursor = conn.execute(cmd)
    for row in cursor:
        return row[0]


def get_audio_file(gesture_name):
    audio_folder = "audio"
    files = [f for f in os.listdir(audio_folder) if f.startswith(gesture_name)]
    if files:
        return os.path.join(audio_folder, files[0])
    return None


def play_audio(audio_file):
    if not pygame.mixer.get_busy():
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()


def get_contour(contour, thresh):
    x1, y1, w1, h1 = cv2.boundingRect(contour)
    save_img = thresh[y1:y1 + h1, x1:x1 + w1]
    text = ""
    if w1 > h1:
        save_img = cv2.copyMakeBorder(save_img, int((w1 - h1) / 2), int((w1 - h1) / 2), 0, 0,
                                      cv2.BORDER_CONSTANT, (0, 0, 0))
    elif h1 > w1:
        save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1 - w1) / 2), int((h1 - w1) / 2),
                                      cv2.BORDER_CONSTANT, (0, 0, 0))
    pred_probab, pred_class = keras_predict(model, save_img)
    if pred_probab * 100 > 90:
        text = get_from_database(pred_class)
    else:
        text = ""
    return text


hist = get_histogram()
x, y, w, h = 350, 150, 200, 200


def get_img_contour_thresh(img, hist, x, y, w, h):
    img = cv2.flip(img, 1)  
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Backprojection using the histogram
    dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    cv2.filter2D(dst, -1, disc, dst)

    # Gaussian Blur
    blur = cv2.GaussianBlur(dst, (15, 15), 0)

    # Otsu's thresholding
    otsu_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Morphological Operations
    kernel = np.ones((5, 5), np.uint8)
    otsu_thresh = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel)  # Removes small noise
    otsu_thresh = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel)  # Closes gaps

    # Canny Edge Detection 
    edges = cv2.Canny(otsu_thresh, 100, 200)

    contours = cv2.findContours(otsu_thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

    cv2.imshow("Edges", edges)

    return img, contours, otsu_thresh



def text_mode(cam):
    text = ""
    word = ""
    count_same_frame = 0
    hist = get_histogram()
    x, y, w, h = 350, 150, 200, 200

    while True:
        img = cam.read()[1]
        img = cv2.resize(img, (640, 480))
        img, contours, thresh = get_img_contour_thresh(img, hist, x, y, w, h)
        old_text = text

        contour = max(contours, key=cv2.contourArea) if contours else None
        if contour is None or cv2.contourArea(contour) < 1000:
            text = ""

        if contour is not None and cv2.contourArea(contour) > 10000:
            text = get_contour(contour, thresh)
            if old_text == text:
                count_same_frame += 1
            else:
                count_same_frame = 0

            if count_same_frame > 15 and text != "":
                audio_file = get_audio_file(text)
                if audio_file:
                    Thread(target=play_audio, args=(audio_file,)).start()
                count_same_frame = 0

        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blackboard, "Predicted text- " + text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        res = np.hstack((img, blackboard))
        cv2.imshow("Recognizing gesture", res)
        cv2.imshow("thresh", thresh)
        keypress = cv2.waitKey(1)

        if keypress == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


def recognize():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Camera not accessible.")
        return
    text_mode(cam)


keras_predict(model, np.zeros((50, 50), dtype=np.uint8))
recognize()
