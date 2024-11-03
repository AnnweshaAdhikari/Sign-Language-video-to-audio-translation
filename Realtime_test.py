import cv2, pickle
import numpy as np
import tensorflow as tf
from CNN_train_test import cnn_model
import os
import sqlite3, pyttsx3
from keras.models import load_model
from threading import Thread



engine = pyttsx3.init()
engine.setProperty('rate', 150)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = load_model('cnn_model_keras2.keras')

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
	cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
	cursor = conn.execute(cmd)
	for row in cursor:
		return row[0]

def get_contour(contour, thresh):
	x1, y1, w1, h1 = cv2.boundingRect(contour)
	save_img = thresh[y1:y1+h1, x1:x1+w1]
	text = ""
	if w1 > h1:
		save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
	elif h1 > w1:
		save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
	pred_probab, pred_class = keras_predict(model, save_img)
	if pred_probab*100 > 90:
		text = get_from_database(pred_class)
	return text


hist = get_histogram()
x, y, w, h = 300, 100, 300, 300
is_voice_on = True

def get_img_contour_thresh(img, hist, x, y, w, h):
    img = cv2.flip(img, 1)  # Flip the image horizontally
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    cv2.filter2D(dst, -1, disc, dst)
    
    # Gaussian and median blur
    blur = cv2.GaussianBlur(dst, (15, 15), 0)
    blur = cv2.medianBlur(blur, 15)

    # Otsu's thresholding
    otsu_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(blur, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY,
                                            11, 2)

    combined_thresh = cv2.bitwise_or(otsu_thresh, adaptive_thresh)

    combined_thresh = cv2.merge((combined_thresh, combined_thresh, combined_thresh))
    combined_thresh = cv2.cvtColor(combined_thresh, cv2.COLOR_BGR2GRAY)
    combined_thresh = combined_thresh[y:y + h, x:x + w]

    contours = cv2.findContours(otsu_thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    
    return img, contours, otsu_thresh


def say_text(text):
	if not is_voice_on:
		return
	while engine._inLoop:
		pass
	engine.say(text)
	engine.runAndWait()


def text_mode(cam):
    global is_voice_on
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
        
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 10000:
                text = get_contour(contour, thresh)
                if old_text == text:
                    count_same_frame += 1
                else:
                    count_same_frame = 0

                if count_same_frame > 20:
                    if len(text) == 1:
                        Thread(target=say_text, args=(text,)).start()
                    word = word + text
                    count_same_frame = 0

            elif cv2.contourArea(contour) < 1000:
                if word != '':
                    Thread(target=say_text, args=(word,)).start()
                text = ""
                word = ""
        else:
            if word != '':
                Thread(target=say_text, args=(word,)).start()
            text = ""
            word = ""

        if text is None:
            text = ""

        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blackboard, "Predicted text- " + text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
        cv2.putText(blackboard, word, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
        
        if is_voice_on:
            cv2.putText(blackboard, "Voice ON", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))
        else:
            cv2.putText(blackboard, "Voice OFF", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        res = np.hstack((img, blackboard))
        cv2.imshow("Recognizing gesture", res)
        cv2.imshow("thresh", thresh)
        keypress = cv2.waitKey(1)

        if keypress == ord('q') or keypress == ord('c'):
            break
        if keypress == ord('v'):
            is_voice_on = not is_voice_on  # Toggle voice on/off
        if keypress == ord('o') and word != '':  # Speak the recognized word when 'o' is pressed
            Thread(target=say_text, args=(word,)).start()

    if keypress == ord('c'):
        return 2
    else:
        return 0
  

def recognize():
	cam = cv2.VideoCapture(0)
	if cam.read()[0]==False:
		cam = cv2.VideoCapture(0)
	text = ""
	word = ""
	count_same_frame = 0
	keypress = 1
	keypress = text_mode(cam)
 

keras_predict(model, np.zeros((50, 50), dtype = np.uint8))		
recognize()
