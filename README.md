# Sign-Language-video-to-audio-translation

-> Run Sign_Language_data_collection.ipynb to click images  
-> Run Create_gestures_from_exist.py to create database with gesture names and labels (gesture_db.db), and process the 
   images
-> Run Generate_histogram.py and create a histogram of hand
-> Run Augment_images.py to perform image augmentations
-> Run Load_images to partition into training and validation dataset
-> Run CNN_train_test.py to train the model with the training dataset and validate it against the validation dataset
-> Realtime_test.py trains the model and also performs realtime testing with camera feed
-> Audio_CNN_train_test.py trains the model with audio labels
-> Audio_Realtime_test.py performs realtime testing with camera feed and predicts the audio labels corresponding to a gesture
