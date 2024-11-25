# Sign-Language-video-to-audio-translation

-> Run Sign_Language_data_collection.ipynb to click images  

-> Run Create_gestures_from_exist.py to create database with gesture names and labels (gesture_db.db), and process the images

-> Run Generate_histogram.py and create a histogram of hand

-> Run Augment_images.py to perform image augmentations

-> Run Load_images to partition into training and validation dataset

-> Run CNN_train_test.py to train the model with the training dataset and validate it against the validation dataset

-> Realtime_test.py trains the model and also performs realtime testing with camera feed

-> Audio_CNN_train_test.py trains the model with audio labels

-> Audio_Realtime_test.py performs realtime testing with camera feed and predicts the audio labels corresponding to a gesture


Model Architecture Hyperparameters

  Number of Filters in Convolutional Layers:
    First Conv2D layer: 32 filters
    
    Second Conv2D layer: 64 filters
    
    Third Conv2D layer: 128 filters
  
  Filter Size in Convolutional Layers:  All Conv2D layers use a (3, 3) filter size.
  
  Pooling Size in MaxPooling Layers:  All MaxPooling2D layers use a (2, 2) pool size.
  
  Dropout Rate:  Dropout rate in the fully connected layer: 0.6
  
  Dense Layer Neurons:  Fully connected dense layer has 256 neurons.


Training Hyperparameters

  Learning Rate:  1e-4.
  
  Batch Size:  4 samples per batch.
  
  Number of Epochs:  20 epochs for training.
  
  Loss Function:  categorical_crossentropy
  
  Optimizer:  Adam

