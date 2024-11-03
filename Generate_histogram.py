import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

def RoI_on_screen(image):
    x, y, w, h = 420, 140, 10, 10
    d = 10
    imageCrop = None
    crop = None
    for i in range(10):
        for j in range(7):
            if np.any(imageCrop is None):
                imageCrop = image[y:y+h, x:x+w]
            else:
                imageCrop = np.hstack((imageCrop, image[y:y+h, x:x+w]))
            # Drawing the grid of squares
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
            x += w + d
        if np.any(crop is None):
            crop = imageCrop
        else:
            crop = np.vstack((crop, imageCrop)) 
        imageCrop = None
        x = 420
        y += h + d
    return crop

def generate_histogram():
    feed = cv2.VideoCapture(0)
    feed.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    feed.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    x, y, w, h = 420, 140, 10, 10  # Position and size for ROIs
    flagPressedC, flagPressedS = False, False
    imageCrop = None
    hist = None  

    while True:
        image = feed.read()[1]
        image = cv2.flip(image, 1)  # Flipping the image horizontally
        image = cv2.resize(image, (640, 480))  
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        keypress = cv2.waitKey(1)
        if keypress == ord('c'):  # Capturing histogram on pressing 'c'
            flagPressedC = True
            hsvCrop = cv2.cvtColor(imageCrop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            print("Histogram generated successfully")  
            plt.imshow(hist, interpolation='nearest', cmap='viridis')
            plt.title("Histogram")
            plt.show()

        elif keypress == ord('s'):  # Saving histogram on pressing 's'
            flagPressedS = True	
            break

        if flagPressedC and hist is not None:
            dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            cv2.filter2D(dst, -1, disc, dst)

            # Applying Gaussian and median blur to reduce noise
            blur = cv2.GaussianBlur(dst, (15, 15), 0)
            blur = cv2.medianBlur(blur, 15)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            blur = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)


            # Otsu's thresholding method to create a binary mask
            ret2, otsu_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Adaptive thresholding method
            adaptive_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY, 11, 2)
            
            # Combining Otsu and adaptive threshold results
            #combined_thresh = cv2.bitwise_or(otsu_thresh, adaptive_thresh)
            combined_thresh = otsu_thresh

            # Morphological operations to remove small noise
            kernel = np.ones((5, 5), np.uint8)
            combined_thresh = cv2.morphologyEx(combined_thresh, cv2.MORPH_OPEN, kernel)  # Removing noise
            combined_thresh = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel)  # Closing gaps

            #combined_thresh = cv2.merge((combined_thresh, combined_thresh, combined_thresh))
            cv2.imshow("Threshold Mask", otsu_thresh)

        if not flagPressedS:
            imageCrop = RoI_on_screen(image)
        
        # Showing the original image with squares
        cv2.imshow("Set hand histogram", image)  

    feed.release()
    cv2.destroyAllWindows()

    # Saving histogram to file
    if hist is not None:
        with open(r"D:\IE643_Project\hist", "wb") as f:
            pickle.dump(hist, f)
            print("Histogram saved to file successfully.")
    else:
        print("Histogram was not generated.")
    
generate_histogram()
