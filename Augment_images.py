import cv2
import os

def flip_images():
    gesture_folder = "gestures"
    for g_id in os.listdir(gesture_folder):
        processed_folder = os.path.join(gesture_folder, g_id, "processed_images")
        # Checking if the processed_images folder exists
        if not os.path.exists(processed_folder):
            print(f"No processed_images folder found in {g_id}. Aborting operation.")
            continue
        
        for i in range(15):  # Adjusting range if there are more/less images
            image_name = f"processed_{g_id.split('_')[2]}{i + 1}.jpg"  
            path = os.path.join(processed_folder, image_name)
            new_path = os.path.join(processed_folder, f"flipped_{image_name}")  
            
            print(f"Flipping image: {path}")
            img = cv2.imread(path, 1)  
            if img is not None:
                img_flipped = cv2.flip(img, 1)  # Flipping the image horizontally
                cv2.imwrite(new_path, img_flipped)  
            else:
                print(f"Could not read image: {path}")

flip_images()
