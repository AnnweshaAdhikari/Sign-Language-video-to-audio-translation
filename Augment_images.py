import cv2
import os
import numpy as np

def augment_images():
    gesture_folder = "gestures"
    for g_id in os.listdir(gesture_folder):
        processed_folder = os.path.join(gesture_folder, g_id, "processed_images")
        
        # Checking if the processed_images folder exists
        if not os.path.exists(processed_folder):
            print(f"No processed_images folder found in {g_id}. Aborting operation.")
            continue
        
        for i in range(15):  
            base_name = f"processed_{g_id.split('_')[2]}{i + 1}.jpg"  
            path = os.path.join(processed_folder, base_name)
            
            # Checking if the image exists
            img = cv2.imread(path, 1)  
            if img is None:
                print(f"Could not read image: {path}")
                continue
            
            # Flipping the image horizontally
            img_flipped = cv2.flip(img, 1)
            cv2.imwrite(os.path.join(processed_folder, f"flipped_{base_name}"), img_flipped)

            # Adding rotation (15 degrees clockwise)
            center = (img.shape[1] // 2, img.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, 15, 1.0)
            img_rotated = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
            cv2.imwrite(os.path.join(processed_folder, f"rotated_{base_name}"), img_rotated)

            # Brightness adjustment (increase by 50)
            img_bright = cv2.convertScaleAbs(img, alpha=1, beta=50)
            cv2.imwrite(os.path.join(processed_folder, f"bright_{base_name}"), img_bright)

            # Adding Gaussian blur
            img_blur = cv2.GaussianBlur(img, (5, 5), 0)
            cv2.imwrite(os.path.join(processed_folder, f"blur_{base_name}"), img_blur)
            
            # Scaling the image by 1.2x
            scale_matrix = cv2.getRotationMatrix2D(center, 0, 1.2)
            img_scaled = cv2.warpAffine(img, scale_matrix, (img.shape[1], img.shape[0]))
            cv2.imwrite(os.path.join(processed_folder, f"scaled_{base_name}"), img_scaled)

            # Cropping 10% from each side
            h, w = img.shape[:2]
            crop_img = img[int(0.1*h):int(0.9*h), int(0.1*w):int(0.9*w)]
            crop_img = cv2.resize(crop_img, (w, h))
            cv2.imwrite(os.path.join(processed_folder, f"cropped_{base_name}"), crop_img)

            # Increasing contrast
            img_contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)  
            cv2.imwrite(os.path.join(processed_folder, f"contrast_{base_name}"), img_contrast)

            # Random color shift by converting to HSV and adjusting hue
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[..., 0] = (hsv[..., 0] + 10) % 180  
            img_hue_shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite(os.path.join(processed_folder, f"hue_{base_name}"), img_hue_shifted)

            # Adding random noise to include graininess
            noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
            img_noisy = cv2.add(img, noise)
            cv2.imwrite(os.path.join(processed_folder, f"noisy_{base_name}"), img_noisy)
            
            # Applying a shearing transformation to simulate a change in perspective
            shear_matrix = np.float32([[1, 0.3, 0], [0.3, 1, 0]])
            img_sheared = cv2.warpAffine(img, shear_matrix, (img.shape[1], img.shape[0]))
            cv2.imwrite(os.path.join(processed_folder, f"sheared_{base_name}"), img_sheared)

            # Adding random shadow/mask
            mask = np.zeros_like(img)
            top_x, top_y = np.random.randint(0, img.shape[1]//2), np.random.randint(0, img.shape[0]//2)
            bot_x, bot_y = np.random.randint(img.shape[1]//2, img.shape[1]), np.random.randint(img.shape[0]//2, img.shape[0])
            cv2.rectangle(mask, (top_x, top_y), (bot_x, bot_y), (50, 50, 50), -1)
            img_shadowed = cv2.addWeighted(img, 0.8, mask, 0.2, 0)
            cv2.imwrite(os.path.join(processed_folder, f"shadow_{base_name}"), img_shadowed)


            print(f"Augmented images saved for: {path}")

augment_images()
