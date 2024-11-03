import os
import sqlite3
import cv2
import numpy as np

def create_database():
    # Creating the database if it does not exist
    if not os.path.exists("gesture_db.db"):
        con = sqlite3.connect("gesture_db.db")
        create_table_cmd = "CREATE TABLE gesture ( g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, g_name TEXT NOT NULL )"
        con.execute(create_table_cmd)
        con.commit()

def store_in_database(g_id, g_name):
    con = sqlite3.connect("gesture_db.db")
    cmd = "INSERT INTO gesture (g_id, g_name) VALUES (?, ?)"
    try:
        con.execute(cmd, (g_id, g_name))
    except sqlite3.IntegrityError:
        choice = input("g_id already exists in the database. Want to change the existing record? (Press y/n): ")
        if choice.lower() == 'y':
            cmd = "UPDATE gesture SET g_name = ? WHERE g_id = ?"
            con.execute(cmd, (g_name, g_id))
        else:
            print("Aborting updation of databse")
            return
    con.commit()

def process_and_store_images(folder_name, g_id):
    image_count = 0
    images_processed = 0
    
    processed_folder = os.path.join(folder_name, "processed_images")
    os.makedirs(processed_folder, exist_ok=True)
    for image_name in os.listdir(folder_name):
        image_path = os.path.join(folder_name, image_name)
        if image_path.endswith('.jpg') or image_path.endswith('.png'):
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not read image: {image_path}")
                continue
            
            # Resizing the image and apply thresholding
            img_resized = cv2.resize(img, (300, 300))  # Adjust size as needed
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

            # Finding contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(max_contour) > 100:  # Filtering out small contours
                    # Saving the processed image
                    cv2.imwrite(os.path.join(processed_folder, f"processed_{image_name}"), img_resized)
                    images_processed += 1
            
            image_count += 1
    print(f"Processed {images_processed}/{image_count} images in folder: {folder_name}")

def add_existing_images_to_database(base_folder):
    con = sqlite3.connect("gesture_db.db")
    
    # Iterating through each folder in the base folder
    for folder_name in os.listdir(base_folder):
        if folder_name.startswith("gesture_"):  # Checking if folder name starts with 'gesture_'
            parts = folder_name.split('_')  # Splitting folder name by underscores
            if len(parts) >= 3:  
                g_id = int(parts[1])  # Extracting ID (the second part)
                g_name = '_'.join(parts[2:])  # Joining the rest of the parts for the name
                
                # Storing in database
                store_in_database(g_id, g_name)
    
                # Processing images in the current gesture folder
                process_and_store_images(os.path.join(base_folder, folder_name), g_id)

# Initializing the database
create_database()

# Specifying the base folder containing existing gesture images
base_folder = "gestures"
add_existing_images_to_database(base_folder)
