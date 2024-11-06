import os
import cv2
import random
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa

# Paths for the dataset and augmented data
data_path = r'C:\Users\Saarthak\Desktop\face_recognition\data'
augmented_data_path = r'C:\Users\Saarthak\Desktop\face_recognition\augmented_data'

# Augmentation sequence
seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 1.0)),
    iaa.Affine(rotate=(-20, 20)),
    iaa.Multiply((0.8, 1.2))
])

# Process each person in the dataset
for person_name in os.listdir(data_path):
    person_folder = os.path.join(data_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    # Gather all .jpg and .jpeg images
    images = [
        os.path.join(person_folder, img)
        for img in os.listdir(person_folder)
        if img.lower().endswith(('.jpg', '.jpeg'))
    ]
    
    # Set up directory for augmented images
    augmented_person_folder = os.path.join(augmented_data_path, person_name)
    os.makedirs(augmented_person_folder, exist_ok=True)

    # Check and process images to ensure 10 per person
    if len(images) >= 10:
        selected_images = images[:10]
    else:
        selected_images = images[:]
        while len(selected_images) < 10:
            img_path = random.choice(images)
            image = Image.open(img_path)
            image_np = np.array(image)

            # Apply augmentation
            augmented_image_np = seq(image=image_np)
            augmented_image = Image.fromarray(augmented_image_np)

            # Save the augmented image
            new_img_name = f"{person_name}_aug_{len(selected_images) + 1}.jpg"
            augmented_image.save(os.path.join(augmented_person_folder, new_img_name))

            # Append the full path of the newly saved image for tracking
            selected_images.append(os.path.join(augmented_person_folder, new_img_name))
    
    # Copy original images (up to 10) to augmented folder
    for i, img_path in enumerate(selected_images[:10]):
        if os.path.isfile(img_path):  # Ensure it is a valid path
            image = Image.open(img_path)
            new_img_name = f"{person_name}_orig_{i + 1}.jpg"
            image.save(os.path.join(augmented_person_folder, new_img_name))
    
    print(f"{person_name}: Total images after augmentation: {len(selected_images)}")
