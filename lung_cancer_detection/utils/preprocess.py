import cv2
import numpy as np


def preprocess_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Unable to read image file.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image
