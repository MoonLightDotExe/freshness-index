import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import keras
import cv2 as cv

model = keras.models.load_model('classifier_model2.keras')

image_path = "Resources/fresh_tomato/DSCN4068.jpg_0_1176.jpg"

# capture = cv.VideoCapture(0)

img = image.load_img(image_path, target_size=(256, 256))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

predicted_item = model.predict(img_array)

labels = [
"fresh_capsicum",
"fresh_orange",
"fresh_tomato",
"stale_capsicum",
"stale_orange",
"stale_tomato"
]

index = np.argmax(predicted_item[0])
predicted_label = labels[index]

print("Predicted Label: " + predicted_label)
print("Actual Label: " + image_path)
