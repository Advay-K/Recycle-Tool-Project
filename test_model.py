import tensorflow as tf
import numpy as np

model = None
output_class = ["battery", "biological", "brown glass", "cardboard", "clothes", "green glass", "metal", "paper", "plastic", "shoes", "trash", "white glass"]

def load_model():
    global model
    model = tf.keras.models.load_model("det.h5")


def classify_waste(image_path):
	global model, output_class
	test_image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
	test_image = tf.keras.preprocessing.image.img_to_array(test_image) / 255
	test_image = np.expand_dims(test_image, axis = 0)
	predicted_array = model.predict(test_image)
	predicted_value = output_class[np.argmax(predicted_array)]
	return predicted_value

load_model()
# print(classify_waste('test.webp'))
