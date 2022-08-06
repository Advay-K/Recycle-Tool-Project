import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image

gen_train = ImageDataGenerator(rescale = 1/255, shear_range = 0.2, zoom_range = 0.2,
                               brightness_range = (0.1, 0.5), horizontal_flip=True)

train_data = gen_train.flow_from_directory('data', target_size = (224, 224), batch_size = 32, class_mode="categorical")

vgg16 = VGG16(input_shape = (224, 224, 3), weights = "imagenet", include_top = False)

for layer in vgg16.layers:
  layer.trainable = False

x = layers.Flatten()(vgg16.output)

prediction = layers.Dense(units = 12, activation="softmax")(x)

model = tf.keras.models.Model(inputs = vgg16.input, outputs=prediction)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics =["accuracy"])

result = model.fit_generator(train_data, epochs = 7, steps_per_epoch=len(train_data)/8)

model.save('det.h5')