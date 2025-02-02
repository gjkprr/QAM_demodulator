import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data preprocessing
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 4,
                                                 class_mode = 'binary')
test_img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_set = test_img_gen.flow_from_directory('dataset/test_set',
                                            target_size=(64,64),
                                            class_mode='binary')

# Build CNN
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64,64,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
# Output layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics =['accuracy'])
cnn.fit(x=training_set, validation_data=test_set, epochs=8)

# Making prediction
import numpy as np
from tensorflow.keras.preprocessing import image
for idx in range(5):
    test_image = image.load_img(f'dataset/check_set/{idx}.jpg',
                                target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0) # batch is always first dimension
    result = cnn.predict(test_image)
    training_set.class_indices
    if result[0][0] == 1:
      prediction = '4-QAM'
    else:
      prediction = '16-QAM'
    print(str(idx) + ' maps to ' + prediction)
