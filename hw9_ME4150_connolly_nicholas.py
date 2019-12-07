# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 22:46:40 2019

@author: conno
"""
# Load packages for data augmentation
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt

# Import picture of nail and picture of screw
screw_img = load_img('./Screw.jpg')
nail_img = load_img('./Nail.jpg')

# Display original images
print('Original Screw')
plt.imshow(screw_img)
plt.show()
print('Original Nail')
plt.imshow(nail_img)
plt.show()



# Create object to transform data
data_generator = ImageDataGenerator(rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.3,
                                    zoom_range=0.3,
                                    horizontal_flip=True
                                    )

# Preprocess data
X_screw = img_to_array(screw_img)  
X_screw = X_screw.reshape((1,) + X_screw.shape)

X_nail = img_to_array(nail_img)  
X_nail = X_nail.reshape((1,) + X_nail.shape) 

# Apply transformation
i = 0
print('Transfomed Screw Images')
for batch in data_generator.flow(X_screw):
    i += 1
    plt.imshow(array_to_img(batch[0]))
    plt.show()
    if i % 8 == 0:  # Generate eight transformed pictures
        break  # To avoid generator to loop indefinitely
        
i = 0
print('Transfomed Nail Images')
for batch in data_generator.flow(X_nail):
    i += 1
    plt.imshow(array_to_img(batch[0]))
    plt.show()
    if i % 8 == 0:  # Generate eight transformed pictures
        break  # To avoid generator to loop indefinitely
        



