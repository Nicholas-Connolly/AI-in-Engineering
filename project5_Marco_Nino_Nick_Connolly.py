# ME:4150 Artificial Intelligence in Engineering
# Project 5: Image Classification
# Marco Nino, Nick Connolly

import keras
import os, shutil

# Directories for training, validation, and test sets
train_dir = './train'
valid_dir ='./validation'
test_dir = './test'

### The below is not necessary
# Directory with the training nail and screw pictures
train_nails_dir = os.path.join(train_dir, 'nail')
train_screws_dir = os.path.join(train_dir, 'screw')

# Directory with the validation nail and screw pictures
valid_nails_dir = os.path.join(valid_dir, 'nail')
valid_screws_dir = os.path.join(valid_dir, 'screw')

# Directory with the test nail and screw pictures
test_nails_dir = os.path.join(test_dir, 'nail')
test_screws_dir = os.path.join(test_dir, 'screw')

print('Total training nail images:', len(os.listdir(train_nails_dir)))
print('Total training screw images:', len(os.listdir(train_screws_dir)))
print('Total validation nail images:', len(os.listdir(valid_nails_dir)))
print('Total validation screw images:', len(os.listdir(valid_screws_dir)))
print('Total test nail images:', len(os.listdir(test_nails_dir)))
print('Total test screw images:', len(os.listdir(test_screws_dir)))
### The above is not necessary

# Preprocess the images: read the image files; 
# decode the jpg to RBG grids of pixels (150x150); 
# convert to float point tensors; rescale the pixel values

from keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
print('\n')
print('Preprocess the training set')
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        directory = train_dir,      # The target directory
        target_size=(150, 150),     # Being resized to 150x150
        batch_size=20,
        class_mode='binary',        # Binary classification
        seed = 63
        )

# Each batch has 20 samples, and each sample is an 150x150 RGB image 
# (shape 150,150,3) and binary labels.
print('\n')
print('In the first batch')
(data_batch, labels_batch) = train_generator[0]
print('Data batch shape:', data_batch.shape)
print('Labels batch shape:', labels_batch.shape)

# preprocess the validation set
print('\n')
print('Preprocess the validation set')
valid_datagen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_datagen.flow_from_directory(
        directory = valid_dir,      
        target_size=(150, 150),     
        batch_size=20,
        class_mode='binary',        
        seed = 63
        )

# preprocess the test set
print('\n')
print('Preprocess the test set')
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        directory = test_dir,      
        target_size=(150, 150),     
        batch_size=1,
        class_mode='binary',  
        shuffle = False,
        seed = 63
        )

# build a CNN
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))  
model.add(layers.MaxPooling2D((2, 2))) # stride 2 (downsampled by a factor of 2)
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten()) # Flatten the 3D outputs to 1D before adding a few Dense layers
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid')) # binary classificaiton
model.summary()

# configure the model
from keras import optimizers
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# train the model
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,      # 2000/20
      epochs=5,
      validation_data=valid_generator,
      validation_steps=50       # 1000/20
      )

model.save('cat_and_dog.h5')

# plot the training and validation scores
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# evaludate the model
eval_test = model.evaluate_generator(generator=test_generator, steps=1000)
print("The test score (accuracy) is {}%".format(eval_test[1]*100))

# output the predictions compared to the targets
import numpy as np
import pandas as pd
test_generator.reset()
pred=model.predict_generator(test_generator, steps=100, verbose=1)
predicted_class_indices= np.where(pred > 0.5, 1, 0)
predicted_class_indices=predicted_class_indices.reshape(-1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results-NailScrew.csv",index=False)
