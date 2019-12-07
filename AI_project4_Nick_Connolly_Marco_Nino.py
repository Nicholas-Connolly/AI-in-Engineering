# ME:4150 Artificial Intelligence in Engineering
# Project 4: ANN - movie review classification
# Nick Connolly and Marco Nino

import keras
import matplotlib.pyplot as plt

from keras.datasets import imdb
from sklearn.model_selection import train_test_split

(train_data_, train_labels_), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Split training set into train and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(train_data_,
        train_labels_, random_state = 4)

# Here is how to decode one review back to English words
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[100]])

# One-hot-encoding
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results

# Vectorizing the dataset
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
x_val = vectorize_sequences(val_data)

y_train = train_labels
y_test = test_labels
y_val = val_labels


from keras import models
from keras import layers

# Initialize neural network with two hidden layers. 
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='tanh'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Configure the network 
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model using hold-out validation and store its history
history = model.fit(x_train, y_train, epochs=20, batch_size=1024, validation_data = (x_val, y_val))
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# Plot training score and validation score
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluation
test_loss, test_acc = model.evaluate(x_test, y_test)
print('The test score is {:.2f}%'.format(test_acc*100))
print(decoded_review)


"""
We notice that the validation score begins to drop off after about four epochs.
This indicates that the model is overfitting on the training data past this point. 
"""