'''
https://github.com/llSourcell/how_to_deploy_a_keras_model_to_production/blob/master/train.py


Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
#python 2/3 compatibility
from __future__ import print_function
#simplified interface for building models 
import keras
#our handwritten character labeled dataset
from keras.datasets import mnist
#because our models are simple
from keras.models import Sequential
#dense means fully connected layers, dropout is a technique to improve convergence, flatten to reshape our matrices for feeding
#into respective layers
from keras.layers import Dense, Dropout, Flatten
#for convolution (images) and pooling is a technique to help choose the most relevant features in an image
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K





# Import Numpy, TensorFlow, TFLearn, and MNIST data
import numpy as np
import tensorflow as tf
import tflearn
from PIL import Image
import os
from os import listdir
from sklearn.preprocessing import LabelBinarizer
import pickle
from sklearn.model_selection import train_test_split
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# Set or words to train on
WORDS = set([',','.','the','of','to'])
# Or, set FULL_SET to True to use the full set of data
FULL_SET = True


def load_data():
     with open("words.txt") as fp:
      words = []
      labels = []
      pictures = []
      for line in fp:
           word = line.split(' ')[8][:-1] # chop of the last char, which is always newline
           if FULL_SET or word in WORDS:
              # add actual word to labels
              fn = line.split(' ')[0]
              im = Image.open("words/" + fn +'.png') 
              labels.append(word)
              # resize to 28x28 (arbitrarily)
              im28 = im.resize((28, 28))
              pictures.append(im28)

              # flatten the image into a numpy array
              word = np.array(im28, dtype=np.float32).flatten()
              words.append(word)

      return (np.array(words), np.array(labels), pictures)


# https://github.com/udacity/deep-learning/blob/master/intro-to-tensorflow/intro_to_tensorflow_solution.ipynb
def normalize_grayscale(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    a = 0.1
    b = 0.9
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )


def one_hot_encode(training_labels):
   # Turn labels into numbers and apply One-Hot Encoding
   encoder = LabelBinarizer()
   encoder.fit(training_labels)
   training_labels = encoder.transform(training_labels)

   # Change to float32, so it can be multiplied against the features in TensorFlow, which are float32
   training_labels = training_labels.astype(np.float32)
   print('Labels One-Hot Encoded')
   return training_labels

# converts categorical data, say words, to integers
def convert_to_int(arr):
   new_arr = []
   d = {}
   num_to_word = {}
   next_label = 0
   for item in arr:
      if item in d:
         new_arr.append(d[item])
      else:
         d[item] = next_label
         num_to_word[next_label] = item
         new_arr.append(next_label)
         next_label += 1
   # save as pickle for easy use later
   with open('num_to_word.pickle', 'wb') as handle:
      pickle.dump(num_to_word, handle, protocol=pickle.HIGHEST_PROTOCOL)
   return (np.array(new_arr), next_label, num_to_word)

def add_instances(unique_words, labels, words, num_to_word, pictures):

   print("Started add_instances")
   num_low = 0
   num_high = 0
   maxi = 0
   labels = list(labels)
   words  = list(words)

   # There is probably a faster way of doing this

   """
   # This was taking far too long to justify doing each time. Uncomment if the data gets switched

   # Find the word with the highest frequency ('the' in our case, 5826 instances)
   # We scale the other frequencies up relative to their distance from this value
   for item in range(unique_words):
      count = labels.count(item)
      if count > maxi:
         maxi = count
   print("Got the maximum frequency")
   
   """

   maxi = 5826

   for item in range(unique_words):
      # locations of the pictures of this word
      indices = [i for i, x in enumerate(labels) if x == item]
      count = len(indices)
      num_to_add = int(math.sqrt(math.sqrt(maxi-count)))
      word = num_to_word[item]
      # we want to sample from however many different images we ahve as much as we can
      cur_pic = 0
      for _ in range(num_to_add):
         im = pictures[indices[cur_pic]]

         # mess with the image a little
         off1, off2, off3, off4 = [random.randint(0,4) for _ in range(4)]
         im = im.transform((im.size[0], im.size[1]), Image.EXTENT,(0+off1, 0+off2, im.size[0]-off3, im.size[1]-off4))
         roll_to_rotate = random.randint(1,6)
         if roll_to_rotate > 2:
            amount_to_rotate = random.weibullvariate(1, 1.5)
            im.rotate(amount_to_rotate)

         # add the image to out set
         words.append(normalize_grayscale(np.array(im, dtype=np.float32).flatten()))
         labels.append(item)

         # advance to a new image if there is one
         cur_pic += 1
         if cur_pic == count:
            cur_pic = 0


   print(maxi)
   return (np.array(words), np.array(labels))


def main():
   LOAD_FROM_PICKLE = False
   if LOAD_FROM_PICKLE:
      with open('training_labels.pickle', 'rb') as handle:
         training_labels = pickle.load(handle)
      with open('training_words.pickle', 'rb') as handle:
         training_words = pickle.load(handle)
      with open('num_to_word.pickle', 'rb') as handle:
         num_to_word = pickle.load(handle)
      with open('num_classes.pickle', 'rb') as handle:
         num_classes = pickle.load(handle)
      with open('pictures.pickle', 'rb') as handle:
         pictures = pickle.load(handle)
   else:
      training_words, training_labels, pictures  = load_data()
      training_words  = normalize_grayscale(training_words)
      #training_labels = one_hot_encode(training_labels)
      training_labels, num_classes, num_to_word = convert_to_int(training_labels)
      #training_words, training_labels = add_instances(num_classes, list(training_labels), training_words, num_to_word, pictures)

      with open('training_labels.pickle', 'wb') as handle:
         pickle.dump(training_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
      with open('training_words.pickle', 'wb') as handle:
         pickle.dump(training_words, handle, protocol=pickle.HIGHEST_PROTOCOL)
      # this can't be effcient, saving an int as a pickle, but w/e
      with open('num_classes.pickle', 'wb') as handle:
         pickle.dump(num_classes, handle, protocol=pickle.HIGHEST_PROTOCOL)
      with open('pictures.pickle', 'wb') as handle:
         pickle.dump(pictures, handle, protocol=pickle.HIGHEST_PROTOCOL)
   #add_instances(num_classes, training_labels, training_words, num_to_word, pictures)
   #10/0
   # I really don't know what this is doing
   # is from sklearn.model_selection import train_test_split
   # Get randomized datasets for training and validation
   x_train, x_test, y_train, y_test = train_test_split(
       training_words,
       training_labels,
       test_size=0.05,
       random_state=832289)

   #mini batch gradient descent ftw
   batch_size = 128
 
   #very short training time
   epochs = 19
   # input image dimensions
   #28x28 pixel images. 
   img_rows, img_cols = 28, 28
   '''

   # the data downloaded, shuffled and split between train and test sets
   #if only all datasets were this easy to import and format
   (x_train, y_train), (x_test, y_test) = mnist.load_data()
   '''
   #this assumes our data format
   #For 3D data, "channels_last" assumes (conv_dim1, conv_dim2, conv_dim3, channels) while 
   #"channels_first" assumes (channels, conv_dim1, conv_dim2, conv_dim3).
   if K.image_data_format() == 'channels_first':
      x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
      x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
      input_shape = (1, img_rows, img_cols)
      print('did this')
   else:
      x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
      x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
      input_shape = (img_rows, img_cols, 1)

   #more reshaping
   x_train = x_train.astype('float32')
   x_test = x_test.astype('float32')
   x_train /= 255
   x_test /= 255
   print('x_train shape:', x_train.shape)
   print(x_train.shape[0], 'train samples')
   print(x_test.shape[0], 'test samples')
   print('y_train shape:', y_train.shape)
   print('y_test shape:', y_test.shape)

   # convert class vectors to binary class matrices
   y_train = keras.utils.to_categorical(y_train, num_classes)
   y_test = keras.utils.to_categorical(y_test, num_classes)
   print('y_train shape:', y_train.shape)
   print('y_test shape:', y_test.shape)
   #build our model
   model = Sequential()
   #convolutional layer with rectified linear unit activation
   model.add(Conv2D(32, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=input_shape))
   #again
   model.add(Conv2D(64, (3, 3), activation='relu'))
   #choose the best features via pooling
   model.add(MaxPooling2D(pool_size=(2, 2)))
   #randomly turn neurons on and off to improve convergence
   model.add(Dropout(0.25))
   #flatten since too many dimensions, we only want a classification output
   model.add(Flatten())
   #fully connected to get all relevant data
   model.add(Dense(128, activation='relu'))
   #one more dropout for convergence' sake :) 
   model.add(Dropout(0.5))
   #output a softmax to squash the matrix into output probabilities
   model.add(Dense(num_classes, activation='softmax'))
   #Adaptive learning rate (adaDelta) is a popular form of gradient descent rivaled only by adam and adagrad
   #categorical ce since we have multiple classes (10) 
   model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

   #train that ish!
   model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
   #how well did it do? 
   score = model.evaluate(x_test, y_test, verbose=0)
   print('Test loss:', score[0])
   print('Test accuracy:', score[1])


   #Save the model
   # serialize model to JSON
   model_json = model.to_json()
   with open("model.json", "w") as json_file:
      json_file.write(model_json)
   # serialize weights to HDF5
   model.save_weights("model.h5")
   print("Saved model to disk")
if __name__ == '__main__':
   main()
