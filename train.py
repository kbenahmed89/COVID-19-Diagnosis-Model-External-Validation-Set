# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 09:49:30 2020

@author: rra-kbenahmed
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 09:13:23 2020

@author: rra-kbenahmed
"""
from sklearn.preprocessing import LabelBinarizer
import os, sys
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import keras
import tensorflow 
import numpy as np
np.random.seed(2020)
import scipy
import os
import glob
import math
import pickle
import datetime
#import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,AveragePooling2D, Conv2D,MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense,Input
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,AveragePooling2D, Conv2D,MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense,Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
# dimensions of our images.
from imutils import paths
#from tensorflow.keras.layers import Flatten, Dense, Input
#from keras_vggface.vggface import VGGFace
import random
import cv2
from sklearn.metrics import confusion_matrix
from math import pi
from math import cos
from math import floor
from keras import backend



train_data_dir = '/COVID/SpNih/train/'
seen = '/COVID/SpNih/val-test-seen-cropped/'
unseen = '/COVID/SpNih/val-test-unseen-cropped/'

nb_epoch = 1000
img_rows, img_cols, img_channel = 224,224, 3
input_tensor_shape=(img_rows, img_cols, img_channel)
BS = 25
data=[]
labels= []

# grab the image paths and randomly shuffle them
print("[INFO] Calculating statistics...")
imagePaths = sorted(list(paths.list_images(train_data_dir)))
random.seed(14)
random.shuffle(imagePaths)
# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_rows, img_rows))
    image = img_to_array(image)
    data.append(image)

data = np.array(data)

m=np.mean(data)
s=np.std(data)
mstd =  {'mean': 0.0, 'std': 1.0}

mstd['mean'] = m
mstd['std'] = s
with open('MeanSTD_seen', 'wb') as file_pi:
        pickle.dump(mstd, file_pi)
data=[]
labels=[]

# grab the image paths and randomly shuffle them
print("[INFO] loading training images...")
imagePaths = sorted(list(paths.list_images(train_data_dir)))
random.seed(14)
random.shuffle(imagePaths)
# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_rows, img_rows))
    image = img_to_array(image)
    image= (image - m)/s
    image = image.astype(np.float32)
    data.append(image)
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

data = np.array(data)
labels = np.array(labels)
# binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
print("training data: ",len(data))



#####data = (data - mstd['mean']) / mstd['std']

data_s=[]
labels_s= []

# grab the image paths and randomly shuffle them
print("[INFO] loading seen images...")
imagePaths = sorted(list(paths.list_images(seen)))
random.seed(14)
random.shuffle(imagePaths)
# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_rows, img_rows))
    image = img_to_array(image)
    image= (image - m)/s
    image = image.astype(np.float32)
    data_s.append(image)
    label = imagePath.split(os.path.sep)[-2]
    labels_s.append(label)

data_s = np.array(data_s)
labels_s = np.array(labels_s)
# binarize the labels
lb = LabelBinarizer()
labels_s = lb.fit_transform(labels_s)
print("seen data: ",len(data_s))


data_us=[]
labels_us= []

# grab the image paths and randomly shuffle them
print("[INFO] loading unseen images...")
imagePaths = sorted(list(paths.list_images(unseen)))
random.seed(14)
random.shuffle(imagePaths)
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_rows, img_rows))
    image = img_to_array(image)
    image = image.astype(np.float32)
    data_us.append(image)

data_us = np.array(data_us)
m_us=np.mean(data_us)
s_us=np.std(data_us)

mstd_un =  {'mean': 0.0, 'std': 1.0}
mstd_un['mean'] = m_us
mstd_un['std'] = s_us
with open('MeanSTD_unseen', 'wb') as file_pi:
        pickle.dump(mstd_un, file_pi)

data_us=[]
labels_us= []

# grab the image paths and randomly shuffle them
print("[INFO] loading unseen images...")
imagePaths = sorted(list(paths.list_images(unseen)))
random.seed(14)
random.shuffle(imagePaths)
# loop over the input images
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_rows, img_rows))
    image = img_to_array(image)
    image= (image - m_us)/s_us
    image = image.astype(np.float32)
    data_us.append(image)
    label = imagePath.split(os.path.sep)[-2]
    labels_us.append(label)

data_us = np.array(data_us)
labels_us = np.array(labels_us)
# binarize the labels
lb = LabelBinarizer()
labels_us = lb.fit_transform(labels_us)
print("unseen data: ",len(data_us))


base_model = ResNet50(weights='imagenet', include_top=False, input_shape= input_tensor_shape)

#for layer in base_model.layers:
#   if not layer.name.startswith("conv5_block3") and  not isinstance(layer, BatchNormalization):
#      layer.trainable  = False

#for layer in base_model.layers:
#   if not layer.name.startswith("conv5_block3"):
#      layer.trainable  = False

for layer in base_model.layers:
   if not isinstance(layer, BatchNormalization):
      layer.trainable  = False


headModel = base_model.output
headModel = GlobalAveragePooling2D()(headModel)
headModel = Dense(64, activation='relu')(headModel)
headModel = BatchNormalization()(headModel)
headModel = Dense(1, activation='sigmoid')(headModel)
model = Model(inputs=base_model.input, outputs=headModel)

print(model.summary())

#for i, layer in enumerate(model.layers):
#   print(i, layer.name, layer.trainable)

#opt =Adam( lr=0.000001)
opt = SGD(momentum=0.9)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# define custom learning rate schedule
class CosineAnnealingLearningRateSchedule(keras.callbacks.Callback):
	# constructor
	def __init__(self, n_epochs, n_cycles, lrate_max, verbose=0):
		self.epochs = n_epochs
		self.cycles = n_cycles
		self.lr_max = lrate_max
		self.lrates = list()
 
	# calculate learning rate for an epoch
	def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):
		epochs_per_cycle = floor(n_epochs/n_cycles)
		cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
		return lrate_max/2 * (cos(cos_inner) + 1)
 
	# calculate and set learning rate at the start of the epoch
	def on_epoch_begin(self, epoch, logs=None):
		# calculate learning rate
		lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_max)
		# set learning rate
		backend.set_value(self.model.optimizer.lr, lr)
		# log value
		self.lrates.append(lr)

class SnapshotEnsemble(keras.callbacks.Callback):
	# constructor
	def __init__(self, n_epochs, n_cycles, lrate_max, verbose=0):
		self.epochs = n_epochs
		self.cycles = n_cycles
		self.lr_max = lrate_max
		self.lrates = list()
 
	# calculate learning rate for epoch
	def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):
		epochs_per_cycle = floor(n_epochs/n_cycles)
		cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
		return lrate_max/2 * (cos(cos_inner) + 1)
 
	# calculate and set learning rate at the start of the epoch
	def on_epoch_begin(self, epoch, logs={}):
		# calculate learning rate
		lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_max)
		# set learning rate
		backend.set_value(self.model.optimizer.lr, lr)
		# log value
		self.lrates.append(lr)
 
	# save models at the end of each cycle
	def on_epoch_end(self, epoch, logs={}):
		# check if we can save model
		epochs_per_cycle = floor(self.epochs / self.cycles)
		if epoch != 0 and (epoch + 1) % epochs_per_cycle == 0:
			# save model to file
			filename = "snapshot_model_SpNih_%d.h5" % int((epoch + 1) / epochs_per_cycle)
			self.model.save(filename)
			print('>saved snapshot %s, epoch %d' % (filename, epoch))


filepath="res-meanuseen-cosine-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1)


class AdditionalValidationSets(keras.callbacks.Callback):
    def __init__(self, n_epochs, n_cycles, validation_sets, verbose=1, batch_size=None):
        """
        :param validation_sets:
        a list of 3-tuples (validation_data, validation_targets, validation_set_name)
        or 4-tuples (validation_data, validation_targets, sample_weights, validation_set_name)
        :param verbose:
        verbosity mode, 1 or 0
        :param batch_size:
        batch size to be used when evaluating on the additional datasets
        """
        super(AdditionalValidationSets, self).__init__()
        self.validation_sets = validation_sets
        for validation_set in self.validation_sets:
            if len(validation_set) not in [3, 4]:
                raise ValueError()
        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.batch_size = batch_size
        self.epochs = n_epochs
        self.cycles = n_cycles
    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        epochs_per_cycle = floor(self.epochs / self.cycles)
        if epoch != 0 and (epoch + 1) % epochs_per_cycle == 0:
            # evaluate on the additional validation sets
            for validation_set in self.validation_sets:
                if len(validation_set) == 3:
                    validation_data, validation_targets, validation_set_name = validation_set
                    sample_weights = None
                elif len(validation_set) == 4:
                    validation_data, validation_targets, sample_weights, validation_set_name = validation_set
                else:
                    raise ValueError()
    
                results = self.model.evaluate(x=validation_data,
                                              y=validation_targets,
                                              verbose=self.verbose,
                                              sample_weight=sample_weights,
                                              batch_size=self.batch_size)
                pred = self.model.predict(validation_data)
                y_pred = [1 * (x[0]>=0.5) for x in pred]
                c = confusion_matrix(validation_targets, y_pred)
                print('Confusion matrix:\n', c)
                valuename = validation_set_name+'_confusion_matrix'
                self.history.setdefault(valuename, []).append(c)
    
                for metric, result in zip(self.model.metrics_names,results):
                    valuename = validation_set_name + '_' + metric
                    self.history.setdefault(valuename, []).append(result)

n_cycles = nb_epoch / 50
#ca = CosineAnnealingLearningRateSchedule(nb_epoch, n_cycles, 0.0001)
ca = SnapshotEnsemble(nb_epoch, n_cycles, 0.0001)
hist = AdditionalValidationSets(nb_epoch, n_cycles,[(data_s, labels_s, 'seen'),(data_us, labels_us, 'unseen')])


#callbacks_list = [hist,checkpoint,ca]
callbacks_list = [hist,ca]

H =  model.fit(data,labels,batch_size=BS,
                        epochs=nb_epoch,
                        callbacks=callbacks_list, verbose=1)
#print(hist.history.keys())
#print(hist.history["unseen_accuracy"])

with open('trainHistoryDict-spnih', 'wb') as file_pi:
        pickle.dump(H.history, file_pi)

with open('TestHistoryDict-spnih', 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)


# plot the training loss and accuracy
#plt.style.use("ggplot")
#plt.figure()
#plt.plot(ca.lrates)
#plt.savefig("lr")
