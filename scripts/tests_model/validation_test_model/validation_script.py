"""
Importing librairies
"""
#Tensorflow/KERAS
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.layers import ZeroPadding3D, Dense, Activation,Conv3D,MaxPooling3D,AveragePooling3D,Flatten,Dropout
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.models import load_model

# Numpy / Matplotlib / OpenCV / Scipy
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io
import configparser

"""
Loading configuration
"""
config = configparser.ConfigParser()

config.read('./validation.cfg')

print("Loading configuration.. done")

"""
Setting of input data
"""
winSize = int(config['dataConfig']['winSize'])
videoRate = int(config['dataConfig']['videoRate'])

LENGTH_VIDEO = winSize * videoRate #frames
IMAGE_WIDTH = int(config['dataConfig']['imageWidth']) #pixels
IMAGE_HEIGHT = int(config['dataConfig']['imageHeight']) #pixels
IMAGE_CHANNELS = int(config['dataConfig']['imageChannels']) #1 or 2 or 3

#Time notion
SAMPLING = 1 / videoRate #30 Hz
t = np.linspace(0, LENGTH_VIDEO * SAMPLING - SAMPLING, LENGTH_VIDEO)

print("Setting of input data.. done")

"""
Setting of output data
"""
minBPM = int(config['dataConfig']['minBPM'])
maxBPM = int(config['dataConfig']['maxBPM'])
nbLabels = int(config['dataConfig']['nbLabels'])

# Available Outputs
HEART_RATES = np.linspace(minBPM, maxBPM, nbLabels)
NB_CLASSES = len(HEART_RATES)

# prepare labels and label categories
labels = np.zeros(NB_CLASSES + 1)

print("Setting of output data.. done")

"""
Setting of validation session
"""
NB_VIDEOS_BY_CLASS_VALIDATION = int(config['dataConfig']['nbVideoPerClassInValidation'])

VERBOSE = int(config['modelConfig']['verbose'])

MIXED_DATA = False
if (int(config['modelConfig']['mixedData']) == 1):
    MIXED_DATA = True #use both types of datasets


RESULTS_PATH = str(config['modelConfig']['modelPath'])

REAL_VIDEO_DATASET = config.get('dataConfig', 'realVideoDataset').split(',')

useCPU  = int(config['modelConfig']['useCPU'])
if(useCPU == 1):
    #RUN ON CPU 
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print("Setting of training session.. done")

"""
Binarization of classes
"""
for i in range(NB_CLASSES + 1):
    labels[i] = i
labels_cat = np_utils.to_categorical(labels)

print("Binarization of classes.. done")

"""
Trend generation parameters for artificial data creation
"""


# Tendencies (linear, 2nd order, 3rd order)
TENDANCIES_MIN = (-3,-1,-1)
TENDANCIES_MAX = (3,1,1)
TENDANCIES_ORDER = (1,2,3)

# coefficients for the fitted-ppg method
a0 = 0.440240602542388
a1 = -0.334501803331783
b1 = -0.198990393984879
a2 = -0.050159136439220
b2 = 0.099347477830878
w = 2 * np.pi

"""
Trend generation function for artificial data creation
"""
def generate_trend(length, order, min, max, offset):

    if (order==1):   # linear
        tend = np.linspace(min, max, length)

    elif (order==2): # quadratic
        if (offset==0):
            tend = np.linspace(0, 1, length)
            tend = tend*tend
            tend = tend-min
            tend = max*tend/np.max(tend)

        else:
            tend = tend = np.linspace(-0.5, 0.5, length)
            tend = tend*tend
            tend = tend-min
            tend = 0.5*max*tend/np.max(tend)

    elif (order==3): # cubic
        if (offset==0):
            tend = np.linspace(0, 1, length)
            tend = tend*tend*tend
            tend = tend-min
            tend = max*tend/np.max(tend)

        else:
            tend = tend = np.linspace(-0.5, 0.5, length)
            tend = tend*tend*tend
            tend = tend-min
            tend = 0.5*max*tend/np.max(tend)
    return tend

"""
Validation Data generation
"""
print("start : Validation Data generation ...")

xvalidation = np.zeros(shape=((NB_CLASSES + 1) * NB_VIDEOS_BY_CLASS_VALIDATION, LENGTH_VIDEO, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
yvalidation = np.zeros(shape=((NB_CLASSES + 1) * NB_VIDEOS_BY_CLASS_VALIDATION, NB_CLASSES + 1))

c = 0

# for each frequency
for i_freq in range(len(HEART_RATES)):

    for i_videos in range(NB_VIDEOS_BY_CLASS_VALIDATION):

        t2 = t + (np.random.randint(low=0, high=33) * SAMPLING)   # phase. 33 corresponds to a full phase shift for HR=55 bpm
        signal = a0 + a1 * np.cos(t2 * w * HEART_RATES[i_freq] / 60) + b1 * np.sin(t2 * w * HEART_RATES[i_freq] / 60) + a2 * np.cos(2 * t2 * w * HEART_RATES[i_freq] / 60) + b2 * np.sin(2 * t2 * w * HEART_RATES[i_freq] / 60)
        signal = signal - np.min(signal)
        signal = signal / np.max(signal)

        r = np.random.randint(low=0, high=len(TENDANCIES_MAX))
        trend = generate_trend(len(t), TENDANCIES_ORDER[r], 0, np.random.uniform(low=TENDANCIES_MIN[r], high=TENDANCIES_MAX[r]), np.random.randint(low=0, high=2))

        signal = np.expand_dims(signal + trend, 1)
        signal = signal - np.min(signal)

        img = np.tile(signal, (IMAGE_WIDTH, 1, IMAGE_HEIGHT))
        img = np.transpose(img, axes=(0,2,1))

        img = img / (IMAGE_HEIGHT * IMAGE_WIDTH)
        
        amplitude = np.random.uniform(low=1.5, high=4)
        noise_energy = amplitude * 0.25 * np.random.uniform(low=1, high=10) / 100

        for j in range(0, LENGTH_VIDEO):
            temp = 255 * ((amplitude * img[:,:,j]) + np.random.normal(size=(IMAGE_HEIGHT, IMAGE_WIDTH), loc=0.5, scale=0.25) * noise_energy)
            temp[temp < 0] = 0 
            xvalidation[c,j,:,:,0] = temp.astype('uint8') / 255.0

        xvalidation[c] = xvalidation[c] - np.mean(xvalidation[c])
        yvalidation[c] = labels_cat[i_freq]

        c = c + 1


# constant image noise (gaussian distribution)
for i_videos in range(NB_VIDEOS_BY_CLASS_VALIDATION):
    r = np.random.randint(low=0, high=len(TENDANCIES_MAX))
    trend = generate_trend(len(t), TENDANCIES_ORDER[r], 0, np.random.uniform(low=TENDANCIES_MIN[r], high=TENDANCIES_MAX[r]), np.random.randint(low=0, high=2))

    # add a tendancy on noise
    signal = np.expand_dims(trend, 1)
    img = np.tile(signal, (IMAGE_WIDTH, 1, IMAGE_HEIGHT)) / (IMAGE_HEIGHT * IMAGE_WIDTH)
    img = np.expand_dims(np.transpose(img, axes=(1,0,2)), 3)

    xvalidation[c] = np.expand_dims(np.random.normal(size=(LENGTH_VIDEO, IMAGE_HEIGHT, IMAGE_WIDTH)) / 50, 3) + img
    xvalidation[c] = xvalidation[c] - np.mean(xvalidation[c])
    yvalidation[c] = labels_cat[NB_CLASSES]
    c = c + 1

print('Validation data generation done')

"""
Start a validation session
"""
print("Loading model ..")
model = model_from_json(open(f'{RESULTS_PATH}/model_conv3D.json').read())
model.load_weights(f'{RESULTS_PATH}/weights_conv3D.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Loading model .. done")  

print("Test Validation : Synthetic dataset")  

model.evaluate(xvalidation, yvalidation, verbose=VERBOSE)

if(MIXED_DATA == True):

    # manage several data files
    for i in range(len(REAL_VIDEO_DATASET)):  

        del xvalidation
        del yvalidation
  
        data = np.load(str(REAL_VIDEO_DATASET[i]))

        xvalidation = data['c']
        yvalidation = data['d']

        print("Test Validation : " + str(REAL_VIDEO_DATASET[i]) +" dataset")  

        model.evaluate(xvalidation, yvalidation, verbose=VERBOSE)



