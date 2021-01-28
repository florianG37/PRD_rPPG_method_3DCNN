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

config.read('./training.cfg')

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
Setting of training session
"""

# CONSTANTS
NB_VIDEOS_BY_CLASS_TRAIN = int(config['dataConfig']['nbVideosPerClassInTraining'])

EPOCHS = int(config['modelConfig']['epochs'])

VERBOSE = int(config['modelConfig']['verbose'])

BATCH_SIZE = int(config['modelConfig']['batchSize'])

NB_SESSION = int(config['modelConfig']['nbSession'])

CONTINUE_TRAINING = False
if (int(config['modelConfig']['continueTraining']) == 1):
    CONTINUE_TRAINING = True #load or not a old trained model


MIXED_DATA = False
if (int(config['modelConfig']['mixedData']) == 1):
    MIXED_DATA = True #use both types of datasets


RESULTS_PATH = str(config['modelConfig']['modelPath'])

REAL_VIDEO_DATASET = str(config['dataConfig']['realVideoDataset'])

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
Definition of the 3D CNN model
"""
# DEFINE MODEL
model = Sequential()

#feature extraction
model.add(Conv3D(filters=32, kernel_size=(LENGTH_VIDEO-2,IMAGE_HEIGHT-5,IMAGE_WIDTH-5), input_shape=(LENGTH_VIDEO, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)))
model.add(MaxPooling3D(pool_size=(2,2,2)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Flatten())

#Classification
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(NB_CLASSES + 1, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Definition of the 3D CNN model.. done")

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

xtrain = np.zeros(shape=((NB_CLASSES + 1) * NB_VIDEOS_BY_CLASS_TRAIN, LENGTH_VIDEO, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
ytrain = np.zeros(shape=((NB_CLASSES + 1) * NB_VIDEOS_BY_CLASS_TRAIN, NB_CLASSES + 1))


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
Train Data generation
"""

def dataGeneration(xtrain, ytrain):
    print("start : Train Data generation ...")

    c = 0

    for i_freq in range(len(HEART_RATES)):

        for i_videos in range(NB_VIDEOS_BY_CLASS_TRAIN):

            t2 = t + (np.random.randint(low=0, high=33) * SAMPLING)   # phase
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
                xtrain[c,j,:,:,0] = temp.astype('uint8') / 255.0

            xtrain[c] = xtrain[c] - np.mean(xtrain[c])
            ytrain[c] = labels_cat[i_freq]

            c = c + 1

    # constant image noise (gaussian distribution)
    for i_videos in range(NB_VIDEOS_BY_CLASS_TRAIN):
        r = np.random.randint(low=0, high=len(TENDANCIES_MAX))
        trend = generate_trend(len(t), TENDANCIES_ORDER[r], 0, np.random.uniform(low=TENDANCIES_MIN[r], high=TENDANCIES_MAX[r]), np.random.randint(low=0, high=2))

        # add a tendancy on noise
        signal = np.expand_dims(trend, 1)
        img = np.tile(signal, (IMAGE_WIDTH, 1, IMAGE_HEIGHT)) / (IMAGE_HEIGHT * IMAGE_WIDTH)
        img = np.expand_dims(np.transpose(img, axes=(1,0,2)), 3)

        xtrain[c] = np.expand_dims(np.random.normal(size=(LENGTH_VIDEO, IMAGE_HEIGHT, IMAGE_WIDTH)) / 50, 3) + img
        xtrain[c] = xtrain[c] - np.mean(xtrain[c])
        ytrain[c] = labels_cat[NB_CLASSES]
        c = c + 1
    print('Train data generation .. done')
    return xtrain , ytrain

"""
Start a training session + save
"""
print("Start a training session")

if (CONTINUE_TRAINING == True):
    model = model_from_json(open(f'{RESULTS_PATH}/model_conv3D.json').read())
    model.load_weights(f'{RESULTS_PATH}/weights_conv3D.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

for serie in range(NB_SESSION):

    print("Session " + str(serie))
    xtrain, ytrain = dataGeneration(xtrain, ytrain)

    if(MIXED_DATA == True):
        data = np.load(REAL_VIDEO_DATASET)
        xtrain = np.concatenate((xtrain, data['a']), axis=0)
        ytrain =  np.concatenate((ytrain, data['b']), axis=0)

        indices = np.arange(xtrain.shape[0])
        np.random.shuffle(indices)
        xtrain = xtrain[indices]
        ytrain = ytrain[indices]
    
    #start training
    model.fit(xtrain, ytrain, epochs = EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)

    #save data
    model_json = model.to_json()
    open(f'{RESULTS_PATH}/model_conv3D.json', 'w').write(model_json)
    model.save_weights(f'{RESULTS_PATH}/weights_conv3D.h5', overwrite=True)
    print('A new model has been saved!\n')
