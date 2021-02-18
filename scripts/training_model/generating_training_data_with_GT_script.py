##
## Importing Librairies
## 

#PyVHR Framework
from pyVHR.datasets.dataset import Dataset
from pyVHR.datasets.dataset import datasetFactory
from pyVHR.methods.base import methodFactory
from pyVHR.signals.video import Video
from pyVHR.datasets.ubfc2 import UBFC2
from pyVHR.datasets.ubfc1 import UBFC1
from pyVHR.datasets.pure import PURE
from pyVHR.datasets.lgi_ppgi import LGI_PPGI
from pyVHR.datasets.cohface import COHFACE

#Tensorflow/KERAS
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.layers import ZeroPadding3D, Dense, Activation,Conv3D,MaxPooling3D,AveragePooling3D,Flatten,Dropout
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.models import model_from_json

# Copy / numpy / OpenCV / ConfigParser
from copy import copy
import numpy as np
import cv2
import configparser

##
## Configuration
## 

# Loading configuration
config = configparser.ConfigParser()
config.read('./generatingTrainingDatasetWithGT.cfg')

# video config

IMAGE_WIDTH = int(config['dataConfig']['imageWidth']) 
IMAGE_HEIGHT = int(config['dataConfig']['imageHeight'])
IMAGE_CHANNELS = int(config['dataConfig']['imageChannels'])
RATE = int(config['dataConfig']['videoRate'])


WIN_SIZE = int(config['dataConfig']['winSize'])
STEP_X = int(config['dataConfig']['stepX'])
STEP_Y = int(config['dataConfig']['stepY'])

LENGTH_VIDEO = WIN_SIZE * RATE

minBPM = int(config['dataConfig']['minBPM'])
maxBPM = int(config['dataConfig']['maxBPM'])
nbLabels = int(config['dataConfig']['nbLabels'])
HEART_RATES = np.linspace(minBPM, maxBPM, nbLabels)
NB_CLASSES = len(HEART_RATES)

# dataset config
NAME_DATASET = str(config['datasetConfig']['name_dataset'])
FIRST_IDX = int(config['datasetConfig']['first_idx'])
LAST_IDX = int(config['datasetConfig']['last_idx']) + 1
SAVE_DATA = str(config['datasetConfig']['dataPath'])
SAVE_DATA_SPLITTED = str(config['datasetConfig']['splittedDataPath'])


print("Loading configuration.. done")

##
## Format the data of a video sequence into input data for a model
## 

def formating_data_test(video, imgs, start, end, step_x, step_y):
    
    xtemp = np.zeros(shape=(0, LENGTH_VIDEO, IMAGE_HEIGHT , IMAGE_WIDTH, 1 ))
    # Displacement on the x axis
    iteration_x = 0
    # Our position at n + 1 on the X axis
    axis_x = IMAGE_WIDTH
    
    # width of video
    width = video.cropSize[1]
    # height of video
    height = video.cropSize[0]
    
    # Browse the X axis
    while axis_x < width:
        # Displacement on the y axis
        axis_y = IMAGE_HEIGHT
        # Our position at n + 1 on the Y axis
        iteration_y = 0
        # Browse the Y axis
        while axis_y < height:
            
            # Start position
            x1 = iteration_x * step_x
            y1 = iteration_y * step_y
            
            # End position
            x2 = x1 + IMAGE_WIDTH
            y2 = y1 + IMAGE_HEIGHT
            
            # Cutting 
            face_copy = copy(imgs[start:end,x1:x2,y1:y2,:])
            
            # randomize pixel locations
            for j in range(LENGTH_VIDEO):
                temp = copy(face_copy[j,:,:,:])
                np.random.shuffle(temp)
                face_copy[j] = temp
            
            # Checks the validity of cutting
            if(np.shape(face_copy)[1] == IMAGE_WIDTH and np.shape(face_copy)[2] == IMAGE_HEIGHT):
                # prediction on the cut part
                face_copy = face_copy - np.mean(face_copy)
                xtest = np.expand_dims(face_copy, axis=0)
                xtemp = np.append(xtemp, xtest, axis=0)
                
            
            # increments
            axis_y = y2 + IMAGE_HEIGHT
            iteration_y = iteration_y +1
        # increments    
        axis_x = x2 + IMAGE_WIDTH
        iteration_x = iteration_x + 1
        
    return xtemp
##
## Management of frame rate differences by interpolation
## 
def interpolation(imgs, video):
    # find the number of missing images
    nb_seconds = int(video.numFrames / video.frameRate)
    diff_frames = nb_seconds * (RATE - video.frameRate)

    # adding images to a random place
    place_interpolation = np.random.randint(1, LENGTH_VIDEO, size=(diff_frames))
    for p in place_interpolation:
        imgs = np.insert(imgs, p, imgs[p], axis=0) 
    return imgs

##
## Protocol for transforming a video into a machine learning dataset
## Generate xtest & ytest from one video
##

def extract_data_from_video(video_filename, gt_filename, dataset):
    
    sig_gt = dataset.readSigfile(gt_filename)
    win_size_gt = WIN_SIZE
    bpm_gt, times_gt = sig_gt.getBPM(win_size_gt)
    
    # Format the GT
    bpm = np.round(bpm_gt)
    bpm = bpm - 55
    bpm = np.round(bpm / 2.5)
    
    #extraction
    video = Video(video_filename)
    video.getCroppedFaces(detector='dlib', extractor='skvideo')
    video.setMask(typeROI='skin_adapt',skinThresh_adapt=0.22)

    NB_LAPSE = int(video.numFrames / RATE)

    imgs = np.zeros(shape=(video.numFrames, video.cropSize[0], video.cropSize[1], 1))
    xtest = np.zeros(shape=(0, LENGTH_VIDEO, IMAGE_HEIGHT , IMAGE_WIDTH, 1))
    ytest = np.zeros(shape=(0, NB_CLASSES + 1))

    # prepare labels and label categories
    labels = np.zeros(NB_CLASSES + 1)

    for i in range(NB_CLASSES + 1):
        labels[i] = i
    labels_cat = np_utils.to_categorical(labels)
 
    # channel extraction
    if (video.cropSize[2]<3):
        IMAGE_CHANNELS = 1
    else:
        IMAGE_CHANNELS = video.cropSize[2]

    # load images (imgs contains the whole video)
    for j in range(video.numFrames):

        if (IMAGE_CHANNELS==3):
            temp = video.faces[j]/255
            temp = temp[:,:,1]      # only the G component is currently used
        else:
            temp = video.faces[j] / 255

        imgs[j] = np.expand_dims(temp, 2)

    # frameRate different from the model
    if (RATE > video.frameRate):
        imgs = interpolation(imgs, video)

    # Construction of sequences for each time interval
    for lapse in range(0,NB_LAPSE):  
    
        start = lapse * RATE
        end = start + LENGTH_VIDEO
        if(end > video.numFrames):
            break
        
        xtemp = formating_data_test(video, imgs,start, end, STEP_X, STEP_Y)
        
        #Sequence  
        xtest = np.append(xtest, xtemp, axis=0)
        #GT
        gt = np.expand_dims(labels_cat[int(bpm[lapse+int(WIN_SIZE/2)])], axis=0)
        
        for i in range(np.shape(xtemp)[0]):
            ytest = np.append(ytest, gt, axis=0)
        
    return xtest, ytest


##
## Applying the transformation on dataset
## 

def apply_transformation():
    dataset = datasetFactory(NAME_DATASET)

    xtrain = np.array(np.zeros(shape=(0,LENGTH_VIDEO, IMAGE_HEIGHT, IMAGE_WIDTH, 1)))
    ytrain = np.zeros(shape=(0, NB_CLASSES + 1))

    # For each video in the dataset
    for i in range (FIRST_IDX, LAST_IDX):
        print ("video : " + str(i))
        xtest, ytest = extract_data_from_video(dataset.videoFilenames[i], dataset.sigFilenames[i], dataset)
        print(np.shape(xtest))
        xtrain = np.concatenate((xtrain, xtest), axis=0)
        ytrain = np.concatenate((ytrain, ytest), axis=0)


    # Mix the sequences
    indices = np.arange(xtrain.shape[0])
    np.random.shuffle(indices)
    xtrain = xtrain[indices]
    ytrain = ytrain[indices]

    # save
    np.savez(SAVE_DATA, a=xtrain, b=ytrain)
    print("final dataset")
    print(np.shape(xtrain))

##
## Division into 1 test dataset and 1 validation dataset
## 

def spliting_data():

    data = np.load(SAVE_DATA)
    # (pct*100)% -> test & ((1-pct)*100)% -> validation
    pct = 0.9
    size_dataset = data['a'].shape[0]
    size_train_data = int(size_dataset * pct) 

    xtrain = data['a'][:size_train_data,:]
    xvalidation = data['a'][size_train_data:,:]

    ytrain = data['b'][:size_train_data,:]
    yvalidation = data['b'][size_train_data:,:]

    np.savez(SAVE_DATA_SPLITTED, a=xtrain, b=ytrain, c=xvalidation, d=yvalidation)

    print("Saving data .. done")
    print("Shape xtrain : " + str(np.shape(xtrain)))
    print("Shape ytrain : " + str(np.shape(ytrain)))
    print("Shape xvalidation : " + str(np.shape(xvalidation)))
    print("Shape yvalidation : " + str(np.shape(yvalidation)))

##
## MAIN
## 

apply_transformation()
spliting_data()