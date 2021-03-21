# Task 1 Research and Development Project : Development of a new rPPG method to be integrated into the pyVHR framework

<em> Author : Florian GIGOT </em>

## Objective

In this repository, we want to develop and test a new rPPG method in order to integrate it into pyVHR to compare our results with other rPPG methods.

## Installation of dependencies

    > cd scripts/setup
    > pip install -r requirements.txt

Note that a python version higher than 3.6 is required

## Repository architecture :

### Jupyter Notebooks - Documentation

* notebooks --> Research, explanations, tests

    
    * Train_3DCNN_model_BPM --> Jupyter notebook to train the model used by the new method (MAP_3DCNN) (with explanations)
    * Predict_3DCNN_model_BPM --> Jupyter notebook to make predictions with the method on a specific sequence of a real video (with explanations). 
    * BPM_estimation_on_real_video --> Jupyter notebook to make predictions with the method with [pyVHR framework](https://github.com/phuselab/pyVHR) constraints.
    * Generating_training_data_with_GT --> Jupyter Notebook to create a training dataset from traditional rPPG datasets (Pre-Processing = reshape). Judge its relevance to improve the training of our model
    * Generating_training_data_with_GT_2 --> Jupyter Notebook to create a training dataset from traditional rPPG datasets (here UBFC2 case but generalizable to others) with Pre-processing designed according to our prediction strategy detailed in Predict_3DCNN_model_BPM. 

### Final implementation - Code 

* scripts --> final code

    * "model" Folder --> Trained model files (final version)

    * "setup" Folder --> Installation tools
        * Installing_dependencies --> Install libraries for scripts (Windows File)
        * requirements --> List of used libraries + versions

    * "tests_model" Folder --> Testing tools
        * validation_script --> Script to launch a validation session
        * validation -->Script configuration file - validation_script
        * BPM_estimation_on_real_video --> Script to test on real data
        * BPMEstimationOnRealVideo --> Script configuration file - BPM_estimation_on_real_video

    * "training_model" Folder --> Training tools
        * training --> Script configuration file - training_script
        * training_script --> Script to launch a training session
        * generating_training_data_with_GT_script --> Script to create a training dataset from traditional rPPG datasets (with preprocessing).
        * generatingTrainingDatasetWithGT --> Script configuration file - generating_training_data_with_GT_script

### Example of use - experiments

* experimentation --> Different approaches to model training / Evaluation of the different methods

## Main sources / Reading materials

* G. Boccignone, D. Conte, V. Cuculo, A. D’Amelio, G. Grossi and R. Lanzarotti, "An Open Framework for Remote-PPG Methods and Their Assessment," in IEEE Access, vol. 8, pp. 216083-216103, 2020, doi: 10.1109/ACCESS.2020.3040936. ([Link](https://ieeexplore.ieee.org/document/9272290)) ([GitHub](https://github.com/phuselab/pyVHR))

* Frédéric Bousefsaf, Alain Pruski, Choubeila Maaoui, 3D convolutional neural networks for remote pulse rate measurement and mapping from facial video, Applied Sciences, vol. 9, n° 20, 4364 (2019). ([Link](https://www.mdpi.com/2076-3417/9/20/4364)) ([GitHub](https://github.com/frederic-bousefsaf/ippg-3dcnn))
