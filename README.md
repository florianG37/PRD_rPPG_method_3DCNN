# Task 1 Research and Development Project : Development of a new rPPG method to be integrated into the pyVHR framework

<em> Author : Florian GIGOT </em>

## Repository architecture :

* notebooks --> Research, explanations, tests

    * experimentation --> Different approaches to model training
    * Train_3DCNN_model_BPM --> Jupyter notebook of the new method (3DCNN) implementation (with explanations)
    * Predict_3DCNN_model_BPM --> Jupyter notebook to make predictions with the model with [pyVHR framework](https://github.com/phuselab/pyVHR) constraints (with explanations).
    * Generating_training_data_with_GT --> Jupyter Notebook to create a training dataset from traditional rPPG datasets (Processing = reshape). Judge its relevance to improve the training of our model

* scripts --> final code

    * Installing_dependencies --> Install libraries for scripts (Windows File)
    * requirements --> List of used libraries + versions
    * training --> Script configuration file
    * training_script --> Script to launch a training session
    * validation_script --> Script to launch a validation session
    * Model Folder --> Trained model files (final version)
    * BPM_estimation_on_real_video --> Script to test on real data
    * testOnRealVideo --> Script configuration file

## Main sources / Reading materials

* G. Boccignone, D. Conte, V. Cuculo, A. D’Amelio, G. Grossi and R. Lanzarotti, "An Open Framework for Remote-PPG Methods and Their Assessment," in IEEE Access, vol. 8, pp. 216083-216103, 2020, doi: 10.1109/ACCESS.2020.3040936. ([Link](https://ieeexplore.ieee.org/document/9272290)) ([GitHub](https://github.com/phuselab/pyVHR))

* Frédéric Bousefsaf, Alain Pruski, Choubeila Maaoui, 3D convolutional neural networks for remote pulse rate measurement and mapping from facial video, Applied Sciences, vol. 9, n° 20, 4364 (2019). ([Link](https://www.mdpi.com/2076-3417/9/20/4364)) ([GitHub](https://github.com/frederic-bousefsaf/ippg-3dcnn))
