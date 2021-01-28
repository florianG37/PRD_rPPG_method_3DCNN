# Task 1 Research and Development Project : Development of a new rPPG method to be integrated into the pyVHR framework

<em> Author : Florian GIGOT </em>

Repository architecture :

* notebooks --> Research, explanations, tests

    * Model folder --> Trained model files for experimentation
    * Train_3DCNN_model_BPM --> Jupyter notebook of the new method (3DCNN) implementation (with explanations)
    * Predict_3DCNN_model_BPM --> Jupyter notebook to make predictions with the model with [pyVHR framework](https://github.com/phuselab/pyVHR) constraints (with explanations).
    * Generating_training_data_with_GT --> Jupyter Notebook to create a training dataset from traditional rPPG datasets. Judge its relevance to improve the training of our model
