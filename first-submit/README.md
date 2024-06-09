# Tensorflow example code for the ACM/IEEE Intelligent Embedded System Design Contest at ICMC 2024

## What's in this repository?

This repository contains a simple example to illustrate how to train the model with tensorflow and evaluate the comprehensive performances in terms of detection performance, flash occupation and latency. You can try it by running the following commands on the given training dataset. 

For this example, we implemented a convolutional neural network (CNN). You can use a different classifier and software for your implementation. 

This code uses four main scripts, described below, to train and test your model for the given dataset.

## How do I run these scripts?

You can run this classifier code by installing the requirements

    pip install requirements.txt

and running

    python training_save_deep_models_tf.py 
    python testing_performances_tf.py

where `models` is a folder of model structure file, `saved_models` is a folder for saving your models, `data_indices` is a folder of data indices (the given dataset has been partitioned into training and testing dataset, you can create more partitions of the training data locally for debugging and cross-validation), and `records` is a folder for saving the statistics outputs. The [IESD Contest 2024 webpage](https://iesdcontest.github.io/iesd-2024/Problems.html) provides a description of the data files.

After running the scripts, one of the scoring metrics (i.e., **F-B**) will be reported in the file *seg_stat.txt* in the folder `records`. 

