# CarND-LeNet-Lab
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

![LeNet-5 Architecture](lenet.png)
Implement the LeNet-5 deep neural network model.

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Install

This project requires **Python** and the following Python libraries installed:

-[NumPy]（http://www.numpy.org/）

-[matplotlib]（http://matplotlib.org/）

-[scikit-learn]（http://scikit-learn.org/stable/）

[Tensorflow=1.14](pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.0.0-cp35-cp35m-win_x86_64.whl)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included

### Data

The data sets used in this project are contained in "traffic-signs-data". And contains the following attributes:

**Feature**

Input picture is 28 * 28 * 1
Training set samples: 55000
Number of validation set samples: 5000
Number of test set samples: 10000


**Target variable**

Class 10

### Analysis

The LeNet-5 model has a total of 7 layers. The structure of each layer of the LeNet-5 model is summarized below.

First layer: Convolution layer

Input: The original image pixel matrix (length, width, color), the size is 32 * 32 * 1.

Convolution layer parameters: The filter size is 5 * 5, the depth is 6, the padding is not all 0s, and the step size is 1.

Output: The size is 28 * 28 * 6.

Analysis: Because there is no all-zero padding, the output size of this layer is 32-5 + 1 = 28, and the depth is 6;There are 
5 * 5 * 1 * 6 + 6 = 156 parameters in this convolution layer, of which 6 are offset parameters;Because the node matrix of the next layer has 28 * 28 * 6 = 4704 nodes, each node is connected to 5 * 5 = 25 nodes of the current layer.So the convolutional layer in this layer has a total of 4704 * (25 + 1) = 122304 connections.

Second layer: pooling layer

Input: The size is 28 * 28 * 6.

Pooling layer parameters: The filter size is 2 * 2, and the step length is 2 for both length and width.

Output: the size is 14 * 14 * 6.

Analysis: The 6 feature maps include a total of 6 * 2 = 12 trainable parameters, and 14 * 14 * 6 * (2 * 2 + 1) = 5880 connections.

Third layer: Convolution layer

Input: The size is 14 * 14 * 6.

Convolution layer parameters: The filter size is 5 * 5, the depth is 16, the padding is not all 0s, and the step size is 1.

Output: The size is 10 * 10 * 16.

Analysis: Because there is no full 0 padding, the output size of this layer is 14-5 + 1 = 10, and the depth is 16;The convolutional layer has 5 * 5 * 6 * 16 + 16 = 2416 parameters, of which 16 are offset parameters;Because the node matrix of the next layer has 10 * 10 * 16 = 1600 nodes,each node is connected to 5 * 5 = 25 nodes of the current layer.So the convolutional layer in this layer has a total of 1600 * (25 + 1) = 41600 connections.

The fourth layer: pooling layer

Input: The size is 10 * 10 * 16.

Pooling layer parameters: The filter size is 2 * 2, and the step length is 2 for both length and width.

Output: The size is 5 * 5 * 16.

Analysis: The 16 feature maps contain a total of 16 * 2 = 32 trainable parameters and 5 * 5 * 16 * (2 * 2 + 1) = 2000 connections.

Fifth layer: Fully connected layer

Number of input nodes: 5 * 5 * 16 = 400.

Number of parameters: 5 * 5 * 16 * 120 + 120 = 48120.

Number of output nodes: 120.

Layer 6: Fully connected layer

Number of input nodes: 120.

Number of parameters: 120 * 84 + 84 = 10164.

Number of output nodes: 84.

Layer 7: Fully connected layer

Number of input nodes: 84.

Number of parameters: 84 * 10 + 10 = 850.

Number of output nodes: 10.

## Conclusion

Through this learning, I have a general understanding of the convolutional neural network. Compared with the 93% accuracy rate of handwriting recognition of SVM, the CNN algorithm greatly reduces the amount of calculation and greatly improves its accuracy. In this case, the accuracy rate of the CNN algorithm can reach 98.9% through only 10 iterations of training. This reflects the advantages of the CNN algorithm and has more advantages for image recognition processing, which greatly expands the application range of the CNN algorithm.