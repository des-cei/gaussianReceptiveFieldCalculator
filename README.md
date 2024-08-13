# Spiking Neural Networks with PyTorch
This repository includes libraries for the training and execution of Spiking Neural Networks (SNNs) using PyTorch. The **data preprocessing** module transform [UCI datasets](https://archive.ics.uci.edu/datasets) such as the Iris, Wine and WBC to spike trains using Gaussian Field Encoding. Then, the **SNN neuron** file contains definitions for Leaky Integrate & Fire (LIF) and a normalized Izhikevich model. Finally, **SNN network** defines a basic network composed of two linear and neuron layers intertwined. This network will be trained to perform classification tasks based on rate coding for the previously mentioned datasets.

## Table of contents
- [Software Requirements](#software-requirements)
- [Demos](#demos)
- [Gaussian Field Encoding](#gaussian-field-encoding)
    - [Mathematical Fundamentals](#mathematical-fundamentals)
    - [Methods](#methods)
- [Neuron models](#neuron-models)
    - [Leaky Integrate & Fire](#leaky-integrate-&-fire)
    - [Normalized Izhikevich](#normalized-izhikevich)

## Software Requirements
- 

## Demos


## Gaussian Field Encoding
This repository provides a small set of function to read UCI datasets or similar, process the different variables of the data elements and encode those values into sets of firing intervals for $n$ input neurons.

### Mathematical fundamentals

A Gaussian distribution $f(x)$ is calculated for every variable $x$ with the range $x \in [x_{min}, x_{max}]$ $$f(x)=\frac{1}{\sigma \sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$$

However, the normalization constant $\frac{1}{\sigma \sqrt{2\pi}}$ used to ensure that the sum of all possible hypotheses equals 1 is not used because, in order to compute the input intervals, it is easier to have excitations values from 0 to 1. Thus, a non-normalized Gaussian distribution $g(x)$ is used. $$g(x)=e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$$

A Gaussian field will be placed equidistantly for the $n$ input neurons throughout each variable's range. The position of a receptive field $i \in [0, n]$, is determined by its expected or mean value $$\mu = x_{min} + \frac{(2i-1)}{2}\frac{(x_{max}-x_{min})}{(n - 2)}$$

Each receptive field will have the same standard deviation. A field superposition vale $\beta$ of 1.5 has proven good results on the [literature](https://homepages.cwi.nl/~sbohte/publication/backprop.pdf). $$\sigma = \frac{1}{\beta}\frac{(x_{max}-x_{min})}{(n - 2)}$$

### Methods
Use cases of the following functions are included on the three available demo files.
* **readCSVData()** returns a list of lists that contains the file data. To parse a *.csv* file the element separator would be `\n` and the parameter separator `,`.

* **writeCSVData()** saves the processed data on a *.csv* file. **processedData** must be formatted as a list of lists.

* **processData()** returns a list of lists with the inputs for every neuron. It can be selected as a latency coded Gaussian or not. If the former is selected, the following variables are available.
    * **variablePositions** selects the columns of the file that contain the data to be processed. 
    * **resultPosition** indicates the position of the column which contains the classification of the element.
    * **resultEncoding** is a dictionary with the labels that correspond to each category. 
    * **fieldSuperposition** and **nInputNeurons** control the width of each receptive field and number of receptive fields. Please refer to the mathematical fundamentals for a more detailed explanation.
    * **nIntervals** contains the number of discrete input intervals that will encode the excitation of every receptive field for a given parameter. High excitations (close to 1) are encoded as input spikes on the first intervals, meanwhile spikes on the latter intervals correspond to low excitation values (near 0).

* **plotDataPoint()** Creates a figure in *.pdf* format that represents the  receptive field set, the excitation values and the input intervals for each neuron of one of the parameters of an element of the dataset. The latter two parameters of the function select the element and parameter to plot.

## Neuron models


### Leaky Integrate & Fire

The Leaky Integrate & Fire neuron model is one of the simplest yet powerful neuron models for SNNs. 




### Normalized Izhikevich

