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

## Getting Started
- Download the repository with `git clone `.
- Install Python3 with `sudo apt-get update` and `sudo apt-get install python3`.
- Install pip with `sudo python3 install pip`. 
- Use `pip install -r requirements.txt`, to download the associated libraries. *Numpy*, *matplotlib*, *pandas* and *torch* will be used.

## Demos
Three demos are available. They aim to illustrate the execution of the same operations with each dataset, which can be selected with arguments.
- `-v` will print the majority of the available information on every stage.
- `-p` will process the data according to the selected arguments. In this case, data is just normalized and scrambled for training before being stored in the *processedData* folder.
- `-l` will train and test an SNN with the LIF model, using the available data. The trained network  weights and biases are stored in the *networks* folder.
- `-i` will evaluate the network using the Izhikevich model. Since both models are normalized, it is expected to obtain similar performance with the LIF and Izhikevich models.

## Gaussian Field Encoding
This repository provides a small set of function to read UCI datasets or similar, process the different variables of the data elements and encode those values into sets of firing intervals for $n$ input neurons.

### Mathematical fundamentals

A Gaussian distribution $f(x)$ is calculated for every variable $x$ with the range $x \in [x_{min}, x_{max}]$ 

$$f(x)=\frac{1}{\sigma \sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$$

However, the normalization constant $\frac{1}{\sigma \sqrt{2\pi}}$ used to ensure that the sum of all possible hypotheses equals 1 is not used because, in order to compute the input intervals, it is easier to have excitations values from 0 to 1. Thus, a non-normalized Gaussian distribution $g(x)$ is used. 

$$g(x)=e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$$

A Gaussian field will be placed equidistantly for the $n$ input neurons throughout each variable's range. The position of a receptive field $i \in [0, n]$, is determined by its expected or mean value 

$$\mu = x_{min} + \frac{(2i-1)}{2}\frac{(x_{max}-x_{min})}{(n - 2)}$$

Each receptive field will have the same standard deviation. A field superposition value $\beta$ of 1.5 has proven good results on the [literature](https://homepages.cwi.nl/~sbohte/publication/backprop.pdf). 

$$\sigma = \frac{1}{\beta}\frac{(x_{max}-x_{min})}{(n - 2)}$$

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

The Leaky Integrate & Fire neuron model is one of the simplest yet powerful neuron models for SNNs. According to this model, neuron membranes behave as capacitors in parallel with  resistors. This voltage sharply increases when a spike is received, and it experiences an exponential with a $\beta$ decay. Usually, only if multiple current spikes are received over a short period of time, the membrane voltage will reach the established threshold. This is the behavior that characterizes integrator neurons with regular spiking.


### Normalized Izhikevich

The Izhikevich neuron model aims to provide an accurate neuron model at a relatively low computational cost. Instead of having a single hyperparameter, such as the LIF model, it includes four. By adjusting this parameters, it is possible to produce a wide variety of behaviors besides the regular spiking that characterizes LIF neurons. Izhikevich models are capable of chattering, bursting and they can even behave as resonators

$$ C\dot v = k(v - v_r)(v - v_t) - u + I $$

$$ \dot u = a [b(v - v_r) - u] $$

$$ \text{if} \\; v \le v_{\text{p}} \\; \\; \\; \\; v \leftarrow c  \\; \\; \\; \\;  u \leftarrow u + d$$

For example to produce regular spiking patterns the following parameters can be used:

- $C = 100 \\;\\;\rightarrow$              Membrane capacitance (pF)
- $k = 0.7\\;\\;\rightarrow$              Input resistance (pA/mV)
- $v_r = -60\\;\\;\rightarrow$              Resting membrane potential (mV)
- $v_t = -40\\;\\;\rightarrow$              Instantaneous threshold potential (mV)
- $a = 0.03 \\;\\;\rightarrow$              Time scale of the recovery variable (1/ms)
- $b = -2 \\;\\;\rightarrow$              Sensitivity of the recovery variable (pA/mV)
- $c = -50\\;\\;\rightarrow$              Potential reset value (mV)
- $d = 100 \\;\\;\rightarrow$              Spike triggered adaptation (pA)
- $v_p = 35\\;\\;\rightarrow$      Spike cutoff (mV)

However, when introducing this neuron model as an SNN layer, adjusting the weights and biases of the layers that this neurons are connected to can be really challenging. For this reason, researchers have proposed alternatives to [normalize this model](https://www.mdpi.com/2079-9292/13/5/909).

By operating with the parenthesis, the previous equation can be expressed as:
$$ \dot v = a_1v^2 + a_2v + a_3u + a_4I + a_5$$

$$ \dot u = b_1v+b_2u+b_3  $$

$$ \text{if} \\; v \le v_{\text{p}} \\;\\;\\;\\; v \leftarrow c \\;\\;\\;\\; u \leftarrow u + d$$

$$a_1 = \frac{k}{C}\\;\\;\\; a_2 = -\frac{k}{C}(v_r+v_t)\\;\\;\\; a_3 = -\frac{1}{C}\\;\\;\\; a_4 = \frac{1}{C}\\;\\;\\; a_5 = \frac{k}{C}(v_r+v_t)$$

$$b_1 = ab\\;\\;\\; b_2 = -a\\;\\;\\; b_3 = -abv_r$$

To normalize this equation, it can be established that:
$$L_v = \text{max}_v - \text{min}_v$$

$$L_u = \text{max}_u - \text{min}_u$$

The same process that was previously shown results in:

$$ \dot v = a_1v^2 + a_2v + a_3u + a_4I + a_5$$

$$ \dot u = b_1v+b_2u+b_3  $$

$$ \text{if} \\; v \le c_1 \\;\\;\\;\\; v \leftarrow c_2 \\;\\;\\;\\; u \leftarrow u + c_3$$

$$a_1 = L_v\frac{k}{C}\\;\\;\\; a_2 = (2\text{min}_v - v_r - v_t)\frac{k}{C}\\;\\;\\; a_3 = \frac{L_u}{L_vC}\\;\\;\\; a_4 = \frac{1}{L_vC}\\;\\;\\; a_5= \frac{k}{C}(\text{min}^2_v - v_r\text{min}_v- v_t\text{min}_v + v_rv_t)-\frac{\text{min}_u}{C}$$

$$b_1 = ab\frac{L_v}{L_u}\\;\\;\\; b_2 = -a\\;\\;\\; b_3 = (ab\text{min}_v-abv_r - a\text{min}_u) \frac{1}{L_u}$$

$$c_1 = \frac{v_p - \text{min}_u}{C}\\;\\;\\; c_2 = \frac{c - min_v}{L_v}\\;\\;\\; c_3 = \frac{d}{L_u}$$

However, since the Input $I$ is dependant on the weights and biases of the network, it is not needed to scale its value with $a_4$ or to add an offset $a_5$.


To produce the same spiking behavior as the default Izhikevich neuron, the parameters would be:
- $a_1 = 1.0$
- $a_2 = -0.21$
- $a_3 = -0.019$
- $b_1 = -1/32$
- $b_2 = -1/32$
- $b_3 = 0$
- $c_1 = 1$
- $c_2 = 0.105$
- $c_3 = 0.412$