# Gaussian Field Calculator
This repository provides a small set of function to read UCI datasets or similar, process the different variables of the data elements and encode those values into sets of firing intervals for $n$ input neurons.\
## Mathematical fundamentals
A Gaussian distribution $f(x)$ is calculated for every variable $x$ with the range $x \in [x_{min}, x_{max}]$
$$ f(x) = \frac{1}{\sigma \sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$$
However, the normalization constant $\frac{1}{\sigma \sqrt{2\pi}}$ used to ensure that the sum of all possible hypotheses equals 1 is not used because, in order to compute the input intervals, it is easier to have excitations values from 0 to 1. Thus, a non-normalized Gaussian distribution $g(x)$ is used.
$$ g(x) = e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$$
$n$ Gaussian fields will be placed throughout the range of each variable, each one corresponding to an input neuron for said variable. The position of a receptive field $i$, for $i<n$, is determined by its expected or mean value $$\mu = x_{min} + \frac{(2i-1)}{2}\frac{(x_{max}-x_{min})}{(n - 2)}$$
Each receptive field will have the same standard deviation. A field superposition vale $\beta$ of 1.5 has proven good results on the [literature](https://homepages.cwi.nl/~sbohte/publication/backprop.pdf).
 $$\sigma = \frac{1}{\beta}\frac{(x_{max}-x_{min})}{(n - 2)}$$

## Installation
Windows installation steps on a [virtual environment](https://docs.python.org/3/library/venv.html).
1. Install [Python](https://www.python.org/downloads/)
2. **Download** the repository *.zip* on an empty folder.
3. Navigate to the folder.
4. Create the virtual environment by typing `python -m venv name_of_the_venv`.
5. Activate the virtual environment with `name_of_the_venv\Scripts\activate`. 
The following error message is caused by [problems with the Script Execution Policy](https://stackoverflow.com/questions/18713086/virtualenv-wont-activate-on-windows). Type `Set-ExecutionPolicy Unrestricted -Scope Process` on **Powershell**, which will allow the execution of scripts on the current session.

    > "cannot be loaded because the execution of scripts is disabled on this system"

6. Once the virtual environment is activated, in order to automatically install the associated libraries on the virtual environment, **matplotlib** and **numpy**, use the following command `pip install -r requirements.txt`.
7. Afterwards, the functions of the library can be tested with the three available demos, which will process commonly used [UCI datasets](https://archive.ics.uci.edu/datasets). Stored on the [datasets](datasets) folder, Iris, Wisconsin Breast Cancer and Wine datasets are processed on [irisDemo.py](irisDemo.py), [wdbcDemo.py](wdbcDemo.py) and [wineDemo.py](wineDemo.py).\

## Functions
Use cases of the following functions are included on the three available demo files.
* **readCSVData(fileName, elementSeparator, parameterSeparator)** returns a list of lists that contains the file data. E.g., in order to parse a *.csv* file the element separator would be `\n` and the parameter separator `,`.

* **writeCSVData(processedData, saveFile, elementSeparator, parameterSeparator)** saves the processed data on a *.csv* file. **processedData** must be formatted as a list of lists.

* **processData(data, variablePositions, resultPosition, resultEncoding, fieldSuperposition, nInputNeurons, nIntervals)** returns a list of lists that contains the input intervals for every neuron. 
    * **variablePositions** selects the columns of the file that contain the data to be processed. 
    * On a similar fashion, the **resultPosition** indicates the position of the column which contains the classification of the element.
    * **resultEncoding** is a dictionary which will include the output interval that will be used to classify each element during training and testing. The following example states that flowers from the Iris dataset that the Spiked Neural Network will classify as part of the setosa family will be characterized by a spike on the eleventh time interval.
        ```
        resultEncoding = {
        "Iris-setosa": "11",
        "Iris-versicolor": "16",
        "Iris-virginica": "21"
        }
        ```
    * **fieldSuperposition** and **nInputNeurons** control the width of each receptive field and number of receptive fields. Please refer to the mathematical fundamentals for a better understanding.

    * **nIntervals** contains the number of discrete input intervals that will encode the excitation of every receptive field for a given parameter. High excitations are encoded as input spikes on the first intervals, meanwhile spikes on the latter intervals correspond to near 0 excitation values.


* **plotDataPoint(rawData, filePath, fieldSuperposition, nInputNeurons, nIntervals, elementToPlot, parameterToPlot)** Creates a figure in *.pdf* format that represents the Gaussian Receptive Fields, the excitation values and the input intervals for each neuron of one of the parameters of an element of the dataset. The latter two parameters of the function select the element and parameter to plot. \
This function aims to improve the understanding of the receptive fields. For this reason, it works independently by processing and plotting the raw data from the dataset.