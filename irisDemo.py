from gaussianFieldCalculator import *

# ====================================================================================
#                                     Iris Dataset
# ====================================================================================
# SOURCE: https://archive.ics.uci.edu/dataset/53/iris 
#
# FROM: iris.names
#
# 5. Number of Instances: 150 (50 in each of three classes)
# 
# 6. Number of Attributes: 4 numeric, predictive attributes and the class
# 
# 7. Attribute Information:
#    1. sepal length in cm
#    2. sepal width in cm
#    3. petal length in cm
#    4. petal width in cm
#    5. class: 
#       -- Iris Setosa
#       -- Iris Versicolour
#       -- Iris Virginica
# 
# 8. Missing Attribute Values: None
# 
# Summary Statistics:
# 	               Min  Max   Mean    SD   Class Correlation
#    sepal length: 4.3  7.9   5.84  0.83    0.7826   
#     sepal width: 2.0  4.4   3.05  0.43   -0.4194
#    petal length: 1.0  6.9   3.76  1.76    0.9490  (high!)
#     petal width: 0.1  2.5   1.20  0.76    0.9565  (high!)
# 
# 9. Class Distribution: 33.3% for each of 3 classes.



# Read data
sourceFile = 'datasets\\iris\\iris.data'
elementSeparator = '\n'
parameterSeparator = ','

rawData = readCSVData(sourceFile, elementSeparator, parameterSeparator)

# Process data
variablePositions = [0, 1, 2, 3]
resultPosition = 4
resultEncoding = {
  "Iris-setosa": "11",
  "Iris-versicolor": "16",
  "Iris-virginica": "21"
}
fieldSuperposition = 1.5 # Beta
nInputNeurons = 4 # Input neurons per parameter
nIntervals = 10

processedData = processData(rawData, variablePositions, resultPosition, resultEncoding, \
                            fieldSuperposition, nInputNeurons, nIntervals)

# Save data
saveFile = 'processedData\\iris.data'
elementSeparator = '\n'
parameterSeparator = ','
writeCSVData(processedData, saveFile, elementSeparator, parameterSeparator)

# How to plot a random data point:
figurePath = 'figures\\irisPlot.pdf'
fieldSuperposition = 1.5 # Beta
nInputNeurons = 4 # Input neurons per parameter
nIntervals = 10
elementToPlot = 0
parameterToPlot = 0

plotDataPoint(rawData, figurePath, fieldSuperposition, nInputNeurons, \
              nIntervals, elementToPlot, parameterToPlot)