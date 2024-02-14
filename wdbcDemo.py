from libraries.gaussianFieldCalculator import *
  

# ====================================================================================
#                             Breast Cancer Wisconsin Dataset
# ====================================================================================
# DATA SOURCE: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
#
# FROM: wine.names
#
# 5. Number of instances: 569 
# 
# 6. Number of attributes: 32 (ID, diagnosis, 30 real-valued input features)
# 
# 7. Attribute information
# 
# 1) ID number
# 2) Diagnosis (M = malignant, B = benign)
# 3-32)
# 
# Ten real-valued features are computed for each cell nucleus:
# 
# 	a) radius (mean of distances from center to points on the perimeter)
# 	b) texture (standard deviation of gray-scale values)
# 	c) perimeter
# 	d) area
# 	e) smoothness (local variation in radius lengths)
# 	f) compactness (perimeter^2 / area - 1.0)
# 	g) concavity (severity of concave portions of the contour)
# 	h) concave points (number of concave portions of the contour)
# 	i) symmetry 
# 	j) fractal dimension ("coastline approximation" - 1)
# 
# Several of the papers listed above contain detailed descriptions of
# how these features are computed. 
# 
# The mean, standard error, and "worst" or largest (mean of the three
# largest values) of these features were computed for each image,
# resulting in 30 features.  For instance, field 3 is Mean Radius, field
# 13 is Radius SE, field 23 is Worst Radius.
# 
# All feature values are recoded with four significant digits.
# 
# 8. Missing attribute values: none
# 
# 9. Class distribution: 357 benign, 212 malignant


# Read data
sourceFile = 'datasets\\wdbc\\wdbc.data'
elementSeparator = '\n'
parameterSeparator = ','

rawData = readCSVData(sourceFile, elementSeparator, parameterSeparator)

# Process data
variablePositions = range(2, 12, 1) # Mean value (only the first 10 parameters)

resultPosition = 1
resultEncoding = {
  "M": "13",
  "B": "20"
}
fieldSuperposition = 1.5 # Beta
nInputNeurons = 4 # Input neurons per parameter
nIntervals = 10

processedData = processData(rawData, variablePositions, resultPosition, resultEncoding, \
                            fieldSuperposition, nInputNeurons, nIntervals)

# Save data
saveFile = 'processedData\\wdbc.data'
elementSeparator = '\n'
parameterSeparator = ','
writeCSVData(processedData, saveFile, elementSeparator, parameterSeparator)

# How to plot a random data point:
figurePath = 'figures\\wdbcPlot.pdf'
fieldSuperposition = 0.75 # Beta
nInputNeurons = 4 # Input neurons per parameter
nIntervals = 10
elementToPlot = 1
parameterToPlot = 2

plotDataPoint(rawData, figurePath, fieldSuperposition, \
              nInputNeurons, nIntervals, elementToPlot, parameterToPlot)
