from gaussianFieldCalculator import *

# ====================================================================================
#                             Wine Recognition Dataset
# ====================================================================================
# SOURCE: https://archive.ics.uci.edu/dataset/109/wine
#
# FROM: wine.names
#
# 4. Relevant Information:
# 
#    -- These data are the results of a chemical analysis of
#       wines grown in the same region in Italy but derived from three
#       different cultivars.
#       The analysis determined the quantities of 13 constituents
#       found in each of the three types of wines. 
# 
#    -- I think that the initial data set had around 30 variables, but 
#       for some reason I only have the 13 dimensional version. 
#       I had a list of what the 30 or so variables were, but a.) 
#       I lost it, and b.), I would not know which 13 variables
#       are included in the set.
# 
#    -- The attributes are (dontated by Riccardo Leardi, 
# 	riclea@anchem.unige.it )
#  	1) Alcohol
#  	2) Malic acid
#  	3) Ash
# 	4) Alcalinity of ash  
#  	5) Magnesium
# 	6) Total phenols
#  	7) Flavanoids
#  	8) Nonflavanoid phenols
#  	9) Proanthocyanins
# 	10)Color intensity
#  	11)Hue
#  	12)OD280/OD315 of diluted wines
#  	13)Proline            
# 
# 5. Number of Instances
# 
#   class 1 59
# 	class 2 71
# 	class 3 48
# 
# 6. Number of Attributes 
# 	
# 	13
# 
# 7. For Each Attribute:
# 
# 	All attributes are continuous
# 	
# 	No statistics available, but suggest to standardise
# 	variables for certain uses (e.g. for us with classifiers
# 	which are NOT scale invariant)
# 
# 	NOTE: 1st attribute is class identifier (1-3)
# 
# 8. Missing Attribute Values:
# 
# 	None
# 
# 9. Class Distribution: number of instances per class
# 
#   class 1 59
# 	class 2 71
# 	class 3 48


# Read data
sourceFile = '..\\datasets\\wine\\wine.data'
elementSeparator = '\n'
parameterSeparator = ','
rawData = readCSVData(sourceFile, elementSeparator, parameterSeparator)

# Process data
variablePositions = range(1, 13, 1)
resultPosition = 0
resultEncoding = {
  "1": "10",
  "2": "13",
  "3": "16"
}
fieldSuperposition = 1.5 # Beta
nInputNeurons = 4 # Input neurons per parameter
nIntervals = 10
processedData = processData(rawData, variablePositions, resultPosition, resultEncoding, \
                            fieldSuperposition, nInputNeurons, nIntervals)

# Save data
saveFile = '..\\processedData\\wine.data'
elementSeparator = '\n'
parameterSeparator = ','
writeCSVData(processedData, saveFile, elementSeparator, parameterSeparator)

# How to plot a random data point:
figurePath = '...\\figures\\winePlot.pdf'
fieldSuperposition = 1.5 # Beta
nInputNeurons = 4 # Input neurons per parameter
nIntervals = 10
elementToPlot = 1
parameterToPlot = 1
plotDataPoint(rawData, figurePath, fieldSuperposition, \
              nInputNeurons, nIntervals, elementToPlot, parameterToPlot)
