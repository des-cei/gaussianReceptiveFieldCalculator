import numpy as np
import math as math
import sys
import matplotlib
import random
matplotlib.use('Agg') # Correct problem with QThread in Ubuntu
import matplotlib.pyplot as plt 

# =========================================================================================================
#                                              "Private" functions
# =========================================================================================================

def findLimitValues(data, numericParameters):
  # Generate the default limit values (maximum and minimum possible values)
  limitValues = []
  for numericParameter in numericParameters:
    limitValues.append([sys.float_info.max, sys.float_info.min])

  # Find the minimum and maximum values for each parameter
  for j, element in enumerate(data):
    limitValuesIndex = 0
    for i in numericParameters:
      # Convert the string parameter to floating point value
      val = float(element[i])
      # Find the minimum and maximum values
      if val < limitValues[limitValuesIndex][0]:
        limitValues[limitValuesIndex][0] = val
      if val > limitValues[limitValuesIndex][1]:
        limitValues[limitValuesIndex][1] = val
      limitValuesIndex += 1
      
  return limitValues
  
def calculateExpectedValueGroup(minVal, maxVal, numberOfNeurons):
  i = np.arange(0, numberOfNeurons, 1)
  expectedValues = minVal + (2 * i - 1) / 2 * (maxVal - minVal) / ( numberOfNeurons - 2 )
  return expectedValues

  
def calculateStandardDeviation(minVal, maxVal, numberOfNeurons, beta):
  standardDeviation = 1 / beta * ( maxVal - minVal ) / numberOfNeurons - 2
  return standardDeviation
  
  
def calculateExcitation(x, expectedValueGroup, standardDeviation):
  excitation = np.exp(-1/2 * (float(x) - expectedValueGroup) ** 2 / standardDeviation ** 2)
  return excitation


def calculateInputIntervals(excitations, nIntervals):
  intervals = []
  for excitation in excitations:
    interval = round((1 - excitation) * nIntervals)
    #if(interval == n): 
      #interval = 9999
    intervals.append(interval)
  return intervals
  
# ===============================================================================================
#                                              "Public" functions
# ===============================================================================================

def readCSVData(fileName, elementSeparator, parameterSeparator):
  # Open file and separate the elements
  with open(fileName) as f:
    dataList = f.read().split(elementSeparator) 
  # Separate the parameters of each element
  data = []
  for dataLine in dataList:
    # Ignore empty lines
    if(dataLine == '\n' or dataLine == '' or dataLine == '\r\n'):
      continue
    dataLineSplit = dataLine.split(parameterSeparator)
    values = []
    for value in dataLineSplit:
      values.append(value)
    data.append(values)
  return data
  
  
def processData(
    data, 
    variablePositions, 
    nInputNeurons, 
    fieldSuperposition = 1.5, 
    calculate_intervals = False,
    nIntervals=0,
    resultPosition=0,
    resultEncoding={},
    add_results = False,
    gaussian = True,
    normalize = False):
  
  elementsInputs = []
  # Compute Gaussian Receptive Fields
  if(gaussian == True):
    # Calculate the limit values for each parameter (To optimally distribute the Gaussian Fields)
    limitValues = findLimitValues(data, variablePositions)

    # Calculate the gaussian field distribution for every parameter
    expectedValueGroups = []
    standardDeviations = [] 
    for i in range(len(variablePositions)):
      expectedValueGroup = \
        calculateExpectedValueGroup(limitValues[i][0], limitValues[i][1], nInputNeurons)
      standardDeviation = \
        calculateStandardDeviation(limitValues[i][0], limitValues[i][1], nInputNeurons, fieldSuperposition)
      
      expectedValueGroups.append(expectedValueGroup)
      standardDeviations.append(standardDeviation)


    # Calculate the excitation and input intervals for every parameter of every elementç      
    for element in data:
      elementInputs = np.zeros(shape=(4,4))
      for variablePos,variableIndex in enumerate(variablePositions):
        parameterExcitation = calculateExcitation(element[variableIndex], \
                  expectedValueGroups[variablePos], standardDeviations[variablePos])
        
        if(calculate_intervals == True):
          parameterInputs = calculateInputIntervals(parameterExcitation, nIntervals)
        else:
          parameterInputs = parameterExcitation

        elementInputs[variableIndex] = parameterInputs
      elementInputs = np.reshape(elementInputs,  16, order='F')
      elementsInputs.append(elementInputs.tolist())

  # Pass data without processing
  else:
    for element in data:
      element_row = []
      for variablePos, variableIndex in enumerate(variablePositions):
        element_row.append(float(element[variableIndex]))
      elementsInputs.append(element_row)

    if(normalize == True):
      limitValues = findLimitValues(data, variablePositions)
      for element_idx, element in enumerate(elementsInputs):
        for value_idx, value in enumerate(element):
          elementsInputs[element_idx][value_idx] = (value - limitValues[value_idx][0]) / (limitValues[value_idx][1] - limitValues[value_idx][0])

  # Add the result encoding
  if(add_results == True):
    for i in range(len(elementsInputs)):
      result = data[i][resultPosition]
      resultEncoded = resultEncoding[result]
      elementsInputs[i].append(resultEncoded)
  return elementsInputs
  


def writeCSVData(data, fileName, elementSeparator, parameterSeparator, rearrange=False):
  f = open(fileName, "w")

  if(rearrange):
    random.shuffle(data)
  
  for element in data:
    dataString = parameterSeparator.join(str(x) for x in element)
    if element != data[-1]:
      dataString += elementSeparator
    f.write(dataString)
  f.close()
  

def plotDataPoint(data, filePath, fieldSuperposition, nInputNeurons, \
                  nIntervals, elementToPlot, parameterToPlot):
  # Configure plot resolution
  res = 0.01
  
  # Calculate the expectedValueGroup (mean values of the distributions) 
  limitValues = findLimitValues(data, [parameterToPlot])
  expectedValueGroup = \
    calculateExpectedValueGroup(limitValues[0][0], limitValues[0][1], nInputNeurons) 

  # Calculate the standard deviations 
  standardDeviation = \
    calculateStandardDeviation(limitValues[0][0], limitValues[0][1], nInputNeurons, fieldSuperposition)
  
  # Set up subplots
  fig, ax = plt.subplots(2, figsize=(8,5))
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
  fig.suptitle('Gaussian Receptive Fields for variable ' + str(parameterToPlot))
  ax[0].set_xlabel('Parameter')
  ax[0].set_ylabel('Excitation')
  ax[1].set_xlabel('Input neurons')
  ax[1].set_ylabel('Firing intervals')
  ax[0].spines['top'].set_visible(False)
  ax[1].spines['top'].set_visible(False)
  ax[0].set_xticks(expectedValueGroup)
  ax[1].set_xticks(expectedValueGroup)
  ax[1].set_yticks(np.arange(0,11,1))
  
  ticklabels = []
  for i, expectedValue in enumerate(expectedValueGroup):
    ticklabels.append(str(i+1))
  ax[1].set_xticklabels(ticklabels)
  
  # Calculate and set up print limits
  diffExpectedValues = expectedValueGroup[1] - expectedValueGroup[0]
  pMin = expectedValueGroup[0] - diffExpectedValues
  pMax = expectedValueGroup[nInputNeurons-1] + diffExpectedValues
  ax[0].set_xlim([pMin, pMax])
  ax[1].set_xlim([pMin, pMax])
  
  # Print gaussian distributions
  for expectedValue in expectedValueGroup:
    xs = np.arange(pMin, pMax, res)
    ys = []
    for x in xs:
      y = calculateExcitation(x, expectedValue, standardDeviation)
      ys.append(y)
    ax[0].plot(xs, ys, linewidth=1) 
    
    # Print variable limits
  xMin = limitValues[0][0]
  xMax = limitValues[0][1]
  x = data[elementToPlot][parameterToPlot]
  
  ax[0].plot([xMin,xMin],[0,1], 'k-')
  ax[0].plot([xMax,xMax], [0,1], 'k-')
  
  # Print example value
  x = float(data[elementToPlot][parameterToPlot])
  ys = calculateExcitation(float(data[elementToPlot][parameterToPlot]), \
                           expectedValueGroup, standardDeviation)
  
  ax[0].plot([x, x], [0,1], 'r', linewidth=1)
  ax[0].plot([x]*nInputNeurons, ys , 'k.')
  
  for y in ys:
     ax[0].text(x+0.1, y-0.05, '('+str(round(y,2))+')')
  
  
  # Print gaussian excitations for the example value
  inputInterval = calculateInputIntervals(ys, nIntervals)
  
  for i, firingTime in enumerate(inputInterval):
      # Values
      ax[1].plot([expectedValueGroup[i],expectedValueGroup[i]] , [0,firingTime])
      ax[1].plot(expectedValueGroup[i] , firingTime, 'k_')
      # Dotted lines
      ax[1].plot([0,expectedValueGroup[i]] , [firingTime, firingTime], 'k:', linewidth=.3)
      
  # Save the figure
  plt.savefig(filePath)