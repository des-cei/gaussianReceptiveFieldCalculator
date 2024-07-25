# Custom libraries
from libraries.dataPreprocessing import *
from libraries.snnNetwork import *
from libraries.datasetLoader import *

# Pytorch imports
from torch.utils.data import DataLoader

# Matplotlib imports
matplotlib.use('Agg') # Correct problem with QThread in Ubuntu
import matplotlib.pyplot as plt 

# Parse .csv files
import pandas as pd

# Script options
import getopt
import sys

# File paths
sourceFile = './datasets/wdbc/wdbc.data'
saveFile = './processedData/wdbc.data'
networkFile = './networks/wdbcNetwork.pt'

# Data characteristics
variablePositions = range(2, 12, 1) # Mean value (only the first 10 parameters)
resultPosition = 1
resultEncoding = {
  "M": "0",
  "B": "1"
}

# Decode arguments
verbose_flag = False
process_data_flag = False
train_LIF_flag = False
test_Izhi_flag = False

try:
    opts, args = getopt.getopt(sys.argv[1:], "vpli")
except getopt.GetoptError as err:
    print(err)  # will print something like "option -a not recognized"
    sys.exit(2)

for o, a in opts:
  if o == "-v":
    verbose_flag = True
  elif o == "-p":
    process_data_flag = True
  elif o == "-l":
    train_LIF_flag = True
  elif o == "-i":
    test_Izhi_flag = True
  else:
      assert False, "unhandled option"

if(process_data_flag == False and train_LIF_flag == False and test_Izhi_flag == False):
   print("No actions were selected: -p (process data), -l (train LIF net), -i (test Izhi net)")

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


# =========================================================================================================
#                                                 Process data
# =========================================================================================================

if(process_data_flag):
  
  if(verbose_flag):
     print('\n\n')
     print('====================================================')
     print("                  PROCESSING DATA")
     print('====================================================')
  
  # Read data
  elementSeparator = '\n'
  parameterSeparator = ','

  if(verbose_flag):
     print("Reading data: ", sourceFile)
  rawData = readCSVData(sourceFile, elementSeparator, parameterSeparator)

  # Process data
  fieldSuperposition = 1.5 # Beta
  nInputNeurons = 4 # Input neurons per parameter
  nIntervals = 10

  # Example with no result interval calculation (directly write excitation)
  if(verbose_flag):
     print("Processing data")
  processedData = processData(data=rawData, 
                              variablePositions=variablePositions, 
                              nInputNeurons=nInputNeurons, 
                              fieldSuperposition=fieldSuperposition,
                              add_results=True,
                              resultEncoding=resultEncoding,
                              resultPosition=resultPosition,
                              gaussian=False,
                              normalize=True,
                              )

  # Save data
  if(verbose_flag):
     print('Storing processed data: ', saveFile)
  elementSeparator = '\n'
  parameterSeparator = ','
  writeCSVData(processedData, saveFile, elementSeparator, parameterSeparator, rearrange=True)

  # How to plot a random data point (if Gaussian encoding is used)

  #figurePath = 'figures\\wdbcPlot.pdf'
  #fieldSuperposition = 0.75 # Beta
  #nInputNeurons = 4 # Input neurons per parameter
  #nIntervals = 10
  #elementToPlot = 1
  #parameterToPlot = 2
  #
  #plotDataPoint(rawData, figurePath, fieldSuperposition, \
  #              nInputNeurons, nIntervals, elementToPlot, parameterToPlot)



# =========================================================================================================
#                                                Train and test LIF network
# =========================================================================================================

# Network Architecture
num_inputs = len(variablePositions)
num_hidden = 32
num_outputs = len(resultEncoding)
num_steps = 10

# Definitions to load the network to cuda
batch_size = 5
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

if(train_LIF_flag):
  if(verbose_flag):
     print('\n\n')
     print('====================================================')
     print("             TRAINING LIF NETWORK")
     print('====================================================')

  # Initialize datasets for training and testing
  if(verbose_flag):
     print('Reading data: ', saveFile)
  irisDataset_train =  CustomDataset(data_path=saveFile, train=True)
  irisDataset_test =  CustomDataset(data_path=saveFile, train=False)

  # Initialize data loaders (drop last avoids having a last batch with fewer elements)
  irisDataloader_train = DataLoader(irisDataset_train, batch_size=batch_size, drop_last=True)
  irisDataloader_test = DataLoader(irisDataset_test, batch_size=batch_size, drop_last=True)

  # Initialize network
  if(verbose_flag):
     print('Initializing LIF network')
  net = NetLIF(num_inputs, num_hidden, num_outputs, batch_size, num_steps).to(device) # Load the network onto CUDA if available

  # Training variables
  num_epochs = 100
  loss_hist = []
  test_loss_hist = []
  counter = 0
  loss = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(net.parameters(), lr=5e-3)
  total = 0
  correct = 0 

  if(verbose_flag):
     print('Starting training LIF network')
  # Outer training loop
  for epoch in range(num_epochs):
      iter_counter = 0
      train_batch = iter(irisDataloader_train)

      # Minibatch training loop
      for data, targets in train_batch:
          data = data.to(device)
          targets = targets.to(device)

          # forward pass
          net.train()
          spk_rec, mem_rec = net(data)

          # initialize the loss & sum over time
          loss_val = torch.zeros((1), dtype=dtype, device=device)
          for step in range(num_steps):
              loss_val += loss(spk_rec[step], targets)

          # Gradient calculation + weight update
          optimizer.zero_grad()
          loss_val.backward()
          optimizer.step()

          # Store loss history for future plotting
          loss_hist.append(loss_val.item())

          # Print one out of every 100 computed values
          if(verbose_flag):
            counter += 1
            if(counter % 100 == 0):  
              print(f"Train set loss: {loss_val.item()}")

  if(verbose_flag):
    print('Starting LIF network evaluation')
  with torch.no_grad():
    net.eval()
    for data, targets in irisDataloader_test:
      data = data.to(device)
      targets = targets.to(device)

      # forward pass
      test_spk, _ = net(data.view(data.size(0), -1))

      # calculate total accuracy
      _, predicted = test_spk.sum(dim=0).max(1)
      total += targets.size(0)
      correct += (predicted == targets).sum().item()

  print(f"Total correctly classified elements with LIF network: {correct}/{total}")
  print(f"Test accuracy: {100 * correct / total:.2f}%")

  # Save the model
  if(verbose_flag):
     print('Storing LIF network: ', networkFile)
  torch.save(net.state_dict(), networkFile)

  # Plot history loss of the model
  #plt.plot(loss_hist)
  #plt.show()

# =========================================================================================================
#                                                Test Izhikevich network
# =========================================================================================================

if(test_Izhi_flag):
  if(verbose_flag):
     print('\n\n')
     print('====================================================')
     print("            TESTING IZHIKEVICH NETWORK")
     print('====================================================')

  # Initialize datasets for training and testing
  if(verbose_flag):
     print('Reading data: ', saveFile)
  irisDataset_test =  CustomDataset(data_path=saveFile, train=False)

  # Initialize data loaders  (drop last avoids having a last batch with fewer elements)
  irisDataloader_test = DataLoader(irisDataset_test, batch_size=batch_size, drop_last=True)

  if(verbose_flag):
     print('Initializing Izhikevich network')

  # Define new network
  num_steps = 30
  net = NetIzhi(num_inputs, num_hidden, num_outputs, batch_size, num_steps).to(device) # Load the network onto CUDA if available
  
  if(verbose_flag):
     print('Loading network: ', networkFile)
  net.load_state_dict(torch.load(networkFile))

  total = 0
  correct = 0

  if(verbose_flag):
     print('Starting Izhikevich network evaluation')
  with torch.no_grad():
    net.eval()
    for data, targets in irisDataloader_test:
      data = data.to(device)
      targets = targets.to(device)

      # forward pass
      test_spk, _ = net(data.view(data.size(0), -1))

      # calculate total accuracy
      _, predicted = test_spk.sum(dim=0).max(1)
      total += targets.size(0)
      correct += (predicted == targets).sum().item()

  print(f"Total correctly classified elements with Izhi network: {correct}/{total}")
  print(f"Test accuracy: {100 * correct / total:.2f}%")