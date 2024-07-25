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
sourceFile = './datasets/wine/wine.data'
saveFile = './processedData/wine.data'
networkFile = './networks/wineNetwork.pt'

# Data characteristics
variablePositions = range(1, 13, 1)
resultPosition = 0
resultEncoding = {
  "1": "0",
  "2": "1",
  "3": "2"
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

  # How to plot a random data point (if it used Gaussian Encoding)
  #figurePath = 'figures/winePlot.pdf'
  #fieldSuperposition = 1.5 # Beta
  #nInputNeurons = 4 # Input neurons per parameter
  #nIntervals = 10
  #elementToPlot = 1
  #parameterToPlot = 1
  #plotDataPoint(rawData, figurePath, fieldSuperposition, \
  #              nInputNeurons, nIntervals, elementToPlot, parameterToPlot)

# =========================================================================================================
#                                                Train and test LIF network
# =========================================================================================================

# Network Architecture
num_inputs = len(variablePositions)
num_hidden = 32
num_outputs = len(resultEncoding)
num_steps = 20

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
  num_epochs = 25
  loss_hist = []
  test_loss_hist = []
  counter = 0
  loss = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
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
  num_steps = 45
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