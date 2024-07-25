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
sourceFile = './datasets/iris/iris.data'
saveFile = './processedData/iris.data'
networkFile = './networks/irisNetwork.pt'

# Data characteristics
variablePositions = [0, 1, 2, 3]
resultPosition = 4
resultEncoding = {
  "Iris-setosa": "0",
  "Iris-versicolor": "1",
  "Iris-virginica": "2"
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

# =========================================================================================================
#                                                 Iris Dataset information
# =========================================================================================================

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

  #figurePath = './figures/irisPlot.pdf'
  #fieldSuperposition = 1.5 # Beta
  #nInputNeurons = 4 # Input neurons per parameter
  #nIntervals = 10
  #elementToPlot = 0
  #parameterToPlot = 0
  #
  #plotDataPoint(rawData, figurePath, fieldSuperposition, nInputNeurons, \
  #              nIntervals, elementToPlot, parameterToPlot)


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

  # Initialize data loaders  (drop last avoids having a last batch with fewer elements)
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