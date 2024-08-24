# Pytorch imports
import torch
import torch.nn as nn

from libraries.snnNeurons import *

class NetIzhi(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, batch_size, num_steps):
        super(NetIzhi, self).__init__()

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.batch_size = batch_size
        self.num_steps = num_steps

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = Izhi()
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = Izhi()

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = torch.zeros((self.batch_size, self.num_hidden),  dtype=torch.float)
        mem2 = torch.zeros((self.batch_size, self.num_outputs), dtype=torch.float)
        rec1 = torch.zeros((self.batch_size, self.num_hidden),  dtype=torch.float)
        rec2 = torch.zeros((self.batch_size, self.num_outputs), dtype=torch.float)

        # Record layers
        x_rec = []
        spk1_rec = []
        spk2_rec = []
        mem1_rec = []
        mem2_rec = []
        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1, rec1 = self.lif1(cur1, mem1, rec1)
            cur2 = self.fc2(spk1)
            spk2, mem2, rec2 = self.lif2(cur2, mem2, rec2)
            x_rec.append(torch.Tensor(x))
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            mem1_rec.append(mem1)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
    
# Define Network
class NetLIF(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, batch_size, num_steps):
        super(NetLIF, self).__init__()

        # Initialization parameters
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.batch_size = batch_size
        self.num_steps = num_steps

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = LeakySurrogate()
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = LeakySurrogate()

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = torch.zeros(self.batch_size, self.num_hidden)
        mem2 = torch.zeros(self.batch_size, self.num_outputs)

        # Record the final layer
        x_rec = []
        spk2_rec = []
        mem1_rec = []
        mem2_rec = []
        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            x_rec.append(torch.Tensor(x))
            mem1_rec.append(mem1)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
    
class NetIzhiExtraLayer(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, batch_size, num_steps):
        super(NetIzhiExtraLayer, self).__init__()

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.batch_size = batch_size
        self.num_steps = num_steps

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = Izhi()
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif2 = Izhi()
        self.fc3 = nn.Linear(num_hidden, num_outputs)
        self.lif3 = Izhi()

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = torch.zeros((self.batch_size, self.num_hidden),  dtype=torch.float)
        mem2 = torch.zeros((self.batch_size, self.num_hidden), dtype=torch.float)
        mem3 = torch.zeros((self.batch_size, self.num_outputs), dtype=torch.float)
        rec1 = torch.zeros((self.batch_size, self.num_hidden),  dtype=torch.float)
        rec2 = torch.zeros((self.batch_size, self.num_hidden), dtype=torch.float)
        rec3 = torch.zeros((self.batch_size, self.num_outputs), dtype=torch.float)

        # Record layers
        x_rec = []
        spk1_rec = []
        spk2_rec = []
        spk3_rec = []
        mem1_rec = []
        mem2_rec = []
        mem3_rec = []
        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1, rec1 = self.lif1(cur1, mem1, rec1)
            cur2 = self.fc2(spk1)
            spk2, mem2, rec2 = self.lif2(cur2, mem2, rec2)
            cur3 = self.fc3(spk2)
            spk3, mem3, rec3 = self.lif2(cur3, mem3, rec3)
            x_rec.append(torch.Tensor(x))
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            spk3_rec.append(spk3)
            mem1_rec.append(mem1)
            mem2_rec.append(mem2)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)
    
# Define Network
class NetLIFExtraLayer(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, batch_size, num_steps):
        super(NetLIFExtraLayer, self).__init__()

        # Initialization parameters
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.batch_size = batch_size
        self.num_steps = num_steps

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = LeakySurrogate()
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif2 = LeakySurrogate()
        self.fc3 = nn.Linear(num_hidden, num_outputs)
        self.lif3 = LeakySurrogate()

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = torch.zeros(self.batch_size, self.num_hidden)
        mem2 = torch.zeros(self.batch_size, self.num_hidden)
        mem3 = torch.zeros(self.batch_size, self.num_outputs)
        # Record the final layer
        spk3_rec = []
        mem1_rec = []
        mem2_rec = []
        mem3_rec = []
        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif2(cur3, mem3)
            spk3_rec.append(spk3)
            mem1_rec.append(mem1)
            mem2_rec.append(mem2)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)