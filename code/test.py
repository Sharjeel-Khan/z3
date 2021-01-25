import os
from multiprocessing import Process
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils import data


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()

        # Number of hidden 
        self.hidden_size = hidden_size

        #Number of Layers
        self.num_layers = num_layers

        # LSTM based on input_size, hidden_size and num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Taking the number of output values and producing an output based on num_classes
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multiprocessing Z3')
    parser.add_argument('-test', dest='test', type=str, required=True,  help='Communication File')
    parser.add_argument('-model', dest='model', type=str, required=True,  help='LSTM Model file')

    args = parser.parse_args()
    N = args.num_procs

    print("LSTM Model: ")
    print("Communication File: "+args.test)
    print("Model: "+args.model)

    input_size = 19
    hidden_size = 128
    num_layers = 2
    num_classes = 2
    batch_size = 100
    n_iters = 3000
    seq_length = 1

    model = LSTMModel()

    model.load_state_dict(torch.load(args.model))

    test_dataset = np.loadtxt(args.test, delimiter=",")

    test_X = test_dataset[:,0:test_dataset.shape[1]-1]
    test_Y = test_dataset[:,-1]
    test_tensor_X = torch.Tensor(test_X)
    test_tensor_Y = torch.Tensor(test_Y)

    test_dataset = data.TensorDataset(test_tensor_X,test_tensor_Y)
    test_loader = data.DataLoader(test_dataset)

    with torch.no_grad():
        correct = 0
        total = 0
        # Iterate through test dataset
        for images, labels in test_loader:
            # Resize image
            images = images.view(-1, seq_length, input_size)

            # Forward pass only to get logits/output
            outputs = model(images)

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

            # Total number of labels
            total += labels.size(0)

            # Total correct predictions
            correct += (predicted == labels).sum()

        accuracy = 100 * correct / float(total)

        print('Test Accuracy of the model: {} %'.format(accuracy))






