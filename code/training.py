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

        # Forward propagate LSTM (We just care about the output not the weights)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step giving you a value
        out = self.fc(out[:, -1, :])
        return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM Load')
    parser.add_argument('-i', dest='input', type=str, required=True,  help='Training data')
    parser.add_argument('-t', dest='test', type=str, required=False,  help='Test data')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_file = args.input.split(".")[0]

    input_size = 19
    hidden_size = 128
    num_layers = 2
    num_classes = 2
    batch_size = 100
    n_iters = 3000
    seq_length = 1

    train_dataset = np.loadtxt(args.input, delimiter=",")
    train_X = train_dataset[:,0:train_dataset.shape[1]-1]
    train_Y = train_dataset[:,-1]
    train_tensor_X = torch.from_numpy(train_X).float().to(device)
    train_tensor_Y = torch.from_numpy(train_Y).float().to(device)

    train_dataset = data.TensorDataset(train_tensor_X,train_tensor_Y)
    train_loader = data.DataLoader(train_dataset)

    #num_epochs = 15
    num_epochs = n_iters / (len(train_dataset) / batch_size)
    num_epochs = int(num_epochs)
    print(num_epochs)

    model = LSTMModel(input_size, hidden_size, num_layers, num_classes)

    criterion = nn.CrossEntropyLoss()

    learning_rate = 0.1

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Load images as torch tensor with gradient accumulation abilities
            images = images.view(-1, seq_length, input_size).requires_grad_()

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            # outputs.size() --> 100, 10
            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels.long())

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

    model.eval()

    if(args.test):
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

    torch.save(model.state_dict(), save_file+".pt")
