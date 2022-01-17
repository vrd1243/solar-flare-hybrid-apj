import torch
from torchvision import *

class Vgg16Conv3D(torch.nn.Module):

    #Our batch shape for input x is (batch_size, input_channels, length, width)
    
    def __init__(self, input_channels):

        super(Vgg16Conv3D, self).__init__()
        self.conv3d = torch.nn.Conv3d(1,1, kernel_size=(3,3,3), stride=(1,1,1), padding=(0,1,1))
        self.vgg16 = models.vgg16(pretrained=False)
        self.vgg16.features[0] = torch.nn.Conv2d(input_channels-2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # Modify the input layer (image). Mapping from input_channels to 64.
        num_features = self.vgg16.classifier[6].in_features # Number of in_features of the last layer
        features = list(self.vgg16.classifier.children())[:-1] # Remove last layer
        features.extend([torch.nn.Linear(num_features, 2)]) # Add our layer with 2 outputs
        self.vgg16.classifier = torch.nn.Sequential(*features) # Replace the model classifier

    def forward(self, x):
        
        # Input: (batch_size, input_channels, length, width)

        # Transform: (batch_size, 1, input_channels, length, width)
        x = torch.unsqueeze(x, 1)
        
        # Conv-3D layer. (batch_size, 1, input_channels - 2, length, width)
        x = self.conv3d(x)

        # Transform: (batch_size, input_channels - 2, length, width)
        x = torch.squeeze(x, 1)

        # Reshape the tensor to shape: batch_size x (n_channels - 2) x length x width
        vgg16_in = x.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        # Run through vgg16
        vgg16_out = self.vgg16(vgg16_in)
        
        # Output: batch_size x 2
        return vgg16_out

class Vgg16(torch.nn.Module):
        
    #Our batch shape for input x is (batch_size, input_channels, length, width)
    
    def __init__(self, input_channels):

        super(Vgg16, self).__init__()
        self.vgg16 = models.vgg16(pretrained=False)
        self.vgg16.features[0] = torch.nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # Modify the input layer (image). Mapping from input_channels to 64.
        num_features = self.vgg16.classifier[6].in_features # Number of in_features of the last layer
        features = list(self.vgg16.classifier.children())[:-1] # Remove last layer
        features.extend([torch.nn.Linear(num_features, 2)]) # Add our layer with 2 outputs
        self.vgg16.classifier = torch.nn.Sequential(*features) # Replace the model classifier
        
    def forward(self, x):
        
        # Reshape the tensor to shape: batch_size x n_channels x length x width
        vgg16_in = x.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        # Run through vgg16
        vgg16_out = self.vgg16(vgg16_in)
        
        # Output: batch_size x 2
        return vgg16_out


class Vgg16LSTM(torch.nn.Module):

    #Our batch shape for input x is (batch_size, sequence_length, length, width)

    def __init__(self):
        super(Vgg16LSTM, self).__init__()
        self.vgg16 = models.vgg16(pretrained=False)
        self.vgg16.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # Modify the input layer (image). Mapping from input_channels to 64.
        num_features = self.vgg16.classifier[6].in_features
        features = list(self.vgg16.classifier.children())[:-1] # Remove last layer
        features.extend([torch.nn.Linear(num_features, 24)]) # Add our layer with 24 outputs
        self.vgg16.classifier = torch.nn.Sequential(*features) # Replace the model classifier
        self.lstm = torch.nn.LSTM(24, 16, batch_first=True, num_layers=1)
        self.fc3 = torch.nn.Linear(16, 2)

    def forward(self, x):
        # We merge the batch size and the channel dimensions.

        # Input: (batch_size, sequence_length, length, width)
        cnn_in = x.view(x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3])

        # Input: (batch_size * sequence_length, 1, length, width)
        cnn_out = self.vgg16(cnn_in)

        # Input: (batch_size * sequence_length, 24)
        r_in = cnn_out.view(x.shape[0], x.shape[1], -1)
        
        # Input: (batch_size, sequence_length, 24)
        r_out, _ = self.lstm(r_in)
        r_out = r_out[:,-1]

        # Input: (batch_size, 16)
        r_out = self.fc3(r_out)

        # Output: (batch_size, 2)
        return r_out