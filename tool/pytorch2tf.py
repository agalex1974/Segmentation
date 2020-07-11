import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import onnx
#from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np
from segmentation.model.deepLabV3 import DeepLabV3

class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpleModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fcs = []  # List of fully connected layers
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_features=in_size, out_features=next_size)
            in_size = next_size
            self.__setattr__('fc{}'.format(i), fc)  # set name for each fullly connected layer
            self.fcs.append(fc)

        self.last_fc = nn.Linear(in_features=in_size, out_features=output_size)

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            x = nn.ReLU()(x)
        out = self.last_fc(x)
        return nn.Sigmoid()(out)

dummy_input = torch.rand(1, 3, 313, 313)
model_pytorch = DeepLabV3(layers=101, dropout=0.1, classes=21, zoom_factor=8, pretrained=True)
model_pytorch.eval()
dummy_output = model_pytorch(dummy_input)

#model_pytorch = DeepLabV3(layers=101, classes=21, zoom_factor=8, pretrained=False)

#model_pytorch = SimpleModel(input_size=100, hidden_sizes=[500], output_size=10)
#model_pytorch.load_state_dict(torch.load('./models/model_simple.pt'))

#X_test = np.random.randn(1, 3, 313, 313).astype(np.float32)

#dummy_input = torch.from_numpy(X_test[0].reshape(1, 3, 313, 313)).float()
#dummy_output = model_pytorch(dummy_input)
#print(dummy_output)

# Export to ONNX format
torch.onnx.export(model_pytorch, dummy_input, 'model_simple.onnx', input_names=['test_input'], output_names=['test_output'])

from onnx2keras import onnx_to_keras

# Load ONNX model
onnx_model = onnx.load('model_simple.onnx')

# Call the converter (input - is the main model input name, can be different for your model)
k_model = onnx_to_keras(onnx_model, ['test_input'])