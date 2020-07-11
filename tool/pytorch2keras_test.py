import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import tensorflow as tf
from pytorch2keras import pytorch_to_keras
from segmentation.model.deepLabV3 import DeepLabV3
class TestConv2d(nn.Module):
    """
    Module for Conv2d testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3):
        super(TestConv2d, self).__init__()
        self.conv2d = nn.Conv2d(inp, out, stride=1, kernel_size=kernel_size, bias=True)

    def forward(self, x):
        x = self.conv2d(x)
        return x

#model = TestConv2d()
input = torch.rand(1, 3, 473, 473).cuda()
model = DeepLabV3(layers=50, dropout=0.1, classes=21, zoom_factor=8, pretrained=True).cuda()
model.eval()
print(model)
output = model(input)

#model = DeepLabV3(layers=101, classes=21, zoom_factor=8, pretrained=False)

#input_np = np.random.uniform(0, 1, (1, 3, 313, 313))
#input_var = Variable(torch.FloatTensor(input_np))

#k_model = pytorch_to_keras(model, input_var, [(3, 313, 313,)], verbose=True)

