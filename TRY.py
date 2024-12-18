import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import math
import matplotlib.pyplot as plt
from einops import rearrange
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class AMA(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(AMA, self).__init__()
        # kernel_size = int(abs(((np.log(channel) / np.log(2)) + b) / gamma))
        # kernel_size = int(abs((1.5*math.pow(channel, 0.35) +6) / 4))
        kernel_size = int(abs((2 * math.pow(channel, 0.35) + 1) / 4))
        print('channel =', channel)
        print('kernel_size =', kernel_size)
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        y = self.conv(x)
        return x * y.expand_as(x)

model = AMA(channel=512)





x = np.arange(0, 1024, 0.01)
y1 = (2 * np.power(x, 0.35) + 1) / 4
y2 = ((np.log(x) / np.log(2)) + 1) / 2

# plt.plot(x, y2, '--', label=" k = ((np.log(c) / np.log(2)) + n) / m ")
plt.plot(x, y1, '--', color="DarkOrange", label=" k = (a * np.power(c, 0.35) + b) / g ")

plt.xlabel("input_channel")
plt.ylabel("kernel_size")
plt.xlim(0, 10)
plt.ylim(0, 2.5)
plt.legend()
plt.show()
