from build_hdf5 import parse_filename
from models import TCN

import torch.nn as nn
import torch

file = 'emg_p0_r0_two.npz'

model = TCN.load_from_checkpoint(checkpoint_path = '/home/pantelis/Documents/SeNic/models/TCN/best-model-0-v1.ckpt', in_channels=8, num_classes=7, loss=nn.CrossEntropyLoss())

input = torch.randn((256, 8, 60))

output = model(input)

print(output.shape)