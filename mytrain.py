import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
#from apex import amp

from data_loader import create_dataloaders
from model import get_trainable_params, create_model, print_model_params
from train import train
from utils import parse_and_override_params

#


# Fix random seed
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


print('Creating datasets')
# Get dataloaders
train_dl, val_both_dl, display_dl_iter = create_dataloaders()

