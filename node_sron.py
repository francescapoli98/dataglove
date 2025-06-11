import torch
from classification.s_ron import SpikingRON  # Adjust import path
from classification.events_classification import test

checkpoint = torch.load("models/sron_checkpoint.pt", map_location='cpu')

# Rebuild model with same config
model = SpikingRON(**checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
