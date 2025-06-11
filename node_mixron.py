import torch
from classification.mixed_ron import MixedRON  
from classification.events_classification import test

checkpoint = torch.load("models/mixedron_checkpoint.pt", map_location='cpu')

# Rebuild model with same config
model = MixedRON(**checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
