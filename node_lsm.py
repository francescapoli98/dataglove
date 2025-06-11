import torch
from classification.lsm import LiquidStateMachine  
from classification.events_classification import test

checkpoint = torch.load("models/lsm_checkpoint.pt", map_location='cpu')

# Rebuild model with same config
model = LiquidStateMachine(**checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

scaler = preprocessing.StandardScaler()
activations = scaler.fit_transform(data) 

