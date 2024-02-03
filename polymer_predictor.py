import numpy as np
from psmiles.psmiles import PolymerSmiles
from rdkit import Chem
from importlib import util
import torch
from sklearn.preprocessing import MinMaxScaler 
import torch.nn as nn
import pandas as pd


class MTmodel(nn.Module):
    def __init__(self):
        super(MTmodel, self).__init__()
        self.my_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(600, 1504),
                nn.Dropout(0.122517721),
                nn.PReLU()
            ),
            nn.Sequential(
                nn.Linear(1504, 1760),
                nn.Dropout(0.125659318),
                nn.PReLU()
            ),
            nn.Sequential(
                nn.Linear(1760, 736),
                nn.Dropout(0.125674157),
                nn.PReLU()
            ),
            
            nn.Linear(736, 1)
        ])
        #self.float()
    
    def forward(self, x):
        for layer_step in self.my_layers:
            x = layer_step(x)
        return x
    

def predict(smiles):
    pm = PolymerSmiles(smiles)
    fingerprint = np.array(pm.fingerprint_polyBERT)
    if fingerprint.size > 0 and isinstance(fingerprint[0], str):
        fingerprint = np.array([float(x) for x in fingerprint])

    model = MTmodel()

    # Load the state_dict into the model
    state_dict_path = 'fine_tuning_polyBERT_no_freeze.pth'  # Replace with the actual path to your state_dict file
    state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))  # Load the state_dict

    # Load the state_dict into the model
    model.load_state_dict(state_dict)

    # Set the model to evaluation mode
    model.eval()

     # Convert data to PyTorch tensor
    input_data = torch.tensor(fingerprint, dtype=torch.float32)
    # Make predictions
    with torch.no_grad():
        output = model(input_data)

    scalar = MinMaxScaler()
    df = pd.read_pickle("data/updated_polymers.pth")
    scalar.fit_transform((df["Egc"]).values.reshape(-1, 1))
    
    return scalar.inverse_transform((output.numpy()).reshape(-1,1))
    
def original(smiles):
    df = pd.read_pickle("data/updated_polymers.pth")
    result_df = df[df['smiles'] == smiles]
    
    return result_df

