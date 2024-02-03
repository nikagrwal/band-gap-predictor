import numpy as np
from psmiles.psmiles import PolymerSmiles
from rdkit import Chem
from importlib import util
from rdkit.Chem import rdFingerprintGenerator
from sklearn.preprocessing import MinMaxScaler 
import torch
import torch.nn as nn
import pandas as pd




class Smiles(PolymerSmiles):
    def __init__(self, smiles: str, deactivate_warnings: bool = True):
        self.smiles = smiles
        psmiles = self.smiles
        super().__init__(psmiles, deactivate_warnings)

    def can_molecule(self):
        mol = Chem.MolFromSmiles(self.psmiles)
        return Chem.MolToSmiles(mol)

    @property
    def fingerprint_polyBERT(self) -> np.ndarray:
        """Compute the polyBERT fingerprint

        Note:
            Calling this will pull polyBERT from the hugging face hub.

        Returns:
            np.ndarray: polyBERT fingerprints
        """
        assert util.find_spec("sentence_transformers"), (
            "PolyBERT fingerprints require the 'sentence-transformers' Python package."
            " Please install with "
            "`pip install 'psmiles[polyBERT]@git+https://github.com/"
            "Ramprasad-Group/psmiles.git'` "
            "Or "
            "`poetry add git+https://github.com/"
            "Ramprasad-Group/psmiles.git -E polyBERT` "
        )

        can_smiles = self.can_molecule()

        from sentence_transformers import SentenceTransformer

        polyBERT = SentenceTransformer("kuelumbus/polyBERT")

        return polyBERT.encode([can_smiles], show_progress_bar=False)[0]

    @property
    def fingerprint_circular(self) -> np.ndarray:
        """Compute the circular (Morgen) count fingerprint
        
        Returns:
            numpy.ndarray: circular fingerprint
        """

        fp_gen = rdFingerprintGenerator.GetMorganGenerator()
        return fp_gen.GetCountFingerprintAsNumPy(
            Chem.MolFromSmiles(self.smiles)
        ).astype(int)
    
class MTmodel(nn.Module):
    def __init__(self):
        super(MTmodel, self).__init__()
        self.my_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2048, 1888),
                nn.Dropout(0.296708814),
                nn.PReLU()
            ),
            nn.Sequential(
                nn.Linear(1888, 416),
                nn.Dropout(0.103316943),
                nn.PReLU()
            ),
            nn.Sequential(
                nn.Linear(416, 1632),
                nn.Dropout(0.178598433),
                nn.PReLU()
            ),
            
            nn.Linear(1632, 1)
        ])
        #self.float()
    
    def forward(self, x):
        for layer_step in self.my_layers:
            x = layer_step(x)
        return x
    

def predict(smiles):
    sm = Smiles(smiles)
    fingerprint = np.array(sm.fingerprint_circular)
    if fingerprint.size > 0 and isinstance(fingerprint[0], str):
        fingerprint = np.array([float(x) for x in fingerprint])

    model = MTmodel()

    # Load the state_dict into the model
    state_dict_path = 'models/molecule_circular.pth'  # Replace with the actual path to your state_dict file
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
    df = pd.read_pickle("data/updated_molecules.pth")
    scalar.fit_transform((df["Egc"]).values.reshape(-1, 1))
    
    return scalar.inverse_transform((output.numpy()).reshape(-1,1))   
   

def original(smiles):
    df = pd.read_pickle("data/updated_molecules.pth")
    result_df = df[df['smiles'] == smiles]
    
    return result_df
    