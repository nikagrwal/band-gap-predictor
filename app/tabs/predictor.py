from tabs.molecule_predictor import predict as m_predictor, original as m_original
from tabs.polymer_predictor import predict as p_predictor, original as p_original

import torch.nn as nn
import streamlit as st

def molecule_predictor(smiles):

    return m_predictor(smiles)

def polymer_predictor(smiles):
    return p_predictor(smiles)



def display(tab):
    
    tab.header("Try our Bandwidth(Egc) predictor")
    
    options = ['Molecule', 'Polymer']
    placeholder = tab.empty()

    # Dropdown Streamlit
    selected_option = tab.selectbox('Select an option:', ['Please select either molecule or polymer', *options])

    smiles_string = tab.text_input('Please enter the smiles string:', placeholder='Type here....')

    # Create a button to trigger the prediction
    if tab.button('Predict'):
        # Validate if both the input field and dropdown have values
        if smiles_string and selected_option:
            smiles_placeholder = tab.empty()
        

            # Call your prediction function here
            if selected_option == "Molecule":
              
                prediction_result = m_predictor(smiles_string)
                original_result = m_original(smiles_string)

            else:
                
                prediction_result = p_predictor(smiles_string)
                original_result = p_original(smiles_string)
           
            
            tab.write("Predicted Egc value:")
            prediction_placeholder = tab.empty()
            tab.write("Original Egc value:")
            original_placeholder = tab.empty()
            prediction_placeholder.write(prediction_result[0,0])
            
            original_placeholder.write(original_result)
        elif not smiles_string:
            # Display an error message if the input field is empty
            placeholder.error('Please enter a smiles string.')
        else:
            # Display an error message if the dropdown is not selected
            placeholder.error('Please select an option.')   

    