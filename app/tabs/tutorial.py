
import streamlit as st
from tabs.trials import base_molecule, base_polymer, fine_tuning, frozen_featurization

def display(tab):
    tab.write("Here are examples on how to train base models for molecules as well as polymers. You can also find tutorial on how to implement transfer learning techniques used in the project.")
    tab1, tab2, tab3, tab4 = tab.tabs(["Molecule Base Model", "Polymer Base Model", "Fine Tuning", "Frozen Featurization"])

    base_molecule.display(tab1)
    base_polymer.display(tab2)
    fine_tuning.display(tab3)
    frozen_featurization.display(tab4)