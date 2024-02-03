import streamlit as st
from tabs import about_project, predictor, tutorial



tab1,tab2, tab3 = st.tabs(["My Project", "Predictor", "Tutorial"])

about_project.display(tab1)
predictor.display(tab2)
tutorial.display(tab3)
