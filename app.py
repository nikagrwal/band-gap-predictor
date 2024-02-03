import streamlit as st
import about_project
import predictor, tutorial



tab1,tab2, tab3 = st.tabs(["My Project", "Predictor", "Tutorial"])

about_project.display(tab1)
predictor.display(tab2)
tutorial.display(tab3)
