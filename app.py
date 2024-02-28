import streamlit as st
import about_project
import predictor



tab1,tab2 = st.tabs(["My Project", "Predictor"])

about_project.display(tab1)
predictor.display(tab2)
