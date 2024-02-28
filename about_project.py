import streamlit as st
from PIL import Image
def display(tab):
   tab.header(":clipboard: Motivation")
   tab.write("Polymers are a crucial part of various industries as well as our daily lives. They are also known as plastics. Whether it is healthcare, electronics, or food packaging material, polymers are used everywhere because of their versatile nature. It is not sustainable to continue using the polymers that are currently used in such large quantities. Therefore, scientists are working to find sustainable polymer substitutes. Physical exploration to find polymers with desired properties is not feasible as polymer space is huge. Machine learning models can be used to narrow down the scope for exploring the polymer space.")
   st.markdown("""---""")
   tab.header(":notebook: Objective")
   tab.write("""
             Polymer Informatics has advanced tremendously
              in the past decade. However, the amount of quality
              polymer data is very small. This hinders the improvement
              of polymer property prediction. Transferring knowledge
              from a domain which shares similarity with polymers 
             might help us improve polymer property prediction. 
             Since polymers are formed by repeating the monomers 
             as can be seen in figure below, we can use molecules 
             to transfer knowledge. The size of molecule data 
             available is large. We use this data for training base 
             models and then polymer data to achieve domain accuracy.
             """)

   image_1 = Image.open("correlation.png")
   image_1 = image_1.resize((400,400))

   
   correlation_placeholder = tab.empty()
   correlation_placeholder.image(image_1, caption = "Polymerization: The process of obtaining polymers by repeating monomers(molecules) infinite number of times")
   
   tab.header("What is Band Gap?")
   tab.write("""Within molecules, electrons are found at distinct energy levels. The lowest
layer at which an electron can excite is known as LUMO, and the highest layer
at which an electron can excite is known as HOMO. The difference between
these two energy levels is known as the HOMO-LUMO gap. However, in
polymers, these energy levels merge to form the energy bands. Electrons
involved in bonding atoms reside in an energy level known as the valence
band. The electrons that are free to move and not being used for bonding
reside in an energy level known as the conduction band. Most polymers have
an energy gap between the valence band and the conduction band. This
energy gap in which electrons cannot exist is known as the band gap. It is
described as the minimum energy required for an electron to excite from the
valence band to the conduction band.

""")
   band = Image.open("Band_gap_comparison.svg.png")
   band = band.resize((500, 350))
   band_placeholder = tab.empty()
   band_placeholder.image(band, caption = "Band gap for conductors, semiconductors, and insulators")
   tab.header("Polymer Informatic Ecosystem")
   tab.write("""The
new advancements in the fields of artificial intelligence (AI) and machine
learning (ML), combined with contemporary data and information-centric
approaches, form the field of polymer informatics, which helps tackle the
wide search space problem in polymer discovery.""")
   tab.write("""Polymer informatics has essentially five elements, as shown in fig. below.""")
   image_3 = Image.open("polymer_informatics_ecosystem.png")
   #image_3 = image_3.resize((600,450))

   polymer_informatics_ecosystem_placeholder = tab.empty()
   polymer_informatics_ecosystem_placeholder.image(image_3, caption = "Essential elements of the Polymer Informatics Ecosystem ")
   tab.write(""" 
""")
   tab.header("Transfer Learning")
   tab.write("""Most machine learning and deep learning algorithms are trained to solve specific tasks. Once the distribution of feature space changes, a new model needs to be trained, regardless of the similarity between tasks. Transfer learning helps to bridge this gap by utilising the knowledge learned from one task to solve another related task. The key motivation behind transfer learning is the shortage of labelled data. """)
   tab.write("""In this work, we compare the results of transfer learning techniques listed below.""")

   tab.subheader("Zero-shot and Few-shot Learning")
   tab.write("""Zero-shot learning is a powerful transfer-learning technique, as we do not use any data for training. Instead, we try to shift the domain to perform a similar task as the base model using semantic data such as a prompt or mathematical language. For problems where obtaining labelled data is difficult, zero-shot learning is invaluable. Since we have very little data for creating a machine learning model to predict the band gap of polymers, we can use zero-shot learning to check if we can improve the accuracy of the prediction score.""")
   tab.write("""In few-shot learning, a model can be trained to perform a task using only a small number of labelled examples per class. Unlike most machine learning algorithms, which require a large amount of data to learn a task, few-shot learning requires only a few data instances to learn a task.""")
   tab.subheader("Fine-Tuning")

   fine_image = Image.open("fine_tuning (2).png")
   tab.write("""
Fine-tuning is further training a pre-trained model to adapt it for a specific domain or task. The pre-trained model is trained using a very large data set. This pre-trained model is then retrained by changing a few hyperparameters, such as learning rate or the number of epochs. We can also choose if we want to train all the layers in the pre-trained model or only a few layers in the pre-trained model.""")
   tab.write("""In our fine-tuning approach, we start with a pre-trained base molecule model for predicting the band gap. This base model is then fine-tuned to predict the band gap for polymers using fewer polymer data instances. We initialise our neural network with the learned parameters of all the layers of the pretrained neural network. We then fix the weights of the first 0 to n layers and retrain the remaining layers. We re-train these layers using either the same learning rate as the base models or a new learning rate that is smaller than the learning rate of the base models. This helps keep common domain knowledge by controlling how much the weights change on each gradient descent.""")
   fine_image_placeholder = tab.empty()
   fine_image_placeholder.image(fine_image, caption = "Fine Tuning: (a) Original DNN. (b) A0B3: Fine-tune the weights of all three layers of DNN. (c) A1B2: Fix the weights of the first layer and fine-tune the weights of the remaining 2 layers of DNN. (d) A2B1: Fix the weights of the first two layers and fine-tune the weights of the remaining layer of DNN.")
   
   tab.subheader("Frozen Featurization")
   tab.write("""In a predictive DNN, the first few layers of the network learn about the structure of the fingerprints, and it is only the later layers that learn how to predict the properties. Since the fingerprint structures for molecules as well as polymers share similarity, we could use the first few layers from base molecule models. In frozen featurization, we start with the first few layers of the pre-trained base molecule models for band gap prediction. These few layers are initialized with the learned parameters of the base model. We fix the weights of these layers that we took from the base model. We then add a few more layers with random weights and train only these layers. We are using the learning rate of the base model to train the new layers.""")
   frozen_image = Image.open("frozen_featurization (4).png")
   frozen_image_placeholder = tab.empty()
   frozen_image_placeholder.image(frozen_image, caption = "Frozen featurization: a) X1Y1: Remove the last layer of the base model and add one extra layer. (b) X1Y2: Remove the last layer of the base model and add two extra layers. (c) X2Y1: Remove the last two layers of the base model and add one extra layer. (d) X2Y2: Remove the last two layers of the base model and add two extra layers.")

   tab.header("Results")
   tab.write("""The image below shows comparison between R2 score of base polymer models and transfer learning techniques. It is evident from the results that transfer learning improves the R2 value for predicting band gap of polymers.""")
   image_2 = Image.open("results.png")
   result_placeholder = tab.empty()
   result_placeholder.image(image_2, caption=" Comparison between r2 score of transfer learning techniques and base polymer models")

   tab.header("Conclusions and Future works")
   tab.write("""The work presented in this study proves that transfer learning methods such as fine-tuning and frozen featurization improves prediction for band gap for polymers. Machine-generated fingerprints using domain related data provides better results for property prediction than hand-crafted fingerprints. Further improvements can be provided by developing  machine-generated fingerprints using all of available material science. Using explainable AI libraries such as LIME, can explain the difference between molecules and polymers. One single model to predict properties of molecules as well as polymers, can be achieved.""")