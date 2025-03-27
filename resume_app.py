# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 12:44:35 2023
@author: Jeevika
"""

import numpy as np
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st 

import docx2txt


from PIL import Image

#loading in the model the vectorizer file
pkl_in = open('vectorizer.pkl', 'rb')
loaded_vectorizer = pickle.load(pkl_in)

# loading in the model to predict on the data
pickle_in = open('classifier_model.pkl', 'rb')
classifier = pickle.load(pickle_in)
  
def welcome():
    return 'welcome all'
  
# defining the function which will make the prediction using 
# the data which the user inputs

def prediction(Skills):  

    r = classifier.predict(loaded_vectorizer.transform([Skills]))
    print(r)
    return r

# this is the main function in which we define our webpage

def main():

    # giving the webpage a title

    st.title("Resume Screening")
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed

    html_temp = '''
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Resume screening ML App </h1>
    </div>
    '''
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code

    st.markdown(html_temp, unsafe_allow_html = True)
      
    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction

    st.subheader("DocumentFiles")
    Skills = st.file_uploader("Upload Document", type = ["pdf","docx","doc","txt"]) 
    

    
    
 
       
    result =""
      
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result

    if st.button("Process"):
      if Skills is not None:
        file_details = {"filename":Skills.name,
                        "filetype":Skills.type,"filesize":Skills.size}
        
        raw_text = docx2txt.process(Skills)
        
        
        
        result = prediction(raw_text)
    st.success('The output is {}'.format(result))

if __name__=='__main__':
    main()
