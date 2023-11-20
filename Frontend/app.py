import requests
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import json
import os
from PIL import Image
import io



st.set_option("deprecation.showfileUploaderEncoding", False)


# defines an h1 header
st.title("Predict the winner champion afrique")
st.subheader("Enter details below")
    
st.subheader("Or Enter your historical transactions csv file")
# displays a file uploader widget
data = st.file_uploader("Choose a csv file for data")
predict = st.file_uploader("Choose a csv file for predict")


if data is not None and predict is not None:
    file = {"file_predict": predict.getvalue(),"data_file": data.getvalue()}
    res = requests.post("http://127.0.0.1:8080/predict/csv", files=file)
    #res = requests.post("https://backendapiahmed.herokuapp.com/predict/csv", files=file)
    #res = requests.post("http://lb-backendapp-1500115353.us-east-1.elb.amazonaws.com/predict/csv", files=file)
    
    final_tables = res.json().get('final_table')
    j=0
    for ft in final_tables:
        j+=1
        st.header(f"Group {'ABCDEF'[j-1]}")
        for i in ft:
           st.text("%s -------- %d"%(i[0], i[1]))
    
    # Côté client (Streamlit)
    import base64

    # Décoder l'image base64
    image_bytes = base64.b64decode(res.json().get("img"))

    # Ouvrir l'image avec PIL
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption='Image from Bytes', use_column_width=True)


    
     