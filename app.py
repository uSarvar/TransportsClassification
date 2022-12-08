
import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# set a title
st.title('Classification model of transports')

# download pics
file = st.file_uploader('Download pics', type=['png','jpeg','jpg','gif','svg'])

if file:
   st.image(file)
   # PIL convert
   img = PILImage.create(file)
   # model
   model = load_learner('transport_model.pkl')
   # prediction
   pred, pred_id, probs = model.predict(img)
   st.success(f'Prediction: {pred}')
   st.info(f'Probability: {probs[pred_id]*100:.1f}%')
   # plotting
   fig = px.bar(x=probs*100, y=model.dls.vocab)
   st.plotly_chart(fig)
