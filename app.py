import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
from PIL import Image
from preprocess import *

# page config
st.set_page_config(
  page_title="Babanana",
  page_icon="🍌",
  menu_items={
    'About': "# Tugas Deep Learning"
  }
)

# main page
st.title('Banana Ripeness Detector')
@st.dialog("Take a Photo")
def capture():
  st.write("Take a photo of a banana. Make sure it is xxx")
  picture = st.camera_input("Take a picture")
  if picture:
    st.session_state.image = picture
    st.rerun()

input_container = st.container(border=True)
input_image = input_container.file_uploader(
  label='Upload banana image',
  type=['jpg','jpeg','png'],
)
if (input_image):
  st.session_state.image =  Image.open(input_image)
  input_image = None

input_container.markdown('''<p style="text-align: center;">OR</p>''', unsafe_allow_html=True)
take_photo = input_container.button('Take a Photo', use_container_width=True)
if take_photo:
  capture()

if "image" in st.session_state:
  st.header('Uploaded Image')
  st.write(st.session_state.image)
  if st.button('Check Ripeness'):
    st.header('Prediction Result')
    
    list_img = preprocess_input(st.session_state.image)
    try:
      yellow_mask, green_mask, img_edge = list_img[0]
      combined_img = list_img[1]
      x_predict = [combined_img]
      x_predict = np.array(x_predict, dtype=float)
      x_predict /= 255.0
      
      prediction = 0.4
      # comment code above when model is exported
      # uncomment code below when model is exported
      # prediction = model.predict(predict)
      if (prediction < 0.5):
        st.success('Pisang Mentah')
      else:
        st.warning('Pisang Matang')
    except:
      st.error('Bukan Pisang')
    
    with st.expander("See Details"):
      col1, col2, = st.columns(2, gap='small')
      col3, col4 = st.columns(2, gap='small')
      with col1:
        st.write('Original Image')
        st.image(st.session_state.image, width=200)
      with col2:
        st.write('Yellow Masked')
        st.image(yellow_mask, width=200)
      with col3:
        st.write('Green Masked')
        st.image(green_mask, width=200)
      with col4:
        st.write('Edge Detection')
        st.image(img_edge, width=200)


# footer
footer_html = """
<style>
  footer {
    position: fixed;
    left: 0;
    padding: 16px;
    bottom: 0;
    width: 100%;
    background-color: #FDFD96;
    color: black;
    text-align: center;
  }
</style>
<footer style='text-align: center;'>
  <p style='margin:0'>Copyright &copy; 2024 by (AB)CDEF: Cindy, Dhea, Erin, Farrel. All Right Reserved.</p>
</footer>
"""
st.markdown(footer_html, unsafe_allow_html=True)