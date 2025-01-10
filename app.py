import streamlit as st

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# All preprocessing function
from preprocess import preprocess_input

# Page configuration
st.set_page_config(
  page_title="Bababa-babanana",
  page_icon="🍌",
  menu_items={
    "About": """
      ### Deep Learning (by: CDEF)
      * 211110347 - [Cindy Sintiya](https://github.com/cindysintiya)
      * 211110933 - [Dhea Romantika Marpaung](https://github.com/aadeeee)
      * 211110101 - [Erin Gunawan](https://github.com/eringnwn)
      * 211111610 - [Farrell Rio Fa](https://github.com/r10f)
    """
  }
)

model, model2 = None, None

def load_model():
  global model, model2
  if model is None:
    _, col_mid, _ = st.columns([1.25, 2, 1])
    with col_mid:
      with st.spinner("Loading model for the first time..."):
        import tensorflow as tf
        from tensorflow.keras import Model

        model = tf.keras.models.load_model('model/my_model.keras')
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Redefine model to output right after the first hidden layer
        ixs = [2,4,6]
        model2 = Model(inputs=model.inputs, outputs=[model.layers[i].output for i in ixs])  
  return model


def process_prediction(uploaded_file, with_discrete):
  next_predict = False
  st.markdown("<h2 style='text-align: center;'>- Preprocessed Image -</h2>", unsafe_allow_html=True)
  with st.spinner("Processing your file..."):
    # Convert to image for preprocessing
    img = Image.open(uploaded_file)
    combined_image, img_clean, yellow_mask, green_mask, img_edge = preprocess_input(img)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
      st.image(img_clean, caption="Segmentation", use_container_width=True)
    with col2:
      st.image(img_edge, caption="Egde", use_container_width=True)
    with col3:
      st.image(yellow_mask, caption="Yellow Mask", use_container_width=True)
    with col4:
      st.image(green_mask, caption="Green Mask", use_container_width=True)

    if combined_image is not None:
      next_predict = True
    else:
      st.error("Classified as Not a Banana based on edge density.")

  if next_predict :
    st.markdown("<h2 style='text-align: center;'>- Predicting Ripeness -</h2>", unsafe_allow_html=True)
    load_model()
    with st.spinner("Predicting..."):
      predict_img = [combined_image]
      predict_img = np.array(predict_img, dtype=float)
      predict_img /= 255.0
      prediction = model.predict(predict_img)

      if (prediction < 0.5):
        st.success("Objek di atas dinyatakan sebagai pisang MENTAH dengan tingkat kematangan {:.5f}%".format(prediction[0][0]*100))
      else:
        st.warning("Objek di atas dinyatakan sebagai pisang MATANG dengan tingkat kematangan {:.5f}%".format(prediction[0][0]*100))
    
    if with_discrete:
      st.markdown("<h2 style='text-align: center;'>- Discrete from Each Layer -</h2>", unsafe_allow_html=True)
      with st.spinner("Getting Each Layers' Discrete..."):
        with st.expander("See Details"):
          get_discrete(predict_img)


def get_discrete(predict_img) :
  global model2
  feature_maps = model2.predict(predict_img)

  # Hasil conv 2D layer 2
  square = 8
  for fmap in feature_maps[:1]:
    ix = 1
    fig, axes = plt.subplots(square, square, figsize=(8, 8))
    for i in range(square):
      for j in range(square):
        # Access the specific subplot (i, j)
        ax = axes[i, j]
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks
        # Plot the filter channel in grayscale (use the correct index for feature map)
        ax.imshow(fmap[0, :, :, ix-1], cmap='gray')
        ix += 1
    st.markdown("<h3 style='text-align: center;'>- 2nd Layer: Convolution 2D -</h3>", unsafe_allow_html=True)
    st.pyplot(fig)  # Display the figure in Streamlit

  # Hasil conv 2D layer 4
  square = 8
  for fmap in feature_maps[1:2]:
    ix = 1
    fig, axes = plt.subplots(square, square, figsize=(8, 8))
    for i in range(square):
      for j in range(square):
        # Access the specific subplot (i, j)
        ax = axes[i, j]
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks
        # Plot the filter channel in grayscale (use the correct index for feature map)
        ax.imshow(fmap[0, :, :, ix-1], cmap='gray')
        ix += 1
    st.markdown("<h3 style='text-align: center;'>- 4th Layer: Convolution 2D -</h3>", unsafe_allow_html=True)
    st.pyplot(fig)  # Display the figure in Streamlit

  # Hasil flatten layer 6 sbg input LSTM
  flatten_map = feature_maps[2:3][0][0]
  st.markdown("<h3 style='text-align: center;'>- 6th Layer: Flatten -</h3>", unsafe_allow_html=True)
  st.write(np.vectorize(lambda x: f'{x:.3f}')(flatten_map).reshape(1, -1))


# Main page
col1, col2 = st.columns([1, 4])
with col1:
  st.image("https://www.shutterstock.com/image-vector/banana-relax-sunglasses-cartoon-mascot-600nw-2402786913.jpg")
with col2:
  st.markdown("<h1 style='text-align: center;'>Banana Ripeness Detector</h1>", unsafe_allow_html=True)

input_container = st.container(border=True)
uploaded_file = input_container.file_uploader(
  label="Upload a Banana Image",
  type=["jpg","jpeg","png"],
)
if uploaded_file is not None:
  st.session_state.image = uploaded_file
  st.session_state.source = "upload"
else :
  # Remove uploaded image and prepare for capturing
  if "source" in st.session_state and st.session_state.source == "upload" and "image" in st.session_state:
    del st.session_state.image

  @st.dialog("Take a Photo")
  def capture_image():
    st.write("Take a picture of a piece of banana. Make sure it's fully visible and not blurry.")
    picture = st.camera_input("Take a picture")
    if picture:
      st.session_state.image = picture
      st.rerun()

  input_container.markdown("<p style='text-align: center;'>OR</p>", unsafe_allow_html=True)
  take_photo = input_container.button("📸 Take a Photo", use_container_width=True)
  if take_photo:
    st.session_state.source = "camera"
    capture_image()

if "image" in st.session_state:
  _, col_mid, _ = st.columns([1, 3, 1])  # Adjust column widths as needed
  # Display images in 2nd column only
  with col_mid:
    st.image(st.session_state.image, caption="User's uploaded Banana Image" if st.session_state.source=="upload" else "User's captured Banana Image", use_container_width=True)

  # col1, col2 = st.columns([2.25, 1])
  # with col1:
  #   predicting = st.button("Predict Ripeness", use_container_width=True)
  # with col2:
  #   with_discrete = st.checkbox("Show each layer's discrete")
  
  predicting = st.button("Predict Ripeness", use_container_width=True)
  with_discrete = st.checkbox("Show each layer's discrete")

  if predicting:
    process_prediction(st.session_state.image, with_discrete)


footer_html = """
  <style>
    footer {
      position: fixed;
      left: 0;
      bottom: 0;
      padding: 10px;
      width: 100%;
      background-color: #FDFD96;
      color: black;
      text-align: center;
    }
  </style>
  <footer style='text-align: center;'>
    <p style='margin:0'>Copyright &copy; 2024 by (AB)CDEF: Cindy, Dhea, Erin, Farrell. All Right Reserved.</p>
  </footer>
"""
st.markdown(footer_html, unsafe_allow_html=True)