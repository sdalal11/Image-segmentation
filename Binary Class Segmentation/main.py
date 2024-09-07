import os
from model import *
from PIL import Image
import streamlit as st

model_path = "/Users/sanjana/Desktop/AMNEX/binary segmenttion/InceptionResNetV2-UNet.h5"
model = load_model(model_path)


# Function to load and display the image
def load_image(file):
    image = Image.open(file)
    image.save('image.png')
    return image
 
def main():
    # Main Streamlit code
    # st.title("Building Semantic Segmentation App")

    custom_style = (
    "<style>"
    "h1 {"
    "    font-size: 38px;"
    "    color: pink;"
    "    font-family: 'Times New Roman', Times, serif;"
    "}"
    "</style>"
)

    # Render the custom style
    st.markdown(custom_style, unsafe_allow_html=True)

    # Display the title
    st.title("Building Semantic Segmentation App")

    # Upload image through Streamlit
    file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    # Check if an image is uploaded
    if file is None:
        st.text("Please upload an image file")
    else:
        # Load and display the image
        image = load_image(file)

        # Make predictions
        predictions = predict('image.png', model)
        cv2.imwrite('mask.png', predictions)

        # Create two columns
        col1, col2 = st.columns(2)

        # Display the first image in the first column
        col1.image('image.png', use_column_width=True, caption='Uploaded Image')

        # Display the second image in the second column
        col2.image('mask.png', use_column_width=True, caption='Predicted Mask')

        
        os.remove('image.png')
        os.remove('mask.png')
        # class_names = ['Building', 'Background']

        # predicted_class = class_names[np.argmax(predictions)]

        # st.write("Predicted Image:", predicted_class)

        
        # if predicted_class == 'Building':
        #     st.balloons()
        #     st.sidebar.success("Building Detected!")

        # elif predicted_class == 'Background':
        #     st.sidebar.warning("Background Detected!")

if __name__ == "__main__":
    main()