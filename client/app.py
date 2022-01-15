import os
import streamlit as st
from PIL import Image
from HandelImage import HandlelImage

def load_image(image_file):
    img = Image.open(image_file)
    return img

def main():
    st.set_page_config(layout="wide")
    
    st.image(os.path.join('image','banner.jpg'), use_column_width  = True)
    st.markdown("<h1 style='text-align: center; color: white;'>COMPUTER VISION</h1>", unsafe_allow_html=True)

    menu = ['CONVERT TO DIGITAL IMAGE', 'OBJECT DECTECTION']
    st.sidebar.header('Selection')
    choice = st.empty()
    choice = st.sidebar.selectbox('', menu)
        
    # Create the Home page
    if choice == 'CONVERT TO DIGITAL IMAGE':
        image_file = st.file_uploader('CONVERT TO DIGITAL IMAGE',type=['jpeg', 'jpg', 'jpe','png', 'bmp'], key=1)
        
        if image_file is not None:
            col1, col2 = st.columns(2)
            
            img = load_image(image_file) 
            img_late = ''
               
            with col1:
                st.image(img)
                if st.button("COVERT"):
                    img_late = HandlelImage(image_file, "gan")
            
            with col2:
                if os.path.exists(img_late) and os.path.isfile(img_late):
                    st.image(img_late)
                    with open(img_late, "rb") as image:
                        btn = st.download_button(
                                label="Download image",
                                data= image,
                                file_name="output.png",
                                mime="image/png"
                            )

    elif choice == 'OBJECT DECTECTION':
        image_file = st.file_uploader('OBJECT DECTECTION',type=['jpeg', 'jpg', 'jpe','png', 'bmp'], key=2)
        
        if image_file is not None:
            col1, col2 = st.columns(2)
            
            img = load_image(image_file) 
            img_late = ''
            
            with col1:
                st.image(img)
                if st.button("DETECT"):
                    img_late = HandlelImage(image_file, "detection")
            
            with col2:
                if os.path.exists(img_late) and os.path.isfile(img_late):
                    st.image(img_late)
                    with open(img_late, "rb") as image:
                        btn = st.download_button(
                                label="Download image",
                                data= image,
                                file_name="output.png",
                                mime="image/png"
                            )
if __name__ == '__main__':
    main()
    