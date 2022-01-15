import os
import streamlit as st
from PIL import Image

def callAPI(model, imageDocker, imagePath):
    cmd = f'cd ../server && sudo cog predict {imageDocker}:latest -i image=@{imagePath} -i model={model} && mv -f ./output.png ../client/image/{model}'    
    os.system(cmd)

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
            img_save_path = os.path.join("../server/content/real/gan", image_file.name)
            
            with col1:
                st.image(img)
            
            with open(img_save_path, "wb") as f:
                f.write((image_file).getbuffer())    
            
            img_late = "./image/gan/output.png"
            img_path = "./content/real/gan/" + image_file.name
            callAPI("gan", 'tvm-com-vision', img_path)
              
            with col2:
                if os.path.exists(img_late) and os.path.isfile(img_late):
                    st.image(img_late)

    elif choice == 'OBJECT DECTECTION':
        image_file = st.file_uploader('OBJECT DECTECTION',type=['jpeg', 'jpg', 'jpe','png', 'bmp'], key=2)
        if image_file is not None:
            col1, col2 = st.columns(2)
            
            img = load_image(image_file) 
            img_save_path = os.path.join("../server/content/real/detection", image_file.name)
            
            with col1:
                st.image(img)
            
            with open(img_save_path, "wb") as f:
                f.write((image_file).getbuffer())    
            
            img_late = "./image/detection/output.png"
            img_path = "./content/real/detection/" + image_file.name
            callAPI("detection", 'tvm-com-vision', img_path)
              
            with col2:
                if os.path.exists(img_late) and os.path.isfile(img_late):
                    st.image(img_late)
                
if __name__ == '__main__':
    main()
    