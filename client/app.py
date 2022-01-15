import os
import streamlit as st


def callAPI(model, imageDocker, imagePath):
    cmd = f'cd ../server && sudo coq predict {imageDocker}:latest -i image=@{imagePath} -i model={model}'    
    os.system(cmd)

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
    
        Image = st.file_uploader('CONVERT TO DIGITAL IMAGE',type=['jpeg', 'jpg', 'jpe','png', 'bmp'], key=1)
        if Image is not None:
            col1, col2 = st.columns(2)
            Image = Image.read()                 
            with col1:
                st.image(Image)
            with col2:
                pass

    elif choice == 'OBJECT DECTECTION':
        Image = st.file_uploader('OBJECT DECTECTION',type=['jpeg', 'jpg', 'jpe','png', 'bmp'], key=2)
        if Image is not None:
            col1, col2 = st.columns(2)
            Image = Image.read()                 
            with col1:
                st.image(Image)
            with col2:
                pass

if __name__ == '__main__':
    callAPI("gan", "cc", )