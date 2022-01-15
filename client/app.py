import os
import streamlit as st
import asyncio

async def callAPI(model, imageDocker, imagePath):
    cmd = f'cd ../server && cog predict {imageDocker}:latest -i image=@{imagePath} -i model={model} && mv ./output.png ../client/image/{model}'    
    await os.system(cmd)

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
        
        img = st.file_uploader('CONVERT TO DIGITAL IMAGE',type=['jpeg', 'jpg', 'jpe','png', 'bmp'], key=1)
            
        if img is not None:
            col1, col2 = st.columns(2)
            img = img.read()                 
            with col1:
                st.image(img)
            with col2:
                callAPI("gan", "tvm-com-vision", str(img))
                st.image("./image/gan/output.png")

    elif choice == 'OBJECT DECTECTION':
        img = st.file_uploader('OBJECT DECTECTION',type=['jpeg', 'jpg', 'jpe','png', 'bmp'], key=2)
        if img is not None:
            col1, col2 = st.columns(2)
            img = img.read()                 
            with col1:
                st.image(img)
            with col2:
                callAPI("detection", "tvm-com-vision", str(img))
                st.image("./image/detection/output.png")

if __name__ == '__main__':
    main()
    