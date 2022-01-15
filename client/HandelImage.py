import os
import streamlit

def callAPI(model, imageDocker, imagePath):
    cmd = f'cd ../server && sudo cog predict {imageDocker}:latest -i image=@{imagePath} -i model={model} && mv -f ./output.png ../client/image/{model}'    
    os.system(cmd)
    
def HandlelImage(image_file, model):
    img_save_path = os.path.join("../server/content/real/{}".format(model), image_file.name)
    
    with open(img_save_path, "wb") as f:
        f.write((image_file).getbuffer())    
            
    img_late = "./image/{}/output.png".format(model)
    img_path_server = "./content/real/{}/".format(model) + image_file.name
    callAPI(model, 'tvm-com-vision', img_path_server)
    
    cmd_remove = f'rm {img_save_path}'
    os.system(cmd_remove)
    
    return img_late
