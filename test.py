import os
import cv2
from tqdm import tqdm

VALID_FORMATS = {
    'jpeg', 'jpg', 'jpe',
    'png', 'bmp',
}

def main():
    img_dir = "./content/images/gan"
    files = os.listdir(img_dir)
    files = [f for f in files if is_valid_file(f)]
    
    for fname in tqdm(files):
        print(fname)
        image = cv2.imread(os.path.join(img_dir, fname))
        

def is_valid_file(fname):
        ext = fname.split('.')[-1]
        return ext in VALID_FORMATS


if __name__ == '__main__':
    main()
    