import os
from utils import compute_data_mean

rootPath = os.path.dirname(os.path.abspath(__file__))

print(compute_data_mean(rootPath+'/content/dataset/Hayao/style'))