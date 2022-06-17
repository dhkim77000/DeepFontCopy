import torch
import torchvision
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
import more_itertools as mit
import sys, os
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from glob import glob
import pdb
import cv2
import pdb
import string
import random
import sys
import pdb
from segmentation import classification_data_gen
from rembg.bg import remove
import numpy as np
import io
from PIL import Image
from rembg.bg import remove
import numpy as np
import io
from PIL import Image

input_path = "/home/dhkim/post_process/image/k1.png"
output_path = '/home/dhkim/post_process/image/out.png'


f = np.fromfile(input_path)
result = remove(f)
img = Image.open(io.BytesIO(result)).convert("RGBA")
img.save(output_path)








#image_file = '/home/dhkim/post_process/image/'
#file_name = sys.argv[1]

#image_file = image_file + file_name
#image_dict = {'7':"A", '8':'O','11':'S', '10':'T','6':'I'}


#save_path = "/home/dhkim/post_process/data_classified/"
#threshold = 100
#contours, _,_,_ = classification_data_gen(save_path, image_file, 'out', image_dict, threshold)








