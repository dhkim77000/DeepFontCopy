import numpy as np
import pandas as pd
import torch
import math
import more_itertools as mit
import sys, os
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import pdb
import cv2
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
from tqdm import tqdm
import cv2
import string
import random
import string
import random
import re
import torch
import torchvision
import torch.nn as nn
import math
from torch.utils.data import Dataset
from torchvision import transforms
from segmentation import classification_data_gen
import math
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms

def data_gen(image_name, imgset_path):

    #dataset_path = /home/dhkim/post_process/data_classified/.....e
    #imgset_path = /home/dhkim/post_process/data_classified/....

    img_path = list(os.walk(imgset_path))

    img_list = []

    text = []
    p = re.compile(".*")
    #pdb.set_trace()
    for i,img in enumerate(img_path[0][2]):

        if i == 0:
            file_format = p.findall(img)[0]
            index = list(file_format).index('.') + 1
            file_format = file_format[index:]

        text.append(img.replace("."+file_format, ""))

        img = imgset_path + img
        img_list.append(img)

    text.sort()
    img_list.sort()

    if len(text) > 5:
        index_list = []
        new_text = []
        new_img_list = []
        for i in range(5):
            num = random.randint(0,len(text)-1)
            while num in index_list:
                num = random.randint(0,len(text)-1)
            index_list.append(num)

        for i in index_list:
            new_text.append(text[i])
            new_img_list.append(img_list[i])
        
        new_text.sort()
        new_img_list.sort()
        
        text = new_text
        img_list = new_img_list

    
    dir = "/home/dhkim/mc-gan-master/datasets/public_web_fonts/" + image_name +"/"

    os.makedirs(dir, exist_ok = True)

    image_saver(dir, text, image_name, img_list)
        
            
def image_saver(dir, text_list, image_name, img_list):

    os.makedirs(dir+"A/", exist_ok = True)
    os.makedirs(dir+"A/train/", exist_ok = True)
    os.makedirs(dir+"A/test/", exist_ok = True)
    os.makedirs(dir+"B/", exist_ok = True)
    os.makedirs(dir+"B/train/", exist_ok = True)
    os.makedirs(dir+"B/test/", exist_ok = True)

    upper = np.array(list(string.ascii_uppercase))
    lower = np.array(list(string.ascii_lowercase))
    chars = list(np.concatenate((upper,lower), axis = 0))

    A_train_save_path = dir + "A" + "/" + "train/" + image_name +"_"
    A_test_save_path = dir + "A" + "/" + "test/" + image_name

    B_train_save_path =  dir + "B" + "/" + "train/" + image_name +"_"
    B_test_save_path = dir + "B" + "/" + "test/" + image_name
    
  
    train_img_gen_A(text_list,chars, img_list, A_train_save_path)
    test_img_gen_A(text_list,chars,img_list, A_test_save_path )
    
    train_img_gen_B(text_list, chars, img_list, B_train_save_path)
    test_img_gen_B(B_test_save_path)

def train_img_gen_A(text_list, chars, img_list, save_path) :#images of 4 characters which one of them is omitted

  
    p = re.compile("(?<=\.).*")
   
    file_format = p.findall(img_list[0])[0]

    #file_format = 'jpg'

    text = ''.join(text_list)

    for i in range(len(img_list)):

        out_train = Image.new('RGB', (64*26,64), color = 'white')
        temp_text = text_list.copy()
        temp_img_list = img_list.copy()
        try:
            del_index = chars.index(temp_text[i]) 
            del temp_text[i]
            del temp_img_list[i]
        except ValueError as e:
            continue
        
        pattern = "(?<=\/)\w+." + file_format
        p = re.compile(pattern)
    
        
        for img in temp_img_list:
      
            character = p.findall(img)[0].replace(file_format,"").replace(".","")
            try:
                index = chars.index(character) 
            except ValueError as e:
                continue
            img = Image.open(img)              #64 X 6
            img = img.resize((64,64))
            out_train.paste(img, (index *64,0))

        out_train.save('{}.png'.format(save_path + str(del_index)))

    print("------A train Done------")

def test_img_gen_A(text_list,chars, img_list, save_path):#images of all five characters

    out_test =  Image.new('RGB', (64*26,64), color = 'white')

    for i in range(len(img_list)):

        character = text_list[i]
        img = Image.open(img_list[i])
        img = img.resize((64,64))
        try:
            index = chars.index(character)
        except ValueError as e:
            continue
        out_test.paste(img, (index*64,0))

    out_test.save('{}.png'.format(save_path))
    print("------A test Done------")


def train_img_gen_B(text_list, chars, img_list, save_path):#image of single characters which is omiited

    

    p = re.compile("(?<=\.).*")
    file_format = p.findall(img_list[0])[0]

    pattern = "(?<=\/)\w+." + file_format
    p = re.compile(pattern)

    for i, img in enumerate(img_list):

     
        character = text_list[i]
        index = chars.index(character) 
        img = Image.open(img)
        img.resize((64,64))
        img.save('{}.png'.format(save_path+str(index)))
    print("------B train Done------")
    
def test_img_gen_B(save_path):  #images of all 26 characters

    out_test =  Image.new('RGB', (64*26,64), color = 'white')
    out_test.save('{}.png'.format(save_path))
    print("------B test Done------")

def vote(voter1, voter2, voter3, threshold = 0.7):

    labels="0123456789ABCDEF G H I J K L M N O P Q R S T U V W X Y Z a b d e f g h n q r t"
    labels = labels.replace(" ", "")
    labels = list(labels)
    count = list(np.zeros(len(labels)))

    percentage1, predicted1 = torch.max(voter1.data, 1)  #use sigmoid as a last layer activation function
    if percentage1 > threshold: count[predicted1] += 1

    percentage2, predicted2 = torch.max(voter2.data, 1)
    if percentage2 > threshold: count[predicted1] += 1

    percentage3, predicted3 = torch.max(voter3.data, 1)
    if percentage3 > threshold: count[predicted1] += 1 

    result = count.index(np.max(count))
    
    return labels[result]

if __name__ == "__main__":

    image_name = sys.argv[1]
    p = re.compile("(?<=\.).*")
    file_format = p.findall(image_name)[0]
    image_name = image_name.replace("."+file_format,"")
    imgset_path = '/home/dhkim/post_process/data_classified/' + image_name + "/"
    data_gen(image_name, imgset_path)
