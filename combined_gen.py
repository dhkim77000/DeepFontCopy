import numpy as np
import pandas as pd
import torch
import math
import more_itertools as mit
import sys, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from glob import glob
from unicode_gen import unicode_gen
import pdb
import cv2
import pdb
import string
import random
from augmentation import augmentation
import sys


def data_gen(fonts_path, dataset_path, start, end, length = 5,  is_upper = True):
    #fonts_path = /home/dhkim/data/
    #dataset_path = /home/dhkim/datasets/

    start = int(start)
    end = int(end)

    font_list = glob(fonts_path+ "/*.ttf")
    font_list.sort()

    if end == -1:
        font_list = font_list[start:]
    else:
        font_list = font_list[start:end]

    os.makedirs(dataset_path  + "train/", exist_ok = True)
    os.makedirs(dataset_path  + "val/", exist_ok = True)
    os.makedirs(dataset_path  + "test/", exist_ok = True)

    for i, font_path in tqdm(enumerate(font_list)):
        font_name = font_path.replace(".ttf", "").replace("/home/dhkim/data/","").replace(" ", "")

        if(check_U_L(font_path)):
            n = i % 10
            if n>0 & n <= 7: mode = "train/"
            elif n > 7 & n < 9: mode = "val/"
            else: mode = "test/"
                
            save_path = dataset_path + mode + font_name 

            combined = combined_img_gen(font_path)
            combined.save('{}.png'.format(save_path))
            

def check_U_L(font_path):

    font = ImageFont.truetype(font = font_path, size = 512)
    lower_upper = []
    
    for char in ['a','A']:
        new_s = 600
        test_img = Image.new('L', (new_s, new_s), color='white')
        test_DrawPad = ImageDraw.Draw(test_img)
        test_DrawPad.text((0,0), char, font=font, fill='black')
        test_img = test_img.resize((64,64),Image.ANTIALIAS)
        img = np.array(test_img)/255
        lower_upper.append(img)
    l = []
    u = []
    #return lower_upper
    for i in range(len(lower_upper[0])):
        count = len(np.where(lower_upper[0][i] == 0)[0])
        l.append(count)
    for i in range(len(lower_upper[1])):
        count = len(np.where(lower_upper[1][i] == 0)[0])
        u.append(count)
        
    u = np.array(u)
    l = np.array(l)

    if(l.sum() == 0): 
        return False
        
    ratio = (u.sum()/l.sum())
    if ratio > 1.2:
        return True
    if ratio < 0.8:
        return True
    else:
        return False



def combined_img_gen(font_path):#images of all 26 characters
    Upper = np.array(list(string.ascii_uppercase))
    Lower = np.array(list(string.ascii_lowercase))
    font = ImageFont.truetype(font = font_path, size = 512)
    
    #out_test =  Image.new('RGB', (64*26,64 * 2))
    out_test =  Image.new('RGB', (64*26*2,64))
    for i, char in enumerate(Upper):
       
        x, y = font.getsize(char)
        m = max(x,y)
        pad = int(m/50)
        new_s = m+pad
        strip_width, strip_height = new_s, new_s

        test_img = Image.new('RGB', (new_s, new_s), color='white')
       
        test_DrawPad = ImageDraw.Draw(test_img)

        text_width, text_height = test_DrawPad.textsize(char, font)
        position = ((strip_width-text_width)/2,(strip_height-text_height)/2)

        test_DrawPad.text(position, char, font=font, fill='black')

                                        
        test_img = test_img.resize((64,64),Image.ANTIALIAS)
        
        out_test.paste(test_img, (i*64,0))
        
    for i, char in enumerate(Lower):
       
        x, y = font.getsize(char)
        m = max(x,y)
        pad = int(m/50)
        new_s = m+pad
        strip_width, strip_height = new_s, new_s

        test_img = Image.new('RGB', (new_s, new_s), color='white')
       
        test_DrawPad = ImageDraw.Draw(test_img)

        text_width, text_height = test_DrawPad.textsize(char, font)
        position = ((strip_width-text_width)/2,(strip_height-text_height)/2)

        test_DrawPad.text(position, char, font=font, fill='black')

                                        
        test_img = test_img.resize((64,64),Image.ANTIALIAS)
        
        out_test.paste(test_img, ((26+i)*64,0))
    return out_test

fonts_path = "/home/dhkim/data/"
dataset_path = "/home/dhkim/mc-gan-master/datasets/Combined2/"
os.makedirs(dataset_path, exist_ok = True)
random.seed(10)
start = sys.argv[1]
end= sys.argv[2]
data_gen(fonts_path,dataset_path, start, end)