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

def random_text_gen(length):
    included = []
    ran = random.randint(0,25)
    
    while len(included) < length:
        if(len(included) == 0):
            included.append(ran)
        else:
            ran = random.randint(0,25)
            if(ran not in included):
                included.append(ran)
                ran = random.randint(0,25)

    included.sort()
    
    return included


def data_gen(fonts_path, dataset_path, start, end, length = 5,  is_upper = True):
    #fonts_path = /home/dhkim/data/
    #dataset_path = /home/dhkim/datasets/

    start = int(start)
    end = int(end)
    if is_upper:
        chars = np.array(list(string.ascii_uppercase))  #total 26 characters
    else:
        chars = np.array(list(string.ascii_lowercase))
    

    font_list = glob(fonts_path+ "/*.ttf")
    font_list.sort()

    if end == -1:
        font_list = font_list[start:]
    else:
        font_list = font_list[start:end]

    for font_path in tqdm(font_list):
        #data_path = "/home/dhkim/datasets/font_name/{letter}/...
        font_name = font_path.replace(".ttf", "").replace("/home/dhkim/data/","").replace(" ", "")
        dir = dataset_path + font_name +"/"
        os.makedirs(dir, exist_ok = True)

        included = random_text_gen(length)   #generate random text
        random_text = chars[included]
        random_text = ''.join(random_text)

        A_test_img = test_img_gen_A(random_text, font_path)
        B_test_img = test_img_gen_B(font_path)
        
        image_saver(random_text, dir, font_path,font_name, A_test_img, B_test_img)
            
            
def image_saver(text, dir, font_path, font_name,  A_test_img, B_test_img):

    chars = np.array(list(string.ascii_uppercase))
    full_text = "".join(chars)

    os.makedirs(dir, exist_ok = True)
    os.makedirs(dir+"A/", exist_ok = True)
    os.makedirs(dir+"A/train/", exist_ok = True)
    os.makedirs(dir+"A/test/", exist_ok = True)
    os.makedirs(dir+"B/", exist_ok = True)
    os.makedirs(dir+"B/train/", exist_ok = True)
    os.makedirs(dir+"B/test/", exist_ok = True)

    A_train_save_path = dir + "A" + "/" + "train/" + font_name +"_"
    A_test_save_path = dir + "A" + "/" + "test/" + font_name

    B_train_save_path =  dir + "B" + "/" + "train/" + font_name +"_"
    B_test_save_path = dir + "B" + "/" + "test/" + font_name
    
  
    A_test_img.save('{}.png'.format(A_test_save_path))

    B_test_img .save('{}.png'.format(B_test_save_path))

    for i in range(len(text)):
        del_char = text[i]                                      #char to be omitted -----> APPLE -> A PLE
        del_index = np.where(chars == del_char)[0][0]

        train_text = text.replace(del_char, "")                 #APLE

        
        A_train_img = train_img_gen_A(train_text, font_path, del_index)
        
        B_train_img = train_img_gen_B(font_path, del_index)
        

        A_train_img.save('{}.png'.format(A_train_save_path + str(del_index)))
        

        B_train_img.save('{}.png'.format(B_train_save_path + str(del_index)))
            




def train_img_gen_A(train_text, font_path, index):#images of 4 characters which one of them is omitted

    chars = np.array(list(string.ascii_uppercase))


    font = ImageFont.truetype(font = font_path, size = 512)
    out_train = Image.new('RGB', (64*26,64))

    for i, char in enumerate(chars):
        x, y = font.getsize(char)
        m = max(x,y)
        pad = int(m/50)
        new_s = m+pad
        strip_width, strip_height = new_s, new_s

        train_img = Image.new('RGB', (new_s, new_s), color='white')
    
        train_DrawPad = ImageDraw.Draw(train_img)

        
        if char in train_text:
            text_width, text_height = train_DrawPad.textsize(char, font)
            position = ((strip_width-text_width)/2,(strip_height-text_height)/2)
            train_DrawPad.text(position, char, font=font, fill='black')
        else:
            train_DrawPad.text((0.0, 0.0), " ", font=font, fill='black')
    
        train_img = train_img.resize((64,64),Image.ANTIALIAS)
        train_img = augmentation(train_img)

        out_train.paste(train_img, (i*64,0))
    return out_train

def test_img_gen_A(full_text, font_path):#images of all five characters
    chars = np.array(list(string.ascii_uppercase))
   
    font = ImageFont.truetype(font = font_path, size = 512)
    
    out_test =  Image.new('RGB', (64*26,64))
    
    for i, char in enumerate(chars):
        x, y = font.getsize(char)
        m = max(x,y)
        pad = int(m/50)
        new_s = m+pad
        strip_width, strip_height = new_s, new_s


        test_img = Image.new('RGB', (new_s, new_s), color='white')

    
        test_DrawPad = ImageDraw.Draw(test_img)

        if char in full_text:
            text_width, text_height = test_DrawPad.textsize(char, font)
            position = ((strip_width-text_width)/2,(strip_height-text_height)/2)
            test_DrawPad.text(position, char, font=font, fill='black')
        else:
            test_DrawPad.text((0.0, 0.0), " ", font=font, fill='black')

    
        test_img = test_img.resize((64,64),Image.ANTIALIAS)

        out_test.paste(test_img, (i*64,0))
    return out_test

def test_img_gen_B(font_path):#images of all 26 characters
    chars = np.array(list(string.ascii_uppercase))
    font = ImageFont.truetype(font = font_path, size = 512)
    
    out_test =  Image.new('RGB', (64*26,64))
    
    for i, char in enumerate(chars):
       
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
    return out_test

def train_img_gen_B(font_path, index):#image of single characters which is omiited

    chars = np.array(list(string.ascii_uppercase))
    font = ImageFont.truetype(font = font_path, size = 512)
    
    x, y = font.getsize(chars[index])
    m = max(x,y)
    pad = int(m/50)
    new_s = m+pad
    strip_width, strip_height = new_s, new_s

    train_img = Image.new('RGB', (new_s, new_s), color='white')
    train_DrawPad = ImageDraw.Draw(train_img)

    text_width, text_height = train_DrawPad.textsize(chars[index], font)
    position = ((strip_width-text_width)/2,(strip_height-text_height)/2)

    train_DrawPad.text(position, chars[index], font=font, fill='black')
    train_img = train_img.resize((64,64),Image.ANTIALIAS)
    return train_img    

fonts_path = "/home/dhkim/data/"
dataset_path = "/home/dhkim/mc-gan-master/datasets/public_web_fonts/"

random.seed(10)
start = sys.argv[1]
end= sys.argv[2]

data_gen(fonts_path,dataset_path, start, end)