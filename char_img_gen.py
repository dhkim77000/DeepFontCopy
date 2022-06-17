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
from unicode_gen import unicode_gen, sym_unicode_gen
import pdb

def char_img_gen(unicodes, fonts_path, img_path):
    #fonts_path = "/home/dhkim/font/

    font_list = glob(fonts_path+ "/*.ttf")
    font_list.sort()
    
    for i,uni in enumerate(tqdm(unicodes)):
        
        unicodeChars = chr(int(uni, 16))
            
        for ttf in font_list:
                
            font = ImageFont.truetype(font = ttf, size = 400)
            x, y = font.getsize(unicodeChars)
            theImage = Image.new('RGB', (x + 3, y + 3), color='white')
            theDrawPad = ImageDraw.Draw(theImage)
            theDrawPad.text((0.0, 0.0), unicodeChars[0], font=font, fill='black' )
            ttf_name = ttf[:-4].replace("/home/dhkim/data/","")
            msg = img_path + ttf_name+"/"
        
            if(i == 0):
                os.makedirs(msg, exist_ok = True)
            
            theImage.save('{}.png'.format(msg + unicodeChars))
            print(unicodeChars+ " "+ ttf)
        
def sym_img_gen(unicodes, fonts_path, img_path):

    font_list = glob(fonts_path+ "/*.ttf")
    font_list.sort()
    
    for i,uni in enumerate(tqdm(unicodes)):
        
        unicodeChars = str(uni)
        pdb.set_trace()
            
        for ttf in font_list:
                
            font = ImageFont.truetype(font = ttf, size = 400)
            x, y = font.getsize(unicodeChars)
            theImage = Image.new('RGB', (x + 3, y + 3), color='white')
            theDrawPad = ImageDraw.Draw(theImage)
            theDrawPad.text((0.0, 0.0), unicodeChars[0], font=font, fill='black' )
            ttf_name = ttf[:-4].replace("/home/dhkim/data/","")
            msg = img_path + ttf_name+"/"
        
            if(i == 0):
                os.makedirs(msg, exist_ok = True)
            
            theImage.save('{}.png'.format(msg + unicodeChars))
            print(unicodeChars+ " "+ ttf)
        


fonts_path = "/home/dhkim/data/"
img_path = "/home/dhkim/data/img/"
unicodes = unicode_gen()
sym_unicodes = sym_unicode_gen()

#char_img_gen(unicodes, fonts_path, img_path)
sym_img_gen(sym_unicodes, fonts_path, img_path)

