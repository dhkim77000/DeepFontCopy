import argparse
from tqdm import tqdm
from pathlib import Path
import pdb
from glob import glob
from fontTools.ttLib import TTFont
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import os

def get_defined_chars(fontfile):
    ttf = TTFont(fontfile)
    chars = [chr(y) for y in ttf["cmap"].tables[0].cmap.keys()]
    return chars


def get_filtered_chars(fontpath):
    ttf = read_font(fontpath)
    defined_chars = get_defined_chars(fontpath)
    avail_chars = []

    for char in defined_chars:

        img = render(ttf, char)

        if img == -1:  #To avoid unknown freetype error
            return -1

        img = np.array(img)
        if img.mean() == 255.:
            pass
        else:
            avail_chars.append(char.encode('utf-16', 'surrogatepass').decode('utf-16'))

    return avail_chars


def read_font(fontfile, size=400):
    font = ImageFont.truetype(str(fontfile), size=size)
    return font


def render(font, char, size=(400, 400), pad=20):
    try:
        width, height = font.getsize(char)
        max_size = max(width, height)

        if width < height:
            start_w = (height - width) // 2 + pad
            start_h = pad
        else:
            start_w = pad
            start_h = (width - height) // 2 + pad

        img = Image.new("L", (max_size+(pad*2), max_size+(pad*2)), 255)
        draw = ImageDraw.Draw(img)
        draw.text((start_w, start_h), char, font=font)
        img = img.resize(size, 2)
        return img
    except OSError as e:
        return -1

def get_chars_from_ttf(fonts_path):
    #font_path = /home/dhkim/data/
    Non_eng = False
    ttffiles = glob(fonts_path+ "*.ttf")
    ttffiles.sort()
    for ttffile in tqdm(ttffiles):
        filename = ttffile[:-4].replace("/home/dhkim/data/","")
        dirname = "/home/dhkim/data/"
        avail_chars = get_filtered_chars(ttffile)

        if avail_chars == -1: #error case
            path = ttffile
            os.remove(path)
            continue

        with open((dirname +"/" + (filename+".txt")), "w") as f:
            try:
                last_index = avail_chars.index('z') + 1
            except ValueError as e:
                path = ttffile
                os.remove(path)
                Non_eng = True

            if(Non_eng == True): 
                pdb.set_trace()
                os.remove(dirname +"/" + (filename+".txt"))    

            avail_chars = avail_chars[:last_index]
            f.write("".join(avail_chars))
get_chars_from_ttf("/home/dhkim/data/")
