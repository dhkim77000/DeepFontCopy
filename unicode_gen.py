from tqdm import tqdm
import numpy as np
import os
import pdb
def alpha_unicode_gen(txtfile): #generate unicode for numbers and alphabets

   co = "0 1 2 3 4 5 6 7 8 9 A B C D E F"

   num_start = "0030"
   num_end = "0039"
   

   co = co.split(" ")

   num = [a+b+c+d 
                        for a in co 
                        for b in co 
                        for c in co 
                        for d in co]

   num = np.array(num)
   s = np.where(num_start == num)[0][0]
   e = np.where(num_end == num)[0][0]
   num = num[s : e + 1]

   cap_start = "0041"
   cap_end = "005A"
   co = "0 1 2 3 4 5 6 7 8 9 A B C D E F"
   co = co.split(" ")

   cap = [a+b+c+d 
                        for a in co 
                        for b in co 
                        for c in co 
                        for d in co]
   cap = np.array(cap)
   s = np.where(cap_start == cap)[0][0]
   e = np.where(cap_end == cap)[0][0]
   cap = cap[s : e + 1]

   sm_start = "0061"
   sm_end = "007A"
   co = "0 1 2 3 4 5 6 7 8 9 A B C D E F"
   co = co.split(" ")

   sm = [a+b+c+d 
                        for a in co 
                        for b in co 
                        for c in co 
                        for d in co]

   sm = np.array(sm)
   s = np.where(sm_start == sm)[0][0]
   e = np.where(sm_end == sm)[0][0]

   sm = sm[s : e + 1]

   return np.concatenate((num, cap, sm), axis=None)

def sym_unicode_gen():
   co = "0 1 2 3 4 5 6 7 8 9 A B C D E F"
   start = "0021"
   end = "007A"
   co = co.split(" ")

   num = [a+b+c+d 
                        for a in co 
                        for b in co 
                        for c in co 
                        for d in co]

   num = np.array(num)
   s = np.where(start == num)[0][0]
   e = np.where(end == num)[0][0]
   num = num[s : e + 1]
   return num
   
def unicode_gen(textfile):
   try:
      txt = open(textfile, "r", encoding = "utf8")
   except FileNotFoundError as e:
      return -1
   characters = txt.read().split(" ")
   characters = list(characters[0])
   unicode_list = []
   for avail_chars in characters:
      unicode_list.append(hex(ord(avail_chars)).replace("x","0"))
   return unicode_list