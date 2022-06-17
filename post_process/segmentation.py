import cv2 
import sys
import os
from torch import index_select,LongTensor
from PIL import Image
import os
import os.path
import numpy as np
from scipy import misc
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torchvision.transforms as transforms
import string
from torch import index_select,LongTensor
import re
import pdb

#------------------Functions------------------#

def toPIL(src_img, bin_img, final_thr):
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    src_img = Image.fromarray(src_img)
    
    bin_img = cv2.cvtColor(bin_img, cv2.COLOR_BGR2RGB)
    bin_img = Image.fromarray(bin_img)
    
    final_thr = cv2.cvtColor(final_thr, cv2.COLOR_BGR2RGB)
    final_thr = Image.fromarray(final_thr)
    
    return src_img, bin_img, final_thr

def closewindows():
	k = cv2.waitKey(0)
	if k & 0xFF == ord('s'):
		comment = input("Comment:-\n ")
		cv2.imwrite('./data/test_result/'+comment+'_thres'+'.jpg',final_thr)
		cv2.imwrite('./data/test_result/'+comment+'_src'+'.jpg',src_img)
		cv2.imwrite('./data/test_result/'+comment+'_contr'+'.jpg',final_contr)
		print("Completed")
	elif k & 0xFF == int(27):
		cv2.destroyAllWindows()
	else:
		closewindows()

def line_array(array):
	list_x_upper = []
	list_x_lower = []
	for y in range(5, len(array)-5):
		s_a, s_p = strtline(y, array)
		e_a, e_p = endline(y, array)
		if s_a>=7 and s_p>=5:
			list_x_upper.append(y)
			# bin_img[y][:] = 255
		if e_a>=5 and e_p>=7:
			list_x_lower.append(y)
			# bin_img[y][:] = 255
	return list_x_upper, list_x_lower

def strtline(y, array):
	count_ahead = 0
	count_prev = 0
	for i in array[y:y+10]:
		if i > 3:
			count_ahead+= 1  
	for i in array[y-10:y]:
		if i==0:
			count_prev += 1  
	return count_ahead, count_prev

def endline(y, array):
	count_ahead = 0
	count_prev = 0
	for i in array[y:y+10]:
		if i==0:
			count_ahead+= 1  
	for i in array[y-10:y]:
		if i >3:
			count_prev += 1  
	return count_ahead, count_prev

def endline_word(y, array, a):
	count_ahead = 0
	count_prev = 0
	for i in array[y:y+2*a]:
		if i < 2:
			count_ahead+= 1  
	for i in array[y-a:y]:
		if i > 2:
			count_prev += 1  
	return count_prev ,count_ahead

def end_line_array(array, a):
	list_endlines = []
	for y in range(len(array)):
		e_p, e_a = endline_word(y, array, a)
		# print(e_p, e_a)
		if e_a >= int(1.5*a) and e_p >= int(0.7*a):
			list_endlines.append(y)
	return list_endlines

def refine_endword(array):
	refine_list = []
	for y in range(len(array)-1):
		if array[y]+1 < array[y+1]:
			refine_list.append(array[y])
	#refine_list.append(array[-1])
	return refine_list

def refine_array(array_upper, array_lower):
	upperlines = []
	lowerlines = []
	for y in range(len(array_upper)-1):
		if array_upper[y] + 5 < array_upper[y+1]:
			upperlines.append(array_upper[y]-10)
	for y in range(len(array_lower)-1):
		if array_lower[y] + 5 < array_lower[y+1]:
			lowerlines.append(array_lower[y]+10)

	upperlines.append(array_upper[-1]-10)
	lowerlines.append(array_lower[-1]+10)
	
	return upperlines, lowerlines

def letter_width(contours):
	letter_width_sum = 0
	count = 0
	for cnt in contours:
		if cv2.contourArea(cnt) > 10:
			x,y,w,h = cv2.boundingRect(cnt)
			letter_width_sum += w
			count += 1

	return letter_width_sum/count

def end_wrd_dtct(lines, i, bin_img, mean_lttr_width):
	count_y = np.zeros(shape = width)
	for x in range(width):
		for y in range(lines[i][0],lines[i][1]):
			if bin_img[y][x] == 255:
				count_y[x] += 1
	end_lines = end_line_array(count_y, int(mean_lttr_width))
	# print(end_lines)
	endlines = refine_endword(end_lines)
	for x in endlines:
		final_thr[lines[i][0]:lines[i][1], x] = 255
	return endlines

def letter_seg(lines_img, x_lines, i):
	copy_img = lines_img[i].copy()
	x_linescopy = x_lines[i].copy()
	
	letter_img = []
	letter_k = []

	contours, hierarchy = cv2.findContours(copy_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)	
	j=0
	for cnt in contours:
		if cv2.contourArea(cnt) > 50:
			x,y,w,h = cv2.boundingRect(cnt)
			# letter_img.append(lines_img[i][y:y+h, x:x+w])
			letter_k.append((x,y,w,h));#cv2.imwrite('selectedHey'+str(j)+'.JPG',lines_img[i][y:y+h,x:x+w]);j=j+1;

	letter = sorted(letter_k, key=lambda student: student[0])
	print(letter)
	
	word = 1
	letter_index = 0
	for e in range(len(letter)):
		if(letter[e][0]<x_linescopy[0]):
			letter_index += 1
			letter_img_tmp = lines_img[i][letter[e][1]-5:letter[e][1]+letter[e][3]+5,letter[e][0]-5:letter[e][0]+letter[e][2]+5]
			letter_img = cv2.resize(letter_img_tmp, dsize =(28, 28), interpolation = cv2.INTER_AREA)
			cv2.imwrite('seg'+str(i+1)+'_'+str(word)+'_'+str(letter_index)+'.jpg', 255-letter_img)
		else:
			x_linescopy.pop(0)
			word += 1
			letter_index = 1
			letter_img_tmp = lines_img[i][letter[e][1]-5:letter[e][1]+letter[e][3]+5,letter[e][0]-5:letter[e][0]+letter[e][2]+5]
			letter_img = cv2.resize(letter_img_tmp, dsize =(28, 28), interpolation = cv2.INTER_AREA)
			cv2.imwrite("seg"+str(i+1)+'_'+str(word)+'_'+str(letter_index)+'.jpg', 255-letter_img)


def segment_img_saver_Train(save_path, contours, image_file):
	image = cv2.imread(image_file, 1)
	copy = image.copy()
	#Resizing
	height = image.shape[0]
	width = image.shape[1]
	image = cv2.resize(copy, dsize =(1320, int(1320*height/width)), interpolation = cv2.INTER_AREA)
	image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,21,20)

	image =Image.fromarray(image)

	p = re.compile("(?<=\.).*")
	file_format = p.findall(image_file)[0]

	pattern = "(?<=\/)\w+." + file_format
	p = re.compile(pattern)
	file_name = p.findall(image_file)[0].replace("."+file_format,"")

	save_path = save_path + file_name + "/"
	os.makedirs(save_path, exist_ok = True)
	#make BW Image
	#fn = lambda x : 0 if x > 64 else 255
	#image = image.convert('L').point(fn, mode='1')

	for i,cnt in enumerate(contours):

		if cv2.contourArea(cnt) > 576:
			try:
				rec = cv2.boundingRect(cnt)
				rec = list(rec)
				rec[2] += rec[0]
				rec[3]+= rec[1]
				rec = tuple(rec)
				
				width = rec[2]-rec[0]
				height = rec[3]-rec[1]
				ratio = width/height
				
				if ratio < 1.5:
					
					crop = image.crop(rec)

					size = max(rec[2]-rec[0], rec[3]-rec[1])
					size += int(size/2)

					img = Image.new('RGB', (size, size), color='black')

					strip_width, strip_height = size, size


					text_width = rec[2] - rec[0]
					text_height = rec[3] - rec[1]
					
					
					x = int((strip_width-text_width)/2)
					y = int((strip_height-text_height)/2)
					position = (x,y)

					img.paste(crop, position)
					img.save(save_path + "/{}.png".format(str(i)))
			except SystemError as e:
				pass



def rgb_threshold(image, rec):
    width = rec[2]-rec[0]
    height = rec[3]-rec[1]

    point1 = (rec[0]+0.2, rec[1]+0.2)
    point2 = (rec[0]+0.2, rec[3]-0.2)
    point3 = (rec[2]-0.2, rec[3]-0.2)
    point4 = (rec[2]-0.2, rec[1]+0.2)

    rgb1 = image.getpixel(point1)
    rgb2 = image.getpixel(point1)
    rgb3 = image.getpixel(point1)
    rgb4 = image.getpixel(point1)
    rgb_list = (rgb1, rgb2, rgb3, rgb4)
    avg_rgb = [sum(y) / len(y) for y in zip(*rgb_list)]

    R = avg_rgb[0]
    G = avg_rgb[1]
    B = avg_rgb[2]
    
    R_max = R + R/2
    if R_max > 255: R_max = 255
    R_min = R - R/4
    if R_min < 0: R_min = 0
    
    G_max = G + R/2
    if G_max > 255: G_max = 255
    G_min = G - G/4
    if G_min < 0: G_min = 0
        
    B_max = B + B/2
    if B_max > 255: B_max = 255
    B_min = B - B/4
    if B_min < 0: B_min = 0
        
        
    return (R_min, G_min, B_min), (R_max, G_max, B_max) 


def segment_img_saver_Out(save_path, contours, image_file, dict_file, threshold):
	
	image = cv2.imread(image_file, 1)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	copy = image.copy()
	height = image.shape[0]
	width = image.shape[1]
	image = cv2.resize(copy, dsize =(1320, int(1320*height/width)), interpolation = cv2.INTER_AREA)
	color_image =Image.fromarray(image)

	p = re.compile("(?<=\.).*")
	file_format = p.findall(image_file)[0]

	pattern = "(?<=\/)\w+." + file_format
	p = re.compile(pattern)

	file_name = p.findall(image_file)[0].replace("."+file_format,"")

	save_path = save_path + file_name + "/"
	os.makedirs(save_path, exist_ok = True)

	fn = lambda x : 0 if x < threshold else 255
	binary_image = color_image.point(fn)


	for i,cnt in enumerate(contours):

		label = dict_file.get(str(i)) 
		if(label != None):
			try:
				rec = cv2.boundingRect(cnt)
				rec = list(rec)
				rec[2] += rec[0]
				rec[3]+= rec[1]
				rec = tuple(rec)
				
				width = rec[2]-rec[0]
				height = rec[3]-rec[1]
				ratio = width/height
				
				if ratio < 1.5:
					
					crop = color_image.crop(rec)
					crop_array = np.array(crop)
					min_rbg, max_rgb = rgb_threshold(color_image,rec)
				
					mask = cv2.inRange(crop_array,min_rbg,max_rgb)
					imask = mask>0
					imask = np.invert(imask)
					background_removed = np.zeros_like(crop_array, np.uint8)
					background_removed = np.invert(background_removed)
					background_removed[imask] = crop_array[imask]

					## save 
					crop = Image.fromarray(background_removed)
				
					size = max(rec[2]-rec[0], rec[3]-rec[1])
					size += int(size/15)

					img = Image.new('RGB', (size, size), color='white')

					strip_width, strip_height = size, size


					text_width = rec[2] - rec[0]
					text_height = rec[3] - rec[1]
					
					
					x = int((strip_width-text_width)/2)
					y = int((strip_height-text_height)/2)
					position = (x,y)

					img.paste(crop, position)
					img.resize((64,64))
					img.save(save_path + "/{}.png".format(label))
			except SystemError as e:
				pass
	#------------------/Functions-----------------#


#-------------Thresholding Image--------------#
def classification_data_gen(save_path, image_file, mode, dictfile, threshold):

	print("\n........Program Initiated.......\n")
	src_img= cv2.imread(image_file, 1)

	copy = src_img.copy()
	height = src_img.shape[0]
	width = src_img.shape[1]

	print("\n Resizing Image........")
	src_img = cv2.resize(copy, dsize =(1320, int(1320*height/width)), interpolation = cv2.INTER_AREA)

	height = src_img.shape[0]
	width = src_img.shape[1]

	print("#---------Image Info:--------#")
	print("\tHeight =",height,"\n\tWidth =",width)
	print("#----------------------------#")

	grey_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

	print("Applying Adaptive Threshold with kernel :- 21 X 21")
	bin_img = cv2.adaptiveThreshold(grey_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,21,20)
	bin_img1 = bin_img.copy()
	bin_img2 = bin_img.copy()

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	kernel1 = np.array([[1,0,1],[0,1,0],[1,0,1]], dtype = np.uint8)
	# final_thr = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
	# final_thr = cv2.dilate(bin_img,kernel1,iterations = 1)
	print("Noise Removal From Image.........")
	final_thr = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
	contr_retrival = final_thr.copy()

	#-------------/Thresholding Image-------------#

	#-------------Letter Width Calculation--------#

	contours, hierarchy = cv2.findContours(contr_retrival,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	final_contr = np.zeros((final_thr.shape[0],final_thr.shape[1],3), dtype = np.uint8)

	min_area = 100
	image_number = 0
	

	for c in contours:
		area = cv2.contourArea(c)
		if area > min_area:
			x,y,w,h = cv2.boundingRect(c)
			cv2.rectangle(src_img, (x, y), (x + w, y + h), (36,255,12), 2)
			image_number += 1

	src_image, binary_image ,final_thr = toPIL(src_img, bin_img, final_thr)

	

	if mode == "train":
		print("#---------Image is being segmented:--------#")
		segment_img_saver_Train(save_path, contours, image_file)
		
	if mode == 'out':
		print("#---------Image is being segmented:--------#")
		segment_img_saver_Out(save_path, contours, image_file, dictfile, threshold)

	print("#---------Done:--------#")
	return contours, src_image, binary_image, final_thr


if __name__ == "__main__":
	save_path = '/home/dhkim/post_process/data/'
	image_file = '/home/dhkim/post_process/image/'
	file_name = sys.argv[1]

	image_file = image_file + file_name

	classification_data_gen(save_path, image_file, 'train', dictfile = None, threshold = None)