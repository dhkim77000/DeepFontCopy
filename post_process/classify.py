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
class VGG(nn.Module):  
    """
    Based on - https://github.com/kkweon/mnist-competition
    from: https://github.com/ranihorev/Kuzushiji_MNIST/blob/master/KujuMNIST.ipynb
    """
    def two_conv_pool(self, in_channels, f1, f2):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s
    
    def three_conv_pool(self,in_channels, f1, f2, f3):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, f3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s
        
    
    def __init__(self, mode,num_classes=62):
        super(VGG, self).__init__()
        self.mode = mode
        self.l1 = self.two_conv_pool(1, 64, 64)
        self.l2 = self.two_conv_pool(64, 128, 128)
        self.l3 = self.three_conv_pool(128, 256, 256, 256)
        self.l4 = self.three_conv_pool(256, 256, 256, 256)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.mode == 'train':
            return F.log_softmax(x, dim=1)
        elif self.mode == 'classification':
            return F.sigmoid(x)
    



class ResNet(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1), 
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(inplace=True)) #32*28*28
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), 
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(2)) #64*14*14
        self.res1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                  nn.ReLU(inplace=True)) #64*14*14
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(128),  
                                  nn.ReLU(inplace=True)) #128*14*14
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(256), 
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(2)) #256*7*7
        self.res2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), 
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                  nn.ReLU(inplace=True)) #256*7*7
        self.classifier = nn.Sequential(nn.Flatten(),
                          nn.Linear(256*7*7, 1024),
                          nn.ReLU(),
                          nn.Linear(1024, 512),
                          nn.ReLU(),
                          nn.Linear(512, 62)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        
        if self.mode == 'train':
            return F.log_softmax(out, dim=1)
        elif self.mode == 'classification':
            return F.sigmoid(out)
        

class SpinalVGG(nn.Module):  
    def two_conv_pool(self, in_channels, f1, f2):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s
    
    def three_conv_pool(self,in_channels, f1, f2, f3):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, f3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s
        
    
    def __init__(self, mode, Half_width, layer_width,num_classes=62):
        self.mode = mode
        self.Half_width = Half_width
        self.layer_width = layer_width
        super(SpinalVGG, self).__init__()
        self.l1 = self.two_conv_pool(1, 64, 64)
        self.l2 = self.two_conv_pool(64, 128, 128)
        self.l3 = self.three_conv_pool(128, 256, 256, 256)
        self.l4 = self.three_conv_pool(256, 256, 256, 256)
        
        
        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p = 0.5), nn.Linear(self.Half_width, self.layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True),)
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p = 0.5), nn.Linear(self.Half_width+self.layer_width, self.layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True),)
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p = 0.5), nn.Linear(self.Half_width+self.layer_width, self.layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True),)
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p = 0.5), nn.Linear(self.Half_width+self.layer_width, self.layer_width),
            nn.BatchNorm1d(self.layer_width), nn.ReLU(inplace=True),)
        self.fc_out = nn.Sequential(
            nn.Dropout(p = 0.5), nn.Linear(self.layer_width*4, num_classes),)
        
    
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = x.view(x.size(0), -1)
        
        x1 = self.fc_spinal_layer1(x[:, 0:self.Half_width])
        x2 = self.fc_spinal_layer2(torch.cat([ x[:,self.Half_width:2*self.Half_width], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([ x[:,0:self.Half_width], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([ x[:,self.Half_width:2*self.Half_width], x3], dim=1))
        
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)
        
        x = self.fc_out(x)

        if self.mode == 'train':
            return F.log_softmax(x, dim=1)
        elif self.mode == 'classification':
            return F.sigmoid(x)

class CustomDataset(Dataset):
    def __init__(self, image_dir):  
        self.dir_path = image_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])

        self.image_arr  = glob.glob(os.path.join(image_dir, '*.png')) 

        
    def __getitem__(self, index):
      
        single_image_path = self.image_arr[index]

        file_format = '.png'
        pattern = "(?<=\/)\w+" + file_format
        p = re.compile(pattern)

        contour_index = int(p.findall(single_image_path)[0].replace(file_format,""))
        img_as_img = Image.open(single_image_path).convert("L")
        img_as_img = img_as_img.resize((28,28))
        img_as_tensor = self.transform(img_as_img)

        return (contour_index,img_as_tensor)

    def __len__(self):
        """
        Returns:
            length (int): length of the data
        """
        return len(self.image_arr)


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

    Half_width =128
    layer_width =128

    model1 = VGG('classification')
    model2 = SpinalVGG('classification', Half_width, layer_width)
    model3 = ResNet('classification')

    device = 'cuda'

    #load model
    model1.load_state_dict(torch.load('/home/dhkim/character_classification/model/VGG.pth'))
    model2.load_state_dict(torch.load('/home/dhkim/character_classification/model/spiral_VGG.pth'))
    model3.load_state_dict(torch.load('/home/dhkim/character_classification/model/ResNet.pth'))


    model1.to(device)
    model2.to(device)
    model3.to(device)
 
    customdata = CustomDataset('/home/dhkim/post_process/data/'+ image_name)
    image_loader = torch.utils.data.DataLoader(dataset=customdata,
                                            batch_size=1,
                                            shuffle=False)
    model1.eval()
    model3.eval()
    model2.eval()
    image_dict = {}
    with torch.no_grad():

    
        for i,(contour_index, image) in enumerate(image_loader):
            image = image.to(device)
            voter1 = model1(image)
            voter3 = model3(image)
            voter2 = model3(image)

            prediction = vote(voter1, voter2, voter3)
            #pdb.set_trace()
            contour_index = str(int(contour_index[0].data.cpu()))

            #if prediction not in list("0123456789"):
            image_dict[contour_index] = prediction
       

    
    save_path = "/home/dhkim/post_process/data_classified/"
    threshold = 100

    image_file = '/home/dhkim/post_process/image/'


    imgset_path = '/home/dhkim/post_process/data_classified/' + image_name + "/"

    image_file = image_file + sys.argv[1]
    print(image_dict)

    num = int(input("If you have some unwanted result, how many you want to delete it?"))
    if num != 0:
        for i in range(num):
            index = input("Give the index of the contour you want to delete")
            #os.remove(imgset_path+image_dict[index]+".png")
            del image_dict[index]
    
    contours, _,_,_ = classification_data_gen(save_path, image_file, 'out', image_dict, threshold)
    
    