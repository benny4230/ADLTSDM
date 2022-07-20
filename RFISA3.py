from PIL import Image
import os
import random
from matplotlib.patches import Ellipse, Circle
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float
from PIL import Image
from PIL import ImageDraw
 #reference:https://blog.csdn.net/qq_37902216/article/details/93632178

def copy_img_random(CopyImgPath,ChangedImgPath,SavedPath):
    copyimgs = os.listdir(CopyImgPath)
    imgNum = len(copyimgs)
    print(imgNum)
    changedimgs=os.listdir(ChangedImgPath)
    imgNum2 = len(changedimgs)

    for i in range(imgNum):
        img1 = Image.open(CopyImgPath + copyimgs[i])
        resizeX = 102
        resizeY = 88
        img = img1.resize((resizeX,resizeY))   #resize image to propoer size
        #box = (0,0,50,50)
        #img = img1.crop(box)
        for j in range(imgNum2):
            # oriImg.paste(img, (image[0]-102, image[1]-102))
            count = 0
            while count < 100:
                oriImg = Image.open(ChangedImgPath + changedimgs[j])#open image need to be changed
                image = oriImg.size   # get the size of the image
                if image[0]<image[1]:   # judge the height and width, incase the copy image over the size
                    #oriImg.paste(img,(random.randint(0,image[0]-resizeX),random.randint(0,image[0]-resizeY)))
                    #oriImg.paste(img,(random.randint(300,2000),random.randint(300,1600)))  #2500*2000

                    n = range(1,10,1)
                    #height = range()
                    line = random.choice(n)
                    # ji shu hang
                    if line % 2 != 0:
                    #y1 = random.choice(height)
                        y1 = 55 + 90*line
                        width = range(75,1507,102)
                        x1 = random.choice(width)
                        oriImg.paste(img,(x1,y1)) #1024*1024
                    # ou shu hang
                    else:
                        y1 = 55 + 90*line
                        width = range(118,1463,102)
                        x1 = random.choice(width)
                        oriImg.paste(img,(x1,y1))
                else:
                    n = range(1,10,1)
                    #height = range()
                    line = random.choice(n)
                    # jishu hang
                    if line % 2 != 0:
                    #y1 = random.choice(height)
                        y1 = 55 + 90*line
                        width = range(75,1507,102)
                        x1 = random.choice(width)
                        oriImg.paste(img,(x1,y1)) #1024*1024
                    # ou shu hang
                    else:
                        y1 = 55 + 90*line
                        width = range(118,1463,102)
                        x1 = random.choice(width)
                        oriImg.paste(img,(x1,y1))
                #oriImg.show()
                oriImg1 = oriImg.convert('RGB')
                oriImg1.save(SavedPath + "valleyB2_" + str(count)+ ".jpg")
                count = count +1


CopyImgPath = ""
#imgFlodName = "try"
ChangedImgPath = ""
SavedPath = ""
#copy_img_random(CopyImgPath,ChangedImgPath,SavedPath)

def copy_img_random_multipletime(CopyImgPath,ChangedImgPath,SavedPath):
    copyimgs = os.listdir(CopyImgPath)
    imgNum = len(copyimgs)
    print(imgNum)
    changedimgs=os.listdir(ChangedImgPath)
    imgNum2 = len(changedimgs)

    for i in range(imgNum):
        img1 = Image.open(CopyImgPath + copyimgs[i])
        resizeX = 102
        resizeY = 88
        img = img1.resize((resizeX,resizeY))   #resize image to propoer size
        #box = (0,0,50,50)
        #img = img1.crop(box)
        for j in range(imgNum2):
            # oriImg.paste(img, (image[0]-102, image[1]-102))
            count = 0
            while count < 10:
                oriImg = Image.open("")#open image need to be changed
                image = oriImg.size   # get the size of the image
                if image[0]<image[1]:   # judge the height and width, incase the copy image over the size
                    #oriImg.paste(img,(random.randint(0,image[0]-resizeX),random.randint(0,image[0]-resizeY)))
                    #oriImg.paste(img,(random.randint(300,2000),random.randint(300,1600)))  #2500*2000

                    n = range(1,10,1)
                    #height = range()
                    line = random.choice(n)
                    # ji shu hang
                    if line % 2 != 0:
                    #y1 = random.choice(height)
                        y1 = 55 + 90*line
                        width = range(75,1507,102)
                        x1 = random.choice(width)
                        oriImg.paste(img,(x1,y1)) #1024*1024
                    # ou shu hang
                    else:
                        y1 = 55 + 90*line
                        width = range(118,1463,102)
                        x1 = random.choice(width)
                        oriImg.paste(img,(x1,y1))
                else:
                    n = range(1,10,1)
                    #height = range()
                    line = random.choice(n)
                    # jishu hang
                    if line % 2 != 0:
                    #y1 = random.choice(height)
                        y1 = 55 + 90*line
                        width = range(75,1507,102)
                        x1 = random.choice(width)
                        oriImg.paste(img,(x1,y1)) #1024*1024
                    # ou shu hang
                    else:
                        y1 = 55 + 90*line
                        width = range(118,1463,102)
                        x1 = random.choice(width)
                        oriImg.paste(img,(x1,y1))
                #oriImg.show()
                oriImg1 = oriImg.convert('RGB')
                oriImg1.save("")
                count = count +1


CopyImgPath = ""
#imgFlodName = "try"
ChangedImgPath = ""
SavedPath = ""
copy_img_random_multipletime(CopyImgPath,ChangedImgPath,SavedPath)
