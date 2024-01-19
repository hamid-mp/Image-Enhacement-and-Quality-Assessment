import numpy as np
import cv2 as cv
import os
from PIL import Image, ImageEnhance

def Cubic(img,a1,a2,a3,a4):
  return a1*img**3+ a2*img**2 + a3*img + a4

def logistic_func(img, b1, b2, b3, b4):
  out = ( (b1 - b2) / (1+np.exp(-((img-b3)/(b4)))) ) + b2
  return out

path = 'G:\DATASETS\my_synthetic_dataset\org'
images = os.listdir(path)
x = 0
Logistic_weights = [[275.0706,-20.0706,127.5000,48.7056],[288.7914,-33.7914,127.5000,59.4267],[268.8418,-13.8418,127.5000,42.9810],[266.3276,-11.3276,127.5000,40.3802]]
Cubic_weights = [[0,-0.0094,1.8028,0],[0,-0.0148,2.2549,0],[0.0001,-0.0199,2.6941,0],[0.0001,-0.0233,2.9828,0]]
path1 = 'G:\DATASETS\my_synthetic_dataset\dis'
path2 = 'G:\DATASETS\my_synthetic_dataset\org_resize'

def adjust_gamma(image, gamma=1.0):

    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    table = np.array([((i / 255.0) ** gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv.LUT(image, table)
x = 0
for i in range(len(images)):
    name = images[i].replace('.bmp','')
    img = cv.imread(os.path.join(path, images[i]))
    img = cv.resize(img,(224,224))
    cv.imwrite(os.path.join(path2,"{0:05d}_org.png").format(int(name)), img)


    L1 = logistic_func(img,*Logistic_weights[0])
    L2 = logistic_func(img,*Logistic_weights[1])
    L3 = logistic_func(img,*Logistic_weights[2])
    L4 = logistic_func(img,*Logistic_weights[3])

    C1 = Cubic(img,*Cubic_weights[0])
    C2 = Cubic(img,*Cubic_weights[1])
    C3 = Cubic(img,*Cubic_weights[2])
    C4 = Cubic(img,*Cubic_weights[3])

    shift_20 = (np.ones(img.shape)*20).astype('uint8')
    bright_20 = cv.subtract(img, shift_20)
    dark_20 = cv.add(img ,shift_20)
    shift_40 = (np.ones(img.shape)*40).astype('uint8')
    bright_40 = cv.subtract(img, shift_40)
    dark_40 = cv.add(img, shift_40)
    shift_60 = (np.ones(img.shape)*60).astype('uint8')
    bright_60 = cv.subtract(img, shift_60)
    dark_60 = cv.add(img, shift_60)
    shift_80 = (np.ones(img.shape)*80).astype('uint8')
    bright_80 = cv.subtract(img, shift_80)
    dark_80 = cv.add(img, shift_80)
    shift_100 = (np.ones(img.shape)*100).astype('uint8')
    bright_100 = cv.subtract(img,shift_100)
    dark_100 = cv.add(img,shift_100)
    shift_120 = (np.ones(img.shape) * 120).astype('uint8')
    bright_120 = cv.subtract(img, shift_120)
    dark_120 = cv.add(img, shift_120)

    img00 = adjust_gamma(img,gamma=1/5)
    img0 = adjust_gamma(img,gamma=1/3)
    img1 = adjust_gamma(img,gamma=1/2)
    img2 = adjust_gamma(img,gamma=1/1.5)
    img3 = adjust_gamma(img,gamma=1.5)
    img4 = adjust_gamma(img,gamma=2)
    img5 = adjust_gamma(img,gamma=3)
    img6 = adjust_gamma(img,gamma=5)


    img = Image.open(os.path.join(path, images[i]))
    img = img.resize((224,224))
    obj = ImageEnhance.Contrast(img)
    lvl1 = obj.enhance(0.9)
    lvl1 = lvl1.convert('RGB')
    lvl2 = obj.enhance(1.18)
    lvl2 = lvl2.convert('RGB')
    lvl3 = obj.enhance(0.75)
    lvl3 = lvl3.convert('RGB')
    lvl4 = obj.enhance(1.41)
    lvl4 = lvl4.convert('RGB')
    lvl5 = obj.enhance(0.52)
    lvl5 = lvl5.convert('RGB')
    cv.imwrite(os.path.join(path1, '{0:05d}_Logistic_1.png').format(int(name)), L1)
    cv.imwrite(os.path.join(path1, '{0:05d}_Logistic_2.png').format(int(name)), L2)
    cv.imwrite(os.path.join(path1, '{0:05d}_Logistic_3.png').format(int(name)), L3)
    cv.imwrite(os.path.join(path1, '{0:05d}_Logistic_4.png').format(int(name)), L4)
    cv.imwrite(os.path.join(path1, '{0:05d}_Cubic_1.png').format(int(name)), C1)
    cv.imwrite(os.path.join(path1, '{0:05d}_Cubic_2.png').format(int(name)), C2)
    cv.imwrite(os.path.join(path1, '{0:05d}_Cubic_3.png').format(int(name)), C3)
    cv.imwrite(os.path.join(path1, '{0:05d}_Cubic_4.png').format(int(name)), C4)
    cv.imwrite(os.path.join(path1,'{0:05d}_MeanShift_1.png').format(int(name)), bright_20)
    cv.imwrite(os.path.join(path1,'{0:05d}_MeanShift_2.png').format(int(name)), bright_40)
    cv.imwrite(os.path.join(path1,'{0:05d}_MeanShift_3.png').format(int(name)), bright_60)
    cv.imwrite(os.path.join(path1,'{0:05d}_MeanShift_4.png').format(int(name)), bright_80)
    cv.imwrite(os.path.join(path1,'{0:05d}_MeanShift_5.png').format(int(name)), bright_100)
    cv.imwrite(os.path.join(path1,'{0:05d}_MeanShift_6.png').format(int(name)), bright_120)
    cv.imwrite(os.path.join(path1,'{0:05d}_MeanShift_7.png').format(int(name)), dark_20)
    cv.imwrite(os.path.join(path1,'{0:05d}_MeanShift_8.png').format(int(name)), dark_40)
    cv.imwrite(os.path.join(path1,'{0:05d}_MeanShift_9.png').format(int(name)), dark_60)
    cv.imwrite(os.path.join(path1,'{0:05d}_MeanShift_10.png').format(int(name)), dark_80)
    cv.imwrite(os.path.join(path1,'{0:05d}_MeanShift_11.png').format(int(name)), dark_100)
    cv.imwrite(os.path.join(path1,'{0:05d}_MeanShift_12.png').format(int(name)), dark_120)
    cv.imwrite(os.path.join(path1,'{0:05d}_Gamma_1.png').format(int(name)), img00)
    cv.imwrite(os.path.join(path1,'{0:05d}_Gamma_2.png').format(int(name)), img0)
    cv.imwrite(os.path.join(path1,'{0:05d}_Gamma_3.png').format(int(name)), img1)
    cv.imwrite(os.path.join(path1,'{0:05d}_Gamma_4.png').format(int(name)), img2)
    cv.imwrite(os.path.join(path1,'{0:05d}_Gamma_5.png').format(int(name)), img3)
    cv.imwrite(os.path.join(path1,'{0:05d}_Gamma_6.png').format(int(name)), img4)
    cv.imwrite(os.path.join(path1,'{0:05d}_Gamma_7.png').format(int(name)), img5)
    cv.imwrite(os.path.join(path1,'{0:05d}_Gamma_8.png').format(int(name)), img6)

    lvl1.save(os.path.join(path1,"{0:05d}_ContrastChange_1.png").format(int(name)),format='png')
    lvl2.save(os.path.join(path1,"{0:05d}_ContrastChange_2.png").format(int(name)),format='png')
    lvl3.save(os.path.join(path1,"{0:05d}_ContrastChange_3.png").format(int(name)),format='png')
    lvl4.save(os.path.join(path1,"{0:05d}_ContrastChange_4.png").format(int(name)),format='png')
    lvl5.save(os.path.join(path1,"{0:05d}_ContrastChange_5.png").format(int(name)),format='png')
