import numpy as np
import cv2
from numpy.ma.core import where 
from scipy import signal



def matlab_style_gauss2D(shape=(11,11),sigma=1.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h






def PCQI (img1, img2, L=256.):

    if img1.shape != img2.shape:
        mpcqi = - np.inf
        pcqi_map = -np.inf


    if len(img1.shape) == 3:
        M, N, C = img1.shape
    else:
        M, N = img1.shape


    if M < 11 or N < 11:
        mpcqi = - np.inf
        pcqi_map = -np.inf
    

    window = matlab_style_gauss2D(shape=(11,11), sigma=1.5)

    window = window / np.sum(window)

    mul = signal.convolve2d(window, img1, mode='valid')
    mul2 = signal.convolve2d(window, img2, mode='valid')

    mul_sq = np.multiply(mul, mul)
    mul2_sq = np.multiply(mul2, mul2)
    mul_mul2 = np.multiply(mul, mul2)
    sigma1_sq = signal.convolve2d(window, np.multiply(img1/100., img1/100.)* 1e4, mode='valid') - mul_sq
    sigma2_sq = signal.convolve2d(window, np.multiply(img2/100., img2/100.)* 1e4, mode='valid') - mul2_sq
    sigma12 = signal.convolve2d(window, np.multiply(img1/100., img2/100.)* 1e4, mode='valid') - mul_mul2
  
    sigma1_sq[sigma1_sq<0] = 0
    sigma2_sq[sigma2_sq<0] = 0


    C = 3

    pcqi_map = (4. / np.pi) * np.arctan(np.divide(sigma12 + C, sigma1_sq + C))


    pcqi_map = np.multiply(pcqi_map, np.divide(sigma12 + C , np.multiply(np.sqrt(sigma1_sq)/100.0,np.sqrt(sigma2_sq)/100.0 )*1e4 + C ))

    pcqi_map = np.multiply(pcqi_map, np.exp(-np.abs(mul-mul2)/L))

    mpcqi = np.mean(pcqi_map)
    
    
    
    return mpcqi , pcqi_map


img = cv2.imread('H:\\Python Codes\\DIP\\PCQI\\ref.png')
img1 = cv2.imread('H:\\Python Codes\\DIP\\PCQI\\contrast_changed.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
score, p_map = PCQI(img, img1)
print(f'The PCQI score is {score}')
cv2.imshow('PCQI_map',p_map)
cv2.waitKey(0)