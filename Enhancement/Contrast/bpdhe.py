
import cv2 as cv
import numpy as np
import scipy.ndimage

# equivalent to fspecial
def matlab_style_gauss2D(shape=(1,9),sigma=1.0762):
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

def bpdhe(im):
  im = im.astype('uint8')
  hsv = cv.cvtColor(im, cv.COLOR_RGB2HSV)
  h,s,v = cv.split(hsv)
  h =  (h /255.0)
  s =  (s /255.0)

  ma = np.max(v)
  mi = np.min(v)
  bins = (ma-mi) + 1
  hist_i = np.histogram(v,bins=bins)
  hist_i = hist_i[0].reshape(1,len(hist_i[0]))
  gausFilter = matlab_style_gauss2D()
  #blur_hist = scipy.ndimage.convolve(hist_i[0], gausFilter, mode='nearest')
  blur_hist = cv.filter2D(hist_i.astype('float32'),-1,gausFilter, borderType=cv.BORDER_REPLICATE)
  derivFilter = np.array([[-1,1]])
  deriv_hist =  cv.filter2D(blur_hist.astype('float32'),-1,derivFilter, borderType=cv.BORDER_REPLICATE)
  sign_hist = np.sign(deriv_hist)
  meanFilter = np.array([[1/3,1/3,1/3]])
  smooth_sign_hist =  np.sign(cv.filter2D(sign_hist.astype('float32'),-1,meanFilter, borderType=cv.BORDER_REPLICATE))
  cmpFilter = np.array([[1,1,1,-1,-1,-1,-1,-1]])
  #index = np.zeros(shape=(1,3))
  #index[0] = 0
  p = 1
  index = [0]
  for n in range(0,bins-7):
    C = (smooth_sign_hist[0][n:n+8] == cmpFilter)*1
    if np.sum(C) == 8.0:
      p+=1
      index.append(n+3)

  index.append(bins)

  factor = np.zeros(shape=(len(index)-1,1))
  span = factor.copy()
  M = factor.copy()
  rangee = factor.copy()
  start = factor.copy()
  endd = factor.copy()
  sub_hist = []
  for m in range(0,len(index)-1):
    sub_hist.append( np.array(hist_i[0][index[m]:index[m+1]]) ) 
    M[m] = np.sum(sub_hist[m])
    low = mi + index[m]
    high = mi + index[m+1] - 1
    span[m] = high - low + 1
    factor[m] = span[m] * np.log10(M[m])
    factor_sum = np.sum(factor)
  for m in range(0,len(index)-1):
    rangee[m] = np.round((256-mi)*factor[m]/factor_sum)
  start[0] = mi
  endd[0] = mi + rangee[0]-1
  for m in range(1,len(index)-1):
    start[m] = start[m-1] + rangee[m-1]
    endd[m] = endd[m-1] + rangee[m]

  y = []
  s_r = np.zeros(shape=(1,mi))
  s_r = s_r.tolist()
  s_r = (s_r[0])
  for m in range(0, len(index)-1):
    hist_cum = np.cumsum(sub_hist[m]) 
    c = hist_cum/M[m]
    y.append( np.array(np.round(start[m]+(endd[m]-start[m])*c)) )
    x = y[m].tolist()
    s_r = s_r + x
  i_s = np.zeros(shape=v.shape)
  for n in range(mi,ma+1):
    lc = (v== n)
    i_s[lc] = (s_r[n])/255
  hsi_0 = cv.merge([h, s, i_s])
  hsi_0 = (hsi_0 * 255).astype('uint8')
  d = cv.cvtColor(hsi_0, cv.COLOR_HSV2RGB)

  return d
