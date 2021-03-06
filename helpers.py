import cv2
import numpy as np
from skimage.transform import resize
from skimage import io, color, img_as_ubyte
from skimage.feature import hog,local_binary_pattern,greycomatrix, greycoprops

def preprocess(img):       
  img = cv2.copyMakeBorder(img,top=3,bottom=3,left=3,right=3,borderType=cv2.BORDER_CONSTANT,value=[255,255,255])        
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.GaussianBlur(img,(3,3),0)
  _, bin_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  return bin_img
def findContours(img):
  contours, _= cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
  contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:]
  return contours

def findHOG(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = resize(gray, (64, 128))
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, multichannel=False)
    return fd,hog_image

def findLBP(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 16, 2,  method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, 27),range=(0, 26))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def findGLCM(img):
  gray = color.rgb2gray(img)
  image = img_as_ubyte(gray)
  bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])
  inds = np.digitize(image, bins)
  max_value = inds.max()+1
  matrix_coocurrence = greycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=max_value, normed=False, symmetric=False)
  contrast = greycoprops(matrix_coocurrence, 'contrast')
  dissimilarity = greycoprops(matrix_coocurrence, 'dissimilarity')	
  homogeneity = greycoprops(matrix_coocurrence, 'homogeneity')
  energy = greycoprops(matrix_coocurrence, 'energy')
  correlation = greycoprops(matrix_coocurrence, 'correlation')
  asm = greycoprops(matrix_coocurrence, 'ASM')
  return [ *contrast[0],*dissimilarity[0],*homogeneity[0],*energy[0],*correlation[0],*asm[0]]

def findHINGE(img):
  img= preprocess(img)
  contours = findContours(img)
  hist = np.zeros((12, 12))
  for countour in contours:
    n = len(countour)
    if n<= 25:
      continue
    points = np.array([point[0] for point in countour])
    xs, ys = points[:, 0], points[:, 1]
    point_1s = np.array([countour[(i + 25) % n][0] for i in range(n)])
    point_2s = np.array([countour[(i - 25) % n][0] for i in range(n)])
    x1s, y1s = point_1s[:, 0], point_1s[:, 1]
    x2s, y2s = point_2s[:, 0], point_2s[:, 1]
    phi_1s = np.degrees(np.arctan2(y1s - ys, x1s - xs) + np.pi)
    phi_2s = np.degrees(np.arctan2(y2s - ys, x2s - xs) + np.pi)
    indices = np.where(phi_2s > phi_1s)[0]
    for i in indices:
      phi1 = int(phi_1s[i] // (360//12)) % 12
      phi2 = int(phi_2s[i] // (360//12)) % 12
      hist[phi1, phi2] += 1
  norm_hist = hist / np.sum(hist)
  return  norm_hist[np.triu_indices_from(norm_hist, k = 1)]

def extract_features(img):
    fd, _ = findHOG(img)
    hist = findLBP(img)
    glcm = findGLCM(img)
    hinge = findHINGE(img)
    features = [*fd, *glcm,*hinge,*hist]
    return features