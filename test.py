import cv2 as cv
import numpy as np

im1 = cv.imread('sources/source3.jpg')

h, w, _ = im1.shape
H, W = 512, 512

bg = np.full(shape=(H, W, 3), fill_value=255 ,dtype=np.uint8)
bg[H//2 - h//2:H//2 + h//2, W//2 - w//2:W//2 + w//2] = im1

cv.imwrite('sources/source3.jpg', bg)