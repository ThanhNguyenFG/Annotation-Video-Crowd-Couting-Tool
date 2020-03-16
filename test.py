import cv2
import numpy as np

img = cv2.imread('E:/cap.jpg')
# cv2.circle(img,(447,63), 63, (0,0,255), 2)
#img = np.array(img)
# img = np.zeros((512,512,3), np.uint8)
# Draw a diagonal blue line with thickness of 5 px
cv2.line(img,(0,0),(511,511),(255,0,0),5)
cv2.imwrite('E:/capx.jpg',img)
cv2.imshow('image', img)
cv2.waitKey()

