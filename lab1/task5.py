import cv2

img1 = cv2.imread('./source/porshe.png')
img2 = cv2.imread('./source/porshe.png')

cv2.namedWindow('porshe', cv2.WINDOW_NORMAL)
cv2.namedWindow('porshe_hsv', cv2.WINDOW_NORMAL)

cv2.imshow('porshe',img1)

hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
cv2.imshow('porshe_hsv', hsv)


cv2.waitKey(0)
cv2.destroyAllWindows()

