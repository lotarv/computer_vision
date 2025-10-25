import cv2

img1 = cv2.imread('./source/orange_cat.jpg',cv2.IMREAD_GRAYSCALE) #ЧБ
img2 = cv2.imread('./source/gray_cat.png',cv2.IMREAD_UNCHANGED) # без изменений
img3 = cv2.imread('./source/sleep_cat.bmp',cv2.IMREAD_ANYDEPTH) # с учетом  глубины цвета
cv2.namedWindow('orange_cat', cv2.WINDOW_NORMAL)
cv2.namedWindow('gray_cat', cv2.WINDOW_NORMAL)
cv2.namedWindow('sleep_cat', cv2.WINDOW_NORMAL)
cv2.imshow('orange_cat',img1)
cv2.imshow('gray_cat', img2)
cv2.imshow('sleep_cat', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

