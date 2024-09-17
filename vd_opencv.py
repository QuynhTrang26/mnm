import cv2
a1= cv2.imread('pic1.png',cv2.IMREAD_GRAYSCALE )
cv2.imshow('anh_ban_dau',a1)
cv2.imwrite('anhxam.png',a1)


