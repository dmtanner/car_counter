import cv2 as cv2
import numpy as np

capture = cv2.VideoCapture(0)
backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(200)

while(1):
	success, img = capture.read()
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = backgroundSubtractor.apply(img)
	cv2.imshow("video", img)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

capture.release()
cv2.destroyAllWindows()
