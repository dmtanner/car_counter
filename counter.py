import cv2 as cv2
import numpy as np

MIN_AREA = 50

capture = cv2.VideoCapture(0)
backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(200)

# Let Camera Adjust to Lighting
for i in range(20):
	success, img = capture.read()

while(1):
	# Get Image
	success, img = capture.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = backgroundSubtractor.apply(gray)
	ret, thresh = cv2.threshold(gray, 1, 255, 0)

	# Erode and Dilate
	thresh = cv2.dilate(thresh, (3,3), iterations=1 )
	thresh = cv2.erode(thresh, (3,3), iterations=2 )
	thresh = cv2.dilate(thresh, (3,3), iterations=2 )

	# extract and draw large contours
	img_contour,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	large_contours = []
	for contour in contours:
		if cv2.contourArea(contour) > MIN_AREA:
			large_contours.append(contour)
	cv2.drawContours(img, large_contours, -1, (255,0,0), 3)

	# Show Image
	cv2.imshow("video", img)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

capture.release()
cv2.destroyAllWindows()
