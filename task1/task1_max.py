#this implementation is to detect the biggest bounding box of the defined colors

#opencv module
import cv2
#opencv takes numpy arrays as valid ranges for color scales (BGR, HSV etc.)
import numpy as np

#instantiates video stream from webcam device
cap = cv2.VideoCapture(0)

#checking if webcam device is properly connected
if cap.isOpened():
	print 'Valid webcam extension.'

while True:
	ret, frame = cap.read()

	#hard to define color ranges in BGR, change scale to HSV instead
	convert_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	kernel = np.ones((5, 5), "uint8")

	'''
	HSV - Hue, Saturation, Value

	Hue - describes the colors (aka tint)

	Saturation - shade of colors (aka amount of gray in the color)

	Value - measures brightness value

	In opencv, values on HSV are normalized to the following scales:
		H: [0, 179]
		S: [0, 255]
		V: [0, 255]
	'''

	#red - in HSV scale, red wraps about the edges of the 360 degree scale, hence occupying (0-20) / 2 and 340-360 / 2 of the Hue scale
	red_low_1 = np.array([170, 120, 70])
	red_high_1 = np.array([180, 255, 255])
	
	red_low_2 = np.array([0, 120 , 70])
	red_high_2 = np.array([10, 255, 255])

	#defining mask for each HSV range of red
	red_mask_1 = cv2.inRange(convert_hsv, red_low_1, red_high_1)
	red_mask_2 = cv2.inRange(convert_hsv, red_low_2, red_high_2)
	
	#mask should combine both ranges
	red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

	#increasing the object recognition size by the kernel size
	red_dilated = cv2.dilate(red_mask, kernel)

	#used for displaying frame with red-only details
	#red = cv2.bitwise_and(frame, frame, mask = red_mask)

	#blue, in HSV scale
	blue_low = np.array([110, 50, 50])
	blue_high = np.array([130, 255, 255])

	#defining mask for HSV range of red
	blue_mask = cv2.inRange(convert_hsv, blue_low, blue_high)

	blue_dilated = cv2.dilate(blue_mask, kernel)

	#blue = cv2.bitwise_and(frame, frame, mask = blue_mask)

	#green, in HSV scale
	green_low = np.array([50, 50, 50])
	green_high = np.array([70, 255, 255])

	green_mask = cv2.inRange(convert_hsv, green_low, green_high)

	green_dilated = cv2.dilate(green_mask, kernel)

	#orange, in HSV scale
	orange_low = np.array([5, 50, 50])
	orange_high = np.array([15, 255, 255])

	orange_mask = cv2.inRange(convert_hsv, orange_low, orange_high)
	orange_dilated = cv2.dilate(orange_mask, kernel)

	def get_bbox_max(color, mask):
		#return values are image, contours, hierarchy. we are interested in the contours of each colored object
		_, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		#ensures that there is at least 1 contour detected for the selected color
		if len(contours) > 0:
			area = max(contours, key = cv2.contourArea)

			#sets the x, y coordinates, width, height of the bounding box (rectangular shape)
			(xg, yg, wg, hg) = cv2.boundingRect(area)

			#draws the bounding box on the frame, border is in green
			cv2.rectangle(frame, (xg, yg), (xg + wg, yg + hg), (0, 255, 0), 2)

			#displays the color detected, also written in green
			cv2.putText(cv2.rectangle(frame, (xg, yg), (xg + wg, yg + hg), (0, 255, 0), 2), color, (xg, yg - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

	get_bbox_max('red', red_mask)
	get_bbox_max('blue', blue_mask)
	get_bbox_max('green', green_mask)
	#get_bbox_max('orange', orange_mask)

	#displays the video output
	cv2.imshow('MECAtron Task 1 - Color Detection (Max)', frame)

	#0xFF is a binary AND operation. when the 'Q' key is pressed, the program exits
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()