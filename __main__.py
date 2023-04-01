import cv2 as cv
import numpy as np
from networktables import NetworkTables
import ntcore

NetworkTables.initialize(server='roborio-2972-frc.local')

conePosPublisher = ntcore.NetworkTableInstance.getDefault().getDoubleTopic("vision/conePos").publish()

hsv = None

CONE_COLOR_LOWER = np.array([10, 140, 150])
CONE_COLOR_UPPER = np.array([30, 255, 255])

CUBE_COLOR_LOWER = np.array([110, 150, 120])
CUBE_COLOR_UPPER = np.array([135, 255, 255])

CUBE_RADIUS_MIN = 40

def mouseHandler(event, x, y, flags, param):
	if event == cv.EVENT_LBUTTONDOWN and hsv is not None:
	 	 print(hsv[y, x])

cap = cv.VideoCapture(0)

if not cap.isOpened():
	print("Cannot open camera")
	exit()

cv.namedWindow('img')

cv.setMouseCallback('img', mouseHandler)


while True:
	_, frame = cap.read()

	hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

	cone_mask = cv.inRange(hsv, CONE_COLOR_LOWER, CONE_COLOR_UPPER)
	cube_mask = cv.inRange(hsv, CUBE_COLOR_LOWER, CUBE_COLOR_UPPER)

	cone_contours, _ = cv.findContours(
		cone_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	cube_contours, _ = cv.findContours(
		cube_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

	cv.drawContours(frame, cube_contours, -1, (255, 0, 255), 5)
	cv.drawContours(frame, cone_contours, -1, (0, 255, 255), 5)

	bestContour = None

	for contour in cone_contours:
		if bestContour is None or len(bestContour) < len(contour):
			bestContour = contour

	if bestContour is not None:
		left = right = up = down = None
		contour = bestContour.reshape((-1, 2))

		for point in contour:
			if left is None or point[0] < left[0]:
				left = point
			if right is None or point[0] > right[0]:
				right = point
			if up is None or point[1] < up[1]:
				up = point
			if down is None or point[1] > down[1]:
				down = point

		cv.drawMarker(frame, left, (0, 255, 0), cv.MARKER_CROSS, 50, 2)
		cv.drawMarker(frame, right, (0, 255, 0), cv.MARKER_CROSS, 50, 2)
		cv.drawMarker(frame, up, (0, 255, 0), cv.MARKER_CROSS, 50, 2)
		cv.drawMarker(frame, down, (0, 255, 0), cv.MARKER_CROSS, 50, 2)

		x = np.average(contour, 0)[0]
		a = x / len(frame[0]) * 2 - 1
		cv.line(frame, (int(x), 0), (int(x), len(frame)), (0, 0, 255), 5)
		conePosPublisher.set(a)

	for contour in cube_contours:
		rect = cv.minAreaRect(contour)
		if rect is not None:
			points = cv.boxPoints(rect).astype(np.int32)

			cv.line(frame, points[0], points[1], (255, 0, 255), 5)
			cv.line(frame, points[1], points[2], (255, 0, 255), 5)
			cv.line(frame, points[2], points[3], (255, 0, 255), 5)
			cv.line(frame, points[3], points[0], (255, 0, 255), 5)


	cv.imshow('img', frame)

	if cv.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv.destroyAllWindows()
