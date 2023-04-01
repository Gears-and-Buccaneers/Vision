import cv2 as cv
import numpy as np
import ntcore

nt = ntcore.NetworkTableInstance.getDefault()

coneX = nt.getDoubleTopic("vision/coneX").publish()
coneY = nt.getDoubleTopic("vision/coneY").publish()
coneCenter = nt.getDoubleTopic("vision/coneCenter").publish()


hsv = None

CONE_COLOR_LOWER = np.array([10, 140, 150])
CONE_COLOR_UPPER = np.array([30, 255, 255])

CUBE_COLOR_LOWER = np.array([110, 150, 120])
CUBE_COLOR_UPPER = np.array([135, 255, 255])

CUBE_RADIUS_MIN = 40

def process(frame):
	hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

	cone_mask = cv.inRange(hsv, CONE_COLOR_LOWER, CONE_COLOR_UPPER)
	cube_mask = cv.inRange(hsv, CUBE_COLOR_LOWER, CUBE_COLOR_UPPER)

	cone_contours, _ = cv.findContours(
		cone_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	cube_contours, _ = cv.findContours(
		cube_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

	cv.drawContours(frame, cube_contours, -1, (255, 0, 255), 5)

	contour = None

	for c in cone_contours:
		if contour is None or len(contour) < len(c):
			contour = c

	if contour is not None:
		left = right = down = up = None

		cv.drawContours(frame, [contour], -1, (255, 0, 0), 2)

		for point in contour:
			if left is None or point[0, 0] < left[0]:
				left = point[0]
			if right is None or point[0, 0] > right[0]:
				right = point[0]
			if up is None or point[0, 1] < up[1]:
				up = point[0]
			if down is None or point[0, 1] > down[1]:
				down = point[0]


		x = (left[0] + right[0]) / 2
		y = (up[1] + down[1]) / 2

		cv.rectangle(frame, (left[0], up[1]), (right[0], down[1]), (0, 255, 0), 2)
		cv.circle(frame, (int(x), int(y)), 7, (0, 255, 0), -1)

		cv.drawMarker(frame, left, (0, 255, 0), cv.MARKER_CROSS, 50, 2)
		cv.drawMarker(frame, right, (0, 255, 0), cv.MARKER_CROSS, 50, 2)
		cv.drawMarker(frame, up, (0, 255, 0), cv.MARKER_CROSS, 50, 2)
		cv.drawMarker(frame, down, (0, 255, 0), cv.MARKER_CROSS, 50, 2)

		coneX.set(x / frame.shape[1])
		coneY.set(y / frame.shape[0])
		coneCenter.set((down[0] - left[0]) / (right[0] - left[0]))

	for contour in cube_contours:
		rect = cv.minAreaRect(contour)
		if rect is not None:
			points = cv.boxPoints(rect).astype(np.int32)

			cv.line(frame, points[0], points[1], (255, 0, 255), 5)
			cv.line(frame, points[1], points[2], (255, 0, 255), 5)
			cv.line(frame, points[2], points[3], (255, 0, 255), 5)
			cv.line(frame, points[3], points[0], (255, 0, 255), 5)


	cv.imshow('img', frame)


def mouseHandler(event, x, y, flags, param):
	if event == cv.EVENT_LBUTTONDOWN and hsv is not None:
	 	 print(hsv[y, x])

cv.namedWindow('img')
cv.setMouseCallback('img', mouseHandler)


cap = cv.VideoCapture(0)

if not cap.isOpened():
	print("Cannot open camera")
	exit()


while True:
	_, frame = cap.read()
	process(frame)
	if cv.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv.destroyAllWindows()
