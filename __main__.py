import cv2 as cv
import numpy as np

hsv = None


def mouseHandler(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(hsv[y, x])


cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

cv.namedWindow('original')
cv.setMouseCallback('original', mouseHandler)

while True:
    ret, frame = cap.read()

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    cone_lower = np.array([10, 140, 100])
    cone_upper = np.array([30, 255, 255])

    cone_mask = cv.inRange(hsv, cone_lower, cone_upper)

    cone_contours, _ = cv.findContours(
        cone_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cube_lower = np.array([125, 120, 120])
    cube_upper = np.array([135, 255, 255])

    cube_mask = cv.inRange(hsv, cube_lower, cube_upper)

    cube_contours, _ = cv.findContours(
        cube_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cube_contours = list(map(lambda contour: cv.approxPolyDP(
        contour, cv.arcLength(contour, True) * 0.02, True), cube_contours))

    cone_contours = list(map(lambda contour: cv.approxPolyDP(
        contour, cv.arcLength(contour, True) * 0.01, True), cone_contours))

    cv.drawContours(frame, cube_contours,  color=(255, 0, 255),
                    contourIdx=-1, thickness=2, lineType=cv.LINE_AA)

    cv.drawContours(frame, cone_contours,  color=(0, 255, 255),
                    contourIdx=-1, thickness=2, lineType=cv.LINE_AA)

    cv.imshow('original', frame)

    cv.imshow('cube_mask', cube_mask)
    cv.imshow('cone_mask', cone_mask)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
