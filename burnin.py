import cv2
import numpy
import time
import turtle

MOTION_THRESHOLD = 2.25
EDGE_THRESHOLD = 1.25
REFERENCE_SCALE = 0.75
INDICATOR_SIZE = 0.02

turtle = turtle.Turtle()
turtle.hideturtle()
turtle.speed(0)
turtle.screen.tracer(False)
turtle.fillcolor(0.75, 0, 0)
turtle.resizemode("user")

cap = cv2.VideoCapture(0)
prev_frame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)

while True:
    # Compute dense optical flow
    current_frame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
    motion = cv2.calcOpticalFlowFarneback(prev_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    weights = numpy.repeat(numpy.expand_dims(numpy.linalg.norm(motion, axis=-1), axis=-1), 2, axis=-1)

    # Compute center of mass
    coords = numpy.stack(numpy.mgrid[0:current_frame.shape[0], 0:current_frame.shape[1]], axis=-1)
    center = numpy.average(coords, axis=(0, 1), weights=weights).astype(numpy.int)

    y = center[0]
    x = center[1]

    if weights[y, x, 0] >= MOTION_THRESHOLD:
        # the X coordinate of the center of mass becomes the X position
        screenx = (0.5 - x / current_frame.shape[1]) * turtle.screen.window_width()

        # now expand outward from the center of mass until we find the edges of the moving shape
        # the width of this approximates our Z position
        left = weights[y, :x, 0]
        right = weights[y, x:, 0]
        leftedge = numpy.argmax(left > EDGE_THRESHOLD)
        rightedge = numpy.argmax(right < EDGE_THRESHOLD)

        width = numpy.minimum((x - leftedge + rightedge) / (REFERENCE_SCALE * current_frame.shape[1]), 0.95)
        screenz = (width - 0.5) * turtle.screen.window_height()

        turtle.screen.clear()
        turtle.penup()
        turtle.goto(screenx, screenz)

        turtle.begin_fill()
        turtle.circle(INDICATOR_SIZE * turtle.screen.window_width())
        turtle.end_fill()

        turtle.screen.update()

    prev_frame = current_frame
    time.sleep(0)
