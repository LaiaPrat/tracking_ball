# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",						# Llegim de la web cam
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, # Buffer per guardar les ubicacions de la pilota
	help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (29, 86, 6)			# Limit inferior del color verd en l'espai de color HSV
greenUpper = (64, 255, 255)			# Limit superior del color verd en l'espai de color HSV
pts = deque(maxlen=args["buffer"])	# Inicialitzem el deque de pts amb la mida max del buffer
# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):			# Dues opcions diferents si posem un vído o la web cam
	vs = VideoStream(src=0).start()
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])
# allow the camera or video file to warm up
time.sleep(2.0)

# keep looping
while True:							# Comença un cicle que no para fins que no apretem la tecla Q
	# grab the current frame		# o en cas d'utilitzar un arxiu de video que aquest acabi
	frame = vs.read()
	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame		# booleano --> Mira si detecta
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break
	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=600)					# Redimensionem el frame, més petit per processar més ràpid
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)				# desenfoquem per reduir el soroll
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)				# passem a l'espai de color HSV
	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, greenLower, greenUpper)				# Màscara binaria, per detectar la pilota
	mask = cv2.erode(mask, None, iterations=2)					# Apliquem una erosió i una dilatació per eliminar impureses dins i fora
	mask = cv2.dilate(mask, None, iterations=2)

# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,		# Busquem els contorns de la pilota
		cv2.CHAIN_APPROX_SIMPLE)								# per poder posar el centre
	cnts = imutils.grab_contours(cnts)
	center = None
	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))	# Obtenim el valor de les coordenades del centre
		# only proceed if the radius meets a minimum size
		if radius > 10:											# Comprobem que el radi sigui suficientment gran
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(x), int(y)), int(radius),	# Dibuixem un cercle en les corrdenades donades
				(0, 255, 255), 2)								# i un altre cercle al centre
			cv2.circle(frame, center, 5, (0, 0, 255), -1)		# les coordenades donades son el centre de la pilota
	# update the points queue
	pts.appendleft(center)										# Posa el centroide al centre de la pilota

# loop over the set of tracked points
	for i in range(1, len(pts)):										# Recorrem el bucle per cada un dels seus punts
		# if either of the tracked points are None, ignore				# Si el punt actual o l'anterior es igual a none
		# them															# vol dir que no s'ha detectat la pilota
		if pts[i - 1] is None or pts[i] is None:						# ignorem l'index i continuem el bucle
			continue													# Si el punt es vàlid calculamrem el gruix de la linia
		# otherwise, compute the thickness of the line and				# i el dibuixem en el frame
		# draw the connecting lines
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break
# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()
# otherwise, release the camera
else:
	vs.release()
# close all windows
cv2.destroyAllWindows()