import sys
import time
from collections import deque

import argparse
import cv2
import imutils
import numpy as np
from imutils.video import VideoStream


class Tracker:

	def __init__(self):
		self.tracks = dict()  # Dicionari

	def create_track(self, id):
		self.tracks[id] = Track(id)

	def check_if_track_exist(self, track_id):
		if self.tracks.get(track_id) is None:
			self.create_track(track_id)

	# def draw_track(self, track_id, center, circle):
	def draw_track(self, track_id, detections):
		pts = self.tracks[str(track_id)].track_positions
		# radius = circle[1]
		# (x, y) = center
		# x = int(keypoints[track_id].pt[0])
		# y = int(keypoints[track_id].pt[1])
		center = detections[1]
		keypoints = detections[0]
		radius = keypoints[track_id].size/2

		if radius > 10:  # Comprobem que el radi sigui suficientment gran
			# cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)  # Dibuixem un cercle en les corrdenades donades
			# cv2.circle(frame, center, 5, (0, 0, 255), -1)  # i un altre cercle al centre
			cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		pts.appendleft(center[track_id])
		for i in range(1, len(pts)):
			if pts[i - 1] is None or pts[i] is None:
				continue
			thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)  # Si el punt es vàlid calculamrem el gruix de la linia
			cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)  # i el dibuixem en el frame
			cv2.putText(frame, str(track_id), center, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2)


class Detector:
	def __init__(self):
		pass

	def img_transformations(self, im):
		blurred = cv2.GaussianBlur(im, (11, 11), 0)  # Desenfoquem per reduir el soroll
		hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  # Passem a l'espai de color HSV
		im_range = cv2.inRange(hsv, greenLower, greenUpper)  # Màscara binaria, per detectar la pilota
		im_range = cv2.erode(im_range, None, iterations=2)  # Apliquem una erosió i una dilatació per eliminar impureses dins i fora
		im_range = cv2.dilate(im_range, None, iterations=2)
		im_floodfill = im_range.copy()
		h, w = im_floodfill.shape[:2]
		mask = np.zeros((h + 2, w + 2), np.uint8)
		cv2.floodFill(im_floodfill, mask, (0, 0), 255)
		im_floodfill_inv = cv2.bitwise_not(im_floodfill)
		im_out = im_range | im_floodfill_inv
		return im_out


	def detect(self, im):
		im = im[1] if args.get("video", False) else im
		if im is None:
			sys.exit()
		mask = self.img_transformations(im)

		params = cv2.SimpleBlobDetector_Params()
		# Change thresholds
		params.minThreshold = greenLower
		params.maxThreshold = greenUpper
		# Filter by Area.
		params.filterByArea = True
		params.minArea = 1000
		params.maxArea = 3000
		# Filter by Circularity
		params.filterByCircularity = True
		params.minCircularity = 0.9
		# Filter by Convexity
		params.filterByConvexity = True
		params.minConvexity = 0.5
		# Filter by Inertia
		params.filterByInertia = True
		params.minInertiaRatio = 0.5
		reverse_mask = 255 - mask
		# keypoints = cv2.SimpleBlobDetector().detect(reverse_mask)
		# # det = cv2.SimpleBlobDetector_create(params)
		keypoints = cv2.SimpleBlobDetector_create(params).detect(reverse_mask)
		# keypoints = detect
		# keypoints = det.detect(mask)

		im_with_keypoints = cv2.drawKeypoints(reverse_mask, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		cv2.imshow("Keypoints", im_with_keypoints)
		cv2.waitKey(0)

		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		# center = None
		# radius = None
		# circle = None
		# if len(cnts) > 0: # Si no faig aquest if em peta si no detecta la pilota
		# for k in keypoints:
		c = max(cnts, key=cv2.contourArea)
		# circle = cv2.minEnclosingCircle(c) # Obtenim el cercle que despres dibuixarem
		M = cv2.moments(c)
			# center[k] = (keypoints[k].pt[0],keypoints[k].pt[1])
			# radius[k] = keypoints[k].size/2
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))  # Coordenades del centre
		detections = [keypoints, center]
		return detections


class Track:
	def __init__(self, track_id):
		self.track_id = track_id
		self.track_positions = deque(maxlen=args["buffer"])

	def get_id(self):
		return self.track_id


if __name__ == '__main__':

	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video",  # Llegim de la web cam o d'un video
					help="path to the (optional) video file")
	ap.add_argument("-b", "--buffer", type=int, default=64,  # Buffer per guardar les ubicacions de la pilota
					help="max buffer size")
	args = vars(ap.parse_args())
	greenLower = (29, 86, 6)  # Limit inferior del color verd en l'espai de color HSV
	greenUpper = (64, 255, 255)  # Limit superior del color verd en l'espai de color HSV
	if not args.get("video", False):  # Dues opcions diferents si posem un vído o la web cam
		vs = VideoStream(src=0).start()
	else:
		vs = cv2.VideoCapture(args["video"])
	time.sleep(2.0)

	detector = Detector()
	tracker = Tracker()
	# tracker.create_track(0)
	while (True):
		frame = vs.read() # un for de les detections? em quedara una matriu
		detections = detector.detect(frame)
		if len(detections) > 0:
			for d in detections:
				tracker.create_track(str(d))
				tracker.draw_track(d, detections)
		# if len(detections[0]) > 0:
		# 	tracker.draw_track(0, detections[1], detections[2]) # Millor així o fent referencia de cada cosa?
		cv2.imshow("Frame", frame)							# Per exemple: contorns = detections[0]
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
	if not args.get("video", False):
		vs.stop()
	else:
		vs.release()
	cv2.destroyAllWindows()

# for t in Tracker:
# 	tracker.create_track(t)
# 	print('1')
# 	if len(cnts) > 0:
# 		tracker.draw_track(t, cnts)
