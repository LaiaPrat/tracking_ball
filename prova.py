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

	def draw_track(self, track_id, detection):
		# center = detections[0]
		(x, y) = detection[0]
		radius = detection[1]
		# for center in detections[0]:
		# 	(x, y) = center[track_id]
		# for r in detections[1]:
		# 	radius = r[track_id]
		# (x, y) = detections[track_id][0]
		# radius = detections[track_id][1]
		# circle = detections[1]
		pts = self.tracks[track_id].track_positions
		# radius = circle[1]
		# (x, y) = circle[0]
		# if int(radius) > 10:  # Comprobem que el radi sigui suficientment gran
		cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)  # Dibuixem un cercle en les corrdenades donades
		cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)  # i un altre cercle al centre
		# M = cv2.moments(detection)
		# center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		# pts.appendleft(detection[0])		# les coordenades donades son el centre de la pilota
		pts.appendleft(detection[0])
		# cv2.imshow("frame", frame)
		# cv2.waitKey(0)

		for i in range(1, len(pts)):  # Recorrem el bucle per cada un dels seus punts
			if pts[i - 1] is None or pts[i] is None:  # Si és none vol dir que no s'ha detectat la pilota
				continue  # ignorem l'index i continuem el bucle
			thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)  # Si el punt es vàlid calculamrem el gruix de la linia
			cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)  # i el dibuixem en el frame
			cv2.putText(frame, str(track_id), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2)


class Detector:
	def __init__(self):
		pass

	def img_transformation(self, ima):
		blurred = cv2.GaussianBlur(ima, (11, 11), 0)  # Desenfoquem per reduir el soroll
		hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  # Passem a l'espai de color HSV

		mask = cv2.inRange(hsv, greenLower, greenUpper)  # Màscara binaria, per detectar la pilota
		mask = cv2.erode(mask, None, iterations=2)  # Apliquem una erosió i una dilatació per eliminar impureses dins i fora
		mask = cv2.dilate(mask, None, iterations=2)
		return mask


	def detect(self, ima):
		ima = ima[1] if args.get("video", False) else ima
		if ima is None:
			sys.exit()
		mask = self.img_transformation(ima)
		# cv2.imshow("mask", mask)
		# cv2.waitKey(0)
		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,  # Busquem els contorns de la pilota
								cv2.CHAIN_APPROX_SIMPLE)  # per poder posar el centre
		cnts = imutils.grab_contours(cnts)
		# center = None
		# circle = None
		detections = []
		if len(cnts) > 0:
			# k = max(cnts, key=cv2.contourArea)
			for c in cnts:
				# c = max(cnts, key=cv2.contourArea)
				circle = cv2.minEnclosingCircle(c) # Obtenim el cercle que despres dibuixarem
				# M = cv2.moments(c)
				# center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))  # Obtenim el valor de les coordenades del centre
				if circle[1] > 50:  # Probar amb 10
					detections.append(circle)
				# detections = [[center][circle]]
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
	tracker = Tracker()
	# tracker.create_track(0)
	detector = Detector()
	while (True):
		frame = vs.read()
		detections = detector.detect(frame)
		# cnts = detections[0]
		# center = detections[1]
		# circle = detections[2]
		id = 0
		# tracker.create_track(id)
		for k in detections:
			tracker.create_track(id)
		# if len(cnts) > 0:
		# 	center = detections[0]
		# 	circle = detections[1] Linia d sota cal indicar detections[k]?
		# 	tracker.draw_track(k, detections)
			tracker.draw_track(id, k)
			id = id + 1
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
	if not args.get("video", False):
		vs.stop()
	else:
		vs.release()
	cv2.destroyAllWindows()