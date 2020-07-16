import sys
import time
from collections import deque
import argparse
import cv2
import imutils
import numpy as np
from imutils.video import VideoStream
import math
from munkres import Munkres

ACTIVE = 1
LOST = -1
DEAD = 0

class Tracker:

	def __init__(self):
		self.tracks = dict()

	def id_assignation(self, detections):
		m = Munkres()
		if len(self.tracks) == 0:
			for d in detections:
				id = len(self.tracks) + 1
				self.create_track(id)
				self.tracks[id].track_positions.append(d[0])
				self.tracks[id].track_radius = d[1]
				self.tracks[id].state = ACTIVE
				self.tracks[id].patience = 5
		else:
			nrows = len(detections)
			ncols = len(self.tracks)
			matrix_detec = np.ones((nrows, ncols)) * 1000
			j = 0
			for track_id in self.tracks.keys():
				i = 0
				for d in detections:
					(x1, y1) = d[0]
					(x2, y2) = self.tracks[track_id].track_positions[-1]
					dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
					matrix_detec[i][j] = dist
					i = i + 1
				j = j + 1
			indexes = m.compute(matrix_detec.tolist())
			values = list()
			for row, column in indexes:
				value = matrix_detec[row][column]
				values.append([row, column, value])
				if value <= 300:
					self.tracks[column+1].track_positions.append(detections[row][0])
					self.tracks[column+1].track_radius = detections[row][1]
					self.tracks[column+1].state = ACTIVE
					self.tracks[column+1].patience = 5
				else:
					if len(self.tracks) > len(detections):
						self.tracks[column + 1].state = LOST
						self.tracks[column + 1].patience = self.tracks[column + 1].patience - 1
						if self.tracks[column + 1].patience == 0:
							self.tracks[column + 1].track_radius = None
							self.tracks[column + 1].track_positions = None
							self.tracks[column + 1].state = DEAD
					else:
						id = len(self.tracks) + 1
						self.create_track(id)
						self.tracks[id].track_positions.append(detections[row][0])
						self.tracks[id].track_radius = detections[row][1]
						self.tracks[id].state = ACTIVE
						self.tracks[id].patience = 5
					# self.tracks[id].frame = len(video_frames)

	def create_track(self, id):
		self.tracks[id] = Track(id)

	def draw_tracks(self, frame):
		for track_id in self.tracks.keys():
			if self.tracks[track_id].state == ACTIVE:
				(x, y) = self.tracks[track_id].track_positions[-1]
				radius = self.tracks[track_id].track_radius
				cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
				cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
				cv2.putText(frame, str(track_id), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2)
				for i in range(1, len(self.tracks[track_id].track_positions)):
					if self.tracks[track_id].track_positions[i - 1] is None or self.tracks[track_id].track_positions[i] is None:
						continue
					thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
					cv2.line(frame, (int(self.tracks[track_id].track_positions[i - 1][0]),
									 int(self.tracks[track_id].track_positions[i - 1][1])),
							 (int(self.tracks[track_id].track_positions[i][0]),
							  int(self.tracks[track_id].track_positions[i][1])), (0, 0, 255), thickness)


class Detector:
	def __init__(self):
		pass

	def img_transformation(self, ima):
		blurred = cv2.GaussianBlur(ima, (11, 11), 0)
		hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
		kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(15,15))
		mask = cv2.inRange(hsv, greenLower, greenUpper)
		mask = cv2.erode(mask, None, iterations = 1)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
		return mask

	def detect(self, ima):
		ima = ima[1] if args.get("video", False) else ima
		if ima is None:
			sys.exit()
		mask = self.img_transformation(ima)
		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		detections = []
		if len(cnts) > 0:
			for c in cnts:
				circle = cv2.minEnclosingCircle(c)
				if circle[1] > 80:
					detections.append(circle)
		return detections


class Track:
	def __init__(self, track_id):
		self.track_id = track_id
		self.track_positions = deque(maxlen=args["buffer"])
		self.track_radius = None
		self.patience_count = None
		self.state = None
		# self.frame = None

	def get_id(self):
		return self.track_id


if __name__ == '__main__':

	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video", help="path to the (optional) video file")
	ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
	args = vars(ap.parse_args())
	greenLower = (29, 86, 6)
	greenUpper = (64, 255, 255)
	if not args.get("video", False):
		vs = VideoStream(src=0).start()
	else:
		vs = cv2.VideoCapture(args["video"])
	time.sleep(2.0)
	tracker = Tracker()
	detector = Detector()
	while (True):
		frame = vs.read()
		detections = detector.detect(frame)
		if len(tracker.tracks) > 0 and detections == 0:
			for id in tracker.tracks.keys():
				if tracker.tracks[id].state == LOST and tracker.tracks[id].patience_count == 1:
					tracker.tracks[id].state = DEAD
					tracker.tracks[id].patience_count = tracker.tracks[id].patience_count - 1
					tracker.tracks[id].track_radius = None
					tracker.tracks[id].track_positions = None
				else:
					tracker.tracks[id].state = LOST
					tracker.tracks[id].patience_count = tracker.tracks[id].patience_count - 1
		elif len(detections) > 0:
			tracker.id_assignation(detections)
		tracker.draw_tracks(frame)
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
	if not args.get("video", False):
		vs.stop()
	else:
		vs.release()
	cv2.destroyAllWindows()
