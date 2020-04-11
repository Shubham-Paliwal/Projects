#1 data collection
#2 load the training data(numpy arrays of all the person)
	#x - features stored in numpy array
	#y - values we need to assign to each person
#3 use knn to find the prediction of face
#4 map the predicted id to nane of the user
#5 Didplay the predictions on the screen
import numpy as np
import cv2 as cv
import os
def dist(x1, x2):
	return np.sqrt(sum((x1 - x2)**2))
def knn(x, y, q, k):
	vals = []
	for i in range(x.shape[0]):
		d = dist(x[i], q)
		vals.append((d, y[i]))
	vals = sorted(vals)
	vals = np.array(vals[:k])
	freq = np.unique(vals[:, 1], return_counts = True)
	idx = freq[1].argmax()
	return freq[0][idx]
#---------------------------------------------------------
data_path = './data/'
xt = []
yt = []
name = {}
class_id = 0
for fx in os.listdir(data_path):
	if fx.endswith('.npy'):
		name[class_id] = fx[:-4]
		xc = np.load(data_path + fx)
		xt.append(xc)
		yc = class_id*np.ones((xc.shape[0], ))
		yt.append(yc) 
xt = np.concatenate(xt, axis = 0)
yt = np.concatenate(yt, axis = 0)
#---------------------------------------------------------
cap = cv.VideoCapture(0)
offset = 10
face_cascade = cv.CascadeClassifier("haarcascade_frontalface.xml")
while True:
	ret, frame = cap.read()
	g_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	if ret == False:
		continue
	faces = face_cascade.detectMultiScale(frame, 1.3, 5);
	if len(faces) == 0:
		continue
	faces = sorted(faces, key = lambda f:f[2]*f[3])
	(x, y, w, h) = faces[-1]
	face_section = g_frame[y - offset : y + h + offset, x - offset : x + w + offset]
	face_section = cv.resize(face_section, (100, 100))
	face_section = face_section.flatten()
	pred_name = name[int(knn(xt, yt, face_section, 5))]
	cv.putText(frame, pred_name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
	cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255))
	cv.imshow("Frame", frame)
	if cv.waitKey(1) == 27:
		break
cap.release()
cv.destroyAllWindows()