# Steps
#1. Read and show video stream, capture images
#2. Detect faces and show bounding boxes
#3. Flatten the largest face image and save in numpy array
#4. Repeat the above for multiple people to generate training data
import cv2 as cv
import numpy as np
file_name = input("Enter the name of the person: ")
cap = cv.VideoCapture(0)
face_data = []
skip = 0 # storing the every 10th frame
offset = 10 # padding
data_path = './data/'
face_cascade = cv.CascadeClassifier("haarcascade_frontalface.xml")
while True:
	ret, frame = cap.read()
	g_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	if ret == False:
		continue
	faces = face_cascade.detectMultiScale(frame, 1.3, 5)
	if len(faces) == 0:
		continue
	faces = sorted(faces, key = lambda f:f[2]*f[3])
	(x, y, w, h) = faces[-1]
	face_section = g_frame[y - offset:y + h + offset, x - offset: x + w + offset]
	face_section = cv.resize(face_section, (100, 100))
	if skip%10 == 0:
		face_data.append(face_section)	
		cv.imshow("Captured", face_section)
	cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255))
	cv.imshow("Smile", frame)	
	skip += 1
	if cv.waitKey(1) == 27:
		break
cap.release()
cv.destroyAllWindows()
# converting face data into numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)
# saving data into file system
np.save(data_path + file_name + '.npy', face_data)
