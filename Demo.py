############################################################################################################
############################################################################################################
#COMPANY:N/A
#PROGRAM: VIDEO CLASSIFICATION
#DEVELOPER: RIYADH ALKHANIN
#DATE:05082019
#VERSION:0.3
############################################################################################################
############################################################################################################


# import the necessary packages
from keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2


# import the necessary packages IMP MAY NEED UPDATING
args= {
    "label-bin": "lb.pickle",
    "model": "Model.h5",
    "input": "Input Sample/Golden_State_Warriors1.mp4",
    "size": 128
}


## Linex Useres May change argument to this ##

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-m", "--model", required=True,
#	help="path to trained serialized model")
#ap.add_argument("-l", "--label-bin", required=True,
#	help="path to  label binarizer")
#ap.add_argument("-i", "--input", required=True,
#	help="path to our input video")
#ap.add_argument("-o", "--output", required=True,
#	help="path to our output video")
#ap.add_argument("-s", "--size", type=int, default=128,
#	help="size of queue for averaging")
#args = vars(ap.parse_args())

# This is to load the trained model and label binarizer

print("Thakaa Demo: loading model and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open('lb.pickle', "rb").read())

# initialize the image mean with the predictions queue
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args["size"])

# Starting the video stream, pointer to input video file, and  frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# looping over frames
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not taken, then we have reached the end of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty take it
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# cloning the output frame, then convert it from BGR to RGB ordering, EXTRA: resize the frame to a fixed 640X368, and then perform mean subtraction
	output = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (640, 368)).astype("float32")
	frame -= mean

	# make predictions queues on the frame and then update the predictions
	preds = model.predict(np.expand_dims(frame, axis=0))[0]
	Q.append(preds)

	# Predecting avergae over the current past of predictions
	results = np.array(Q).mean(axis=0)
	i = np.argmax(results)
	label = lb.classes_[i]

	# writing the scene on output frame
	text = "Scene: {}".format(label)
	cv2.putText(output, text, (15, 25), cv2.FONT_HERSHEY_TRIPLEX,
		1.0, (255, 255, 255), 1)

	# EXTRA: checking video writer
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		writer = cv2.VideoWriter('Result_Golden_State_Warriors.avi', fourcc, 30,
			(W, H), True)

	# EXTRA: show the output Video
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	writer.write(output)
	if key == ord("q"):
		break

# release the file pointers
print("Thakaa Demo: cleaning up...")
writer.release()
vs.release()
