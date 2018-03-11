# Avinash Kothapalli
# Python code to implement motion tracking based on Contours (We first use Background differentiation for motion detection)

import datetime
import imutils
import cv2

# Read the video feed from the webcam
camera = cv2.VideoCapture(0)

# Initialize the first frame in the video stream
initFrame = None

# Looping over the frames of the video
while True:
    # Grab each frame into frame
    (grabbed, frame) = camera.read()
    # Flag to display during tracking vs. not-tracking
    text = "No Motion"


    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed:
        print "No more video to be grabbed"
        break

    # Resizing the frame
    frame = imutils.resize(frame, width=800)
    # Converting the frame to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Blurring the frame using Gaussian blur
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Initialize the initFrame for the background differentiation method
    if initFrame is None:
        initFrame = gray
        continue

    # Calculate the difference between the current frome and the background
    frameDelta = cv2.absdiff(initFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilate the thresholded image to fill in holes, then find contours on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    (image, contours, heirarcy) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours
    for c in contours:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 500:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Motion Detected"

        # draw the text and timestamp on the frame
        cv2.putText(frame, "Status: {}".format(text), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # show the frame and record if the user presses a key
        cv2.imshow("Security Feed", frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)
        key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
