import cv2
import numpy as np
from copy import deepcopy 
import json

class VisualTactile:
    def __init__(self):
        self.camera_num = 1

        self.capture_frame = None
        self.processing_frame = None

        self.TRANSFORM_HEIGHT = int(65.8*7)
        self.TRANSFORM_WIDTH = int(78.0*7)
        self.original_points = np.float32([[0, 0], [100, 0], [100, 300], [1, 140]])

        self.ORIGINAL_HEIGHT = 480
        self.ORIGINAL_WIDTH = 640

        self.blob_params = cv2.SimpleBlobDetector_Params()
        
        self.JSON_PARAMS_PATH = "./params.json"

    def perspectiveTransform(self):
        # Define the four corners of the region of interest (ROI) in the original image
        output_points = np.float32([[0, 0], [self.TRANSFORM_WIDTH, 0], [self.TRANSFORM_WIDTH, self.TRANSFORM_HEIGHT], [0, self.TRANSFORM_HEIGHT]])
        
        # Compute the perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(original_points, output_points)
        self.processing_frame = cv2.warpPerspective(self.capture_frame, matrix, (self.TRANSFORM_WIDTH, self.TRANSFORM_WIDTH))

        cv2.polylines(self.capture_frame, [np.int32(original_points)], isClosed=True, color=(0, 255, 0), thickness=2)
        # cv2.imshow('Video Frame Perspective Transform', self.capture_frame)

    def setInitialParams(self):
        with open(self.JSON_PARAMS_PATH, 'r') as f:
            params = json.load(f)

        params = params["blob"]
        cv2.setTrackbarPos('Min Threshold', 'Blob Parameters', int(params["minThreshold"]))
        cv2.setTrackbarPos('Max Threshold', 'Blob Parameters', int(params["maxThreshold"]))
        cv2.setTrackbarPos('Min Area', 'Blob Parameters', int(params["minArea"] - 0.1))
        cv2.setTrackbarPos('Max Area', 'Blob Parameters', int(params["maxArea"] - 0.1))
        cv2.setTrackbarPos('Min Circularity', 'Blob Parameters', int((params["minCircularity"] - 0.01) * 100))
        cv2.setTrackbarPos('Min Convexity', 'Blob Parameters', int((params["minConvexity"] - 0.01) * 100))
        cv2.setTrackbarPos('Min Inertia', 'Blob Parameters', int((params["minInertiaRatio"] - 0.01) * 100))

    def updateParams(self):
        # Get current trackbar positions and Update blob detector parameters
        self.blob_params.minThreshold = cv2.getTrackbarPos('Min Threshold', 'Blob Parameters')
        self.blob_params.maxThreshold = cv2.getTrackbarPos('Max Threshold', 'Blob Parameters')
        self.blob_params.minArea = cv2.getTrackbarPos('Min Area', 'Blob Parameters') + 0.1
        self.blob_params.maxArea = cv2.getTrackbarPos('Max Area', 'Blob Parameters') + 0.1 
        self.blob_params.minCircularity = cv2.getTrackbarPos('Min Circularity', 'Blob Parameters') / 100 + 0.01
        self.blob_params.minConvexity = cv2.getTrackbarPos('Min Convexity', 'Blob Parameters') / 100  + 0.01
        self.blob_params.minInertiaRatio = cv2.getTrackbarPos('Min Inertia', 'Blob Parameters') / 100 + 0.01

    def saveParams(self):
        with open(self.JSON_PARAMS_PATH, 'r') as f:
            params = json.load(f)
            params["blob"]["minThreshold"] = self.blob_params.minThreshold 
            params["blob"]["maxThreshold"] = self.blob_params.maxThreshold 
            params["blob"]["minArea"] = self.blob_params.minArea
            params["blob"]["maxArea"] = self.blob_params.maxArea
            params["blob"]["minCircularity"] = self.blob_params.minCircularity
            params["blob"]["minConvexity"] = self.blob_params.minConvexity 
            params["blob"]["minInertiaRatio"] = self.blob_params.minInertiaRatio 

        with open(self.JSON_PARAMS_PATH, 'w') as f:
            json.dump(params, f, indent=4)
    
    def createBlobWindow(self):
        cv2.namedWindow('Blob Parameters')
        # Create trackbars for adjusting parameters
        cv2.createTrackbar('Min Threshold', 'Blob Parameters', 0, 500, lambda x : self.updateParams())
        cv2.createTrackbar('Max Threshold', 'Blob Parameters', 0, 500, lambda x : self.updateParams())
        cv2.createTrackbar('Min Area', 'Blob Parameters', 0, 200, lambda x : self.updateParams())
        cv2.createTrackbar('Max Area', 'Blob Parameters', 0, 200, lambda x : self.updateParams())
        cv2.createTrackbar('Min Circularity', 'Blob Parameters', 0, 100, lambda x : self.updateParams())
        cv2.createTrackbar('Min Convexity', 'Blob Parameters', 0, 100, lambda x : self.updateParams())
        cv2.createTrackbar('Min Inertia', 'Blob Parameters', 0, 100, lambda x : self.updateParams())

    def createPerspectiveWindow(self):
        cv2.namedWindow('Perspective Transform Parameters')
        # Create trackbars for adjusting parameters
        cv2.createTrackbar('Min Threshold', 'Blob Parameters', 0, 500, lambda x : self.updateParams())
        cv2.createTrackbar('Max Threshold', 'Blob Parameters', 0, 500, lambda x : self.updateParams())
        cv2.createTrackbar('Min Area', 'Blob Parameters', 0, 200, lambda x : self.updateParams())
        cv2.createTrackbar('Max Area', 'Blob Parameters', 0, 200, lambda x : self.updateParams())
        cv2.createTrackbar('Min Circularity', 'Blob Parameters', 0, 100, lambda x : self.updateParams())
        cv2.createTrackbar('Min Convexity', 'Blob Parameters', 0, 100, lambda x : self.updateParams())
        cv2.createTrackbar('Min Inertia', 'Blob Parameters', 0, 100, lambda x : self.updateParams())


    def createParamtersWindow(self):
        self.createBlobWindow()
        self.createPerspectiveWindow()

        self.setInitialParams()

    def detectBlob(self):
        # Create a blob detector with the parameters
        detector = cv2.SimpleBlobDetector_create(self.blob_params)

        # Detect blobs
        keypoints = detector.detect(self.processing_frame)
 
        # Draw detected blobs as red circles.
        self.processing_frame = cv2.drawKeypoints(self.processing_frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    def captureImage(self):
        # Open the video file
        video_capture = cv2.VideoCapture(self.camera_num)
        self.createParamtersWindow()

        # Check if the video file was successfully opened
        if not video_capture.isOpened():
            print("Error: Could not open video file.")
            exit()
        
        while True:
            # Read a frame from the video
            ret, frame = video_capture.read()            
            # Check if the frame was successfully read
            if not ret:
                break
            
            ######
            self.capture_frame = deepcopy(frame)
            self.processing_frame = deepcopy(frame)
            
            self.processing_frame = cv2.cvtColor(self.processing_frame, cv2.COLOR_BGR2GRAY)
            self.perspectiveTransform()
            self.detectBlob()
            ######

            cv2.imshow('Video Frame', self.capture_frame)
            cv2.imshow('Video Frame Process', self.processing_frame)

            # Check for the 'q' key to quit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            if cv2.waitKey(25) & 0xFF == ord('s'):
                self.saveParams()
                print("save params")
            
        # Release the video capture object and close the OpenCV windows
        video_capture.release()
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
    vt = VisualTactile()
    vt.captureImage()
    