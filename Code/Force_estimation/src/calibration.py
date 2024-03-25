import cv2
import numpy as np
from copy import deepcopy 
import json

class VisualTactile:
    def __init__(self):
        self.camera_num = "/dev/video4" 

        self.capture_frame = None
        self.processing_frame = None

        # self.TRANSFORM_HEIGHT = int(67.8*7)
        # self.TRANSFORM_WIDTH = int(78.0*7)
        self.TRANSFORM_HEIGHT = 640
        self.TRANSFORM_WIDTH = 640
        self.original_points = np.zeros((4,2), dtype=np.float32)

        self.ORIGINAL_HEIGHT = 480
        self.ORIGINAL_WIDTH = 640

        self.blob_params = cv2.SimpleBlobDetector_Params()
        
        self.JSON_PARAMS_PATH = "../params.json"

        self.is_calibrate_perspective = True

        self.is_record = False

        if self.is_record == True:
            self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            self.out = cv2.VideoWriter('output_video.mp4', self.fourcc, fps, (self.TRANSFORM_WIDTH, self.TRANSFORM_HEIGHT))

    def perspectiveTransform(self):
        # Define the four corners of the region of interest (ROI) in the original image
        output_points = np.float32([[0, 0], [self.TRANSFORM_WIDTH, 0], [self.TRANSFORM_WIDTH, self.TRANSFORM_HEIGHT], [0, self.TRANSFORM_HEIGHT]])
        
        # Compute the perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(self.original_points, output_points)

        self.processing_frame = cv2.warpPerspective(self.processing_frame, matrix, (self.TRANSFORM_WIDTH, self.TRANSFORM_WIDTH))

        if self.is_calibrate_perspective == True:
            cv2.polylines(self.capture_frame, [np.int32(self.original_points)], isClosed=True, color=(0, 255, 0), thickness=1)

    def setInitialParams(self):
        with open(self.JSON_PARAMS_PATH, 'r') as f:
            params = json.load(f)

        cv2.setTrackbarPos('Min Threshold', 'Blob Parameters', int(params["blob"]["minThreshold"]))
        cv2.setTrackbarPos('Max Threshold', 'Blob Parameters', int(params["blob"]["maxThreshold"]))
        cv2.setTrackbarPos('Min Area', 'Blob Parameters', int(params["blob"]["minArea"] - 0.1))
        cv2.setTrackbarPos('Max Area', 'Blob Parameters', int(params["blob"]["maxArea"] - 0.1))
        cv2.setTrackbarPos('Min Circularity', 'Blob Parameters', int((params["blob"]["minCircularity"] - 0.01) * 100))
        cv2.setTrackbarPos('Min Convexity', 'Blob Parameters', int((params["blob"]["minConvexity"] - 0.01) * 100))
        cv2.setTrackbarPos('Min Inertia', 'Blob Parameters', int((params["blob"]["minInertiaRatio"] - 0.01) * 100))
    
        cv2.setTrackbarPos('x1', 'Perspective Transform Parameters', int(params["perspective_transform"][0][0]))
        cv2.setTrackbarPos('y1', 'Perspective Transform Parameters', int(params["perspective_transform"][0][1]))
        cv2.setTrackbarPos('x2', 'Perspective Transform Parameters', int(params["perspective_transform"][1][0]))
        cv2.setTrackbarPos('y2', 'Perspective Transform Parameters', int(params["perspective_transform"][1][1]))
        cv2.setTrackbarPos('x3', 'Perspective Transform Parameters', int(params["perspective_transform"][2][0]))
        cv2.setTrackbarPos('y3', 'Perspective Transform Parameters', int(params["perspective_transform"][2][1]))
        cv2.setTrackbarPos('x4', 'Perspective Transform Parameters', int(params["perspective_transform"][3][0]))
        cv2.setTrackbarPos('y4', 'Perspective Transform Parameters', int(params["perspective_transform"][3][1]))

        cv2.setTrackbarPos('Blur', 'Blob Parameters', int(params["processing_parameter"]["blur_filter"]))

    def updateParamsBlob(self):
        # Get current trackbar positions and Update blob detector parameters
        self.blob_params.minThreshold = cv2.getTrackbarPos('Min Threshold', 'Blob Parameters')
        self.blob_params.maxThreshold = cv2.getTrackbarPos('Max Threshold', 'Blob Parameters')
        self.blob_params.minArea = cv2.getTrackbarPos('Min Area', 'Blob Parameters') + 0.1
        self.blob_params.maxArea = cv2.getTrackbarPos('Max Area', 'Blob Parameters') + 0.1 
        self.blob_params.minCircularity = cv2.getTrackbarPos('Min Circularity', 'Blob Parameters') / 100 + 0.01
        self.blob_params.minConvexity = cv2.getTrackbarPos('Min Convexity', 'Blob Parameters') / 100  + 0.01
        self.blob_params.minInertiaRatio = cv2.getTrackbarPos('Min Inertia', 'Blob Parameters') / 100 + 0.01

    def updateParamsPerspective(self):
        # Get current trackbar positions and Update blob detector parameters
        self.original_points[0][0] = cv2.getTrackbarPos('x1', 'Perspective Transform Parameters')
        self.original_points[0][1] = cv2.getTrackbarPos('y1', 'Perspective Transform Parameters')
        self.original_points[1][0] = cv2.getTrackbarPos('x2', 'Perspective Transform Parameters')
        self.original_points[1][1] = cv2.getTrackbarPos('y2', 'Perspective Transform Parameters')
        self.original_points[2][0] = cv2.getTrackbarPos('x3', 'Perspective Transform Parameters')
        self.original_points[2][1] = cv2.getTrackbarPos('y3', 'Perspective Transform Parameters')
        self.original_points[3][0] = cv2.getTrackbarPos('x4', 'Perspective Transform Parameters')
        self.original_points[3][1] = cv2.getTrackbarPos('y4', 'Perspective Transform Parameters')
        
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

            params["perspective_transform"] = self.original_points.tolist()

        with open(self.JSON_PARAMS_PATH, 'w') as f:
            json.dump(params, f, indent=4)
    
    def createBlobWindow(self):
        cv2.namedWindow('Blob Parameters')
        # Create trackbars for adjusting parameters
        cv2.createTrackbar('Min Threshold', 'Blob Parameters', 0, 500, lambda x : self.updateParamsBlob())
        cv2.createTrackbar('Max Threshold', 'Blob Parameters', 0, 500, lambda x : self.updateParamsBlob())
        cv2.createTrackbar('Min Area', 'Blob Parameters', 0, 600, lambda x : self.updateParamsBlob())
        cv2.createTrackbar('Max Area', 'Blob Parameters', 0, 600, lambda x : self.updateParamsBlob())
        cv2.createTrackbar('Min Circularity', 'Blob Parameters', 0, 100, lambda x : self.updateParamsBlob())
        cv2.createTrackbar('Min Convexity', 'Blob Parameters', 0, 100, lambda x : self.updateParamsBlob())
        cv2.createTrackbar('Min Inertia', 'Blob Parameters', 0, 100, lambda x : self.updateParamsBlob())

    def createPerspectiveWindow(self):
        cv2.namedWindow('Perspective Transform Parameters')
        # Create trackbars for adjusting parameters
        cv2.createTrackbar('x1', 'Perspective Transform Parameters', 0, self.ORIGINAL_WIDTH, lambda x : self.updateParamsPerspective())
        cv2.createTrackbar('y1', 'Perspective Transform Parameters', 0, self.ORIGINAL_HEIGHT, lambda x : self.updateParamsPerspective())
        cv2.createTrackbar('x2', 'Perspective Transform Parameters', 0, self.ORIGINAL_WIDTH, lambda x : self.updateParamsPerspective())
        cv2.createTrackbar('y2', 'Perspective Transform Parameters', 0, self.ORIGINAL_HEIGHT, lambda x : self.updateParamsPerspective())
        cv2.createTrackbar('x3', 'Perspective Transform Parameters', 0, self.ORIGINAL_WIDTH, lambda x : self.updateParamsPerspective())
        cv2.createTrackbar('y3', 'Perspective Transform Parameters', 0, self.ORIGINAL_HEIGHT, lambda x : self.updateParamsPerspective())
        cv2.createTrackbar('x4', 'Perspective Transform Parameters', 0, self.ORIGINAL_WIDTH, lambda x : self.updateParamsPerspective())
        cv2.createTrackbar('y4', 'Perspective Transform Parameters', 0, self.ORIGINAL_HEIGHT, lambda x : self.updateParamsPerspective())

    # def createPerspectiveWindow(self):


    def createParamtersWindow(self):
        self.createBlobWindow()
        self.createPerspectiveWindow()

        self.setInitialParams()

    def preprocessing(self):
        self.processing_frame = cv2.blur(self.processing_frame,(5,5))

    def detectBlob(self):
        # Create a blob detector with the parameters
        detector = cv2.SimpleBlobDetector_create(self.blob_params)

        # Detect blobs
        keypoints = detector.detect(self.processing_frame)
 
        # Draw detected blobs as red circles.
        self.processing_frame = cv2.drawKeypoints(self.processing_frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # for kp in keypoints:
        #     x, y = kp.pt
        #     cv2.circle(self.processing_frame, (int(x), int(y)), 3, (0, 0, 255), -1)  # draw a dot at the keypoint position

    def captureImage(self):
        # Open the video file
        video_capture = cv2.VideoCapture(self.camera_num)
        self.createParamtersWindow()
        if self.is_calibrate_perspective == False:
            cv2.destroyWindow("Perspective Transform Parameters")

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
            if self.is_calibrate_perspective == True:
                self.capture_frame = deepcopy(frame)
                self.processing_frame = deepcopy(frame)
                self.perspectiveTransform()
            else:
                self.processing_frame = deepcopy(frame)
                self.perspectiveTransform()
                self.capture_frame = deepcopy(self.processing_frame)
            
            if self.is_record:
                self.out.write(self.capture_frame)

            self.processing_frame = cv2.cvtColor(self.processing_frame, cv2.COLOR_BGR2GRAY)
            self.preprocessing()
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
        if self.is_record:
            self.out.release()
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
    vt = VisualTactile()
    vt.captureImage()
    