import cv2
import numpy as np
from copy import deepcopy 
import json
from lib import find_marker


class VisualTactile:
    def __init__(self):
        self.video_capture = "/dev/video4"
        # self.video_capture = "../video/marker_stable_2mm_1.mp4"


        self.capture_frame = None
        self.processing_frame = None

        self.JSON_PARAMS_PATH = "../params.json"
        

        self.blob_params = cv2.SimpleBlobDetector_Params()

        self.is_perspective_transform = True
        self.is_record = False
        self.record_window_name = ""

        if self.is_record:
            self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            self.out = cv2.VideoWriter('output_video.mp4', self.fourcc, fps, (self.TRANSFORM_WIDTH, self.TRANSFORM_HEIGHT))
        
        self.loadParams()
        self.loadMarkerTracker()

    def loadMarkerTracker(self):
        self.marker_tracker = find_marker.Matching(
            N_=8, 
            M_=10, 
            fps_=20, 
            x0_=153.6230010986328, 
            y0_=45.08559036254883,
            dx_=60, 
            dy_=76)


    def loadParams(self):
        with open(self.JSON_PARAMS_PATH, 'r') as f:
            params = json.load(f)
            self.TRANSFORM_HEIGHT = params["frame_dimension"]["height"]
            self.TRANSFORM_WIDTH = params["frame_dimension"]["width"]

            self.blob_params.minThreshold = params["blob"]["minThreshold"] 
            self.blob_params.maxThreshold = params["blob"]["maxThreshold"]
            self.blob_params.minArea = params["blob"]["minArea"]
            self.blob_params.maxArea = params["blob"]["maxArea"]
            self.blob_params.minCircularity = params["blob"]["minCircularity"] 
            self.blob_params.minConvexity = params["blob"]["minConvexity"]  
            self.blob_params.minInertiaRatio = params["blob"]["minInertiaRatio"]  

            self.original_points = np.array(params["perspective_transform"], dtype=np.float32)

            self.blur_filter_dim = params["processing_parameter"]["blur_filter"]

    def perspectiveTransform(self):
        # Define the four corners of the region of interest (ROI) in the original image
        output_points = np.float32([[0, 0], [self.TRANSFORM_WIDTH, 0], [self.TRANSFORM_WIDTH, self.TRANSFORM_HEIGHT], [0, self.TRANSFORM_HEIGHT]])
        
        # Compute the perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(self.original_points, output_points)
        self.processing_frame = cv2.warpPerspective(self.processing_frame, matrix, (self.TRANSFORM_WIDTH, self.TRANSFORM_WIDTH))

    def preprocessing(self):
        self.processing_frame = cv2.blur(self.processing_frame,(self.blur_filter_dim, self.blur_filter_dim))

    def detectBlob(self):
        # Create a blob detector with the parameters
        detector = cv2.SimpleBlobDetector_create(self.blob_params)

        # Detect blobs
        keypoints = detector.detect(self.processing_frame)
        
        # Draw detected blobs as red circles.
        self.processing_frame = cv2.drawKeypoints(self.processing_frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        return keypoints

    def getCenterFromKeypoints(self, keypoints):
        centers = []
        for kp in keypoints:
            centers.append(list(kp.pt))
        return centers

    def getMarkerTrackParams(self, centers):
        min_dist = centers[0][0]**2 + centers[0][1]**2
        for i in range(len(centers)):
            k = centers[i][0]**2 + centers[i][1]**2
            if k < min_dist:
                min_dist = k
        
        k = np.sum((np.array(centers) - np.array(centers[i]))**2, axis=1)
        idxs = np.argsort(k)
        print(centers[i])
        for j in range(1,10):
            val_1 = centers[idxs[j]]
            print(np.array(val_1) - np.array(centers[i]))
        print("---")


    def drawFlow(self, frame, flow):
        Ox, Oy, Cx, Cy, Occupied = flow
        K = 0
        for i in range(len(Ox)):
            for j in range(len(Ox[i])):
                pt1 = (int(Ox[i][j]), int(Oy[i][j]))
                pt2 = (int(Cx[i][j] + K * (Cx[i][j] - Ox[i][j])), int(Cy[i][j] + K * (Cy[i][j] - Oy[i][j])))
                color = (0, 0, 255)
                if Occupied[i][j] <= -1:
                    color = (127, 127, 255)
                cv2.arrowedLine(frame, pt1, pt2, color, 2,  tipLength=0.2)


    def captureImage(self):
        # Open the video file
        video_capture = cv2.VideoCapture(self.video_capture)

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
            self.processing_frame = deepcopy(frame)

            if self.is_perspective_transform:
                self.perspectiveTransform()

            self.capture_frame = deepcopy(self.processing_frame)
            
            if self.is_record:
                self.out.write(self.capture_frame)

            self.processing_frame = cv2.cvtColor(self.processing_frame, cv2.COLOR_BGR2GRAY)
            self.preprocessing()
            keypoints = self.detectBlob()
            centers = self.getCenterFromKeypoints(keypoints)
            # self.getMarkerTrackParams(centers)
            self.marker_tracker.init(centers)
            self.marker_tracker.run()
            flow = self.marker_tracker.get_flow()
            # print(self.flow)

            frame2 = 255*np.ones((self.TRANSFORM_WIDTH, self.TRANSFORM_HEIGHT, 3))
            for x,y in centers:
                cv2.circle(frame2, (int(x), int(y)), 3, (0, 0, 255), -1)
            self.drawFlow(frame2, flow) 

            cv2.imshow('Video Frame Marker', frame2)

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
    