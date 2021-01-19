import time
import cv2
import dlib
import csv
from threading import Thread, Lock
from imutils import face_utils
from scipy.spatial import distance as dist

from gtu.cse496.src.track_chart import TrackChart
from gtu.cse496.src.track_face_ui import TrackFaceUI


# TrackFace concrete class.
class TrackFace:

    # TrackFace constructor inits TrackFaceUI information and other variables.
    # It starts the tracking.
    def __init__(self):
        self.__face_landmarks = "dataset/landmarks/shape_predictor_68_face_landmarks.dat"

        self.__featureHeaders = ["frame_count", "area_min", "area_avg", "area_max",
                                 "beg_to_min_avg", "beg_to_min_max",
                                 "min_to_end_min", "min_to_end_avg", "min_to_end_max"]

        for i in range(1, 13):
            video = 0
            for j in range(3):
                self.__ui = TrackFaceUI()
                self.__chart = TrackChart()
                self.__video = str(video)
                video += 5
                self.__video_dir = str(i)
                self.__video_file = "dataset/videos/" + self.__video_dir + "/" + self.__video + ".mp4"
                self.__file = ""
                self.__fileWriter = ""

                self.__COUNTER = 0
                self.__TOTAL = 0
                self.__eye_threshold = 0
                self.__underThreshold = []
                self.__underThresholds = []
                self.__underThresholdAreas = []
                self.__underThresholdFrameCounts = []

                self.__underThresholdMin = 1
                self.__underThresholdTimeStart = 0
                self.__underThresholdTimeMidStart = 0
                self.__underThresholdTimeMidFlag = False
                self.__underThresholdTimeFlag = True
                self.__underThresholdFirstPeriodTimes = []
                self.__underThresholdSecondPeriodTimes = []

                self.__featureTimer = time.time()

                self.__featuresFrameCounts = []

                self.__featuresAreaMin = []
                self.__featuresAreaAvg = []
                self.__featuresAreaMax = []

                self.__featuresBegToMinAvg = []
                self.__featuresBegToMinMax = []

                self.__featuresMinToEndMin = []
                self.__featuresMinToEndAvg = []
                self.__featuresMinToEndMax = []

                self.__allEyeRatios = []
                self.__canCountBlink = False

                self.scanFaceMutex = Lock()

                self.__startTrack()

    # Starts to tracking.
    def __startTrack(self):

        # Thread for scan face for 60 seconds to set eye threshold.
        Thread(target=self.scanFace60Sec, args=[]).start()

        # Inits capture, face detector and shape predictor.
        cap = cv2.VideoCapture(self.__video_file)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(self.__face_landmarks)

        # For 10 minutes it will read video file.
        finishTrackingTime = time.time() + 700

        # Starting to tracking.
        while time.time() < finishTrackingTime:
            faces, frame, gray = self.read_and_detect_face(cap, detector)

            # Is exit keys?
            if self.__ui.isKeyEnterOrESC():
                break

            if faces:
                face = faces[0]

                # Gets numpy landmarks and evaluates eye aspect ratio.
                landmarks = self.getLandmarks(gray, face, predictor)
                ear, leftEye, rightEye = self.evaluate_ear_getEyes(landmarks)

                # Inserting ear into chart.
                self.__chart.insertTrackChart(ear)

                # UI draws eye contours.
                self.__ui.drawEyesContours(frame, leftEye, rightEye)

                # If scanning face in 60 seconds is done then can count blinks.
                if self.__canCountBlink:
                    self.countEyeBlinks(ear)
                    self.__ui.printBlink(frame, self.__TOTAL)
                    self.__ui.printThreshold(frame)

                    # Checks if the eyes are closed.
                    self.isUnderEyeThreshold(ear)

                    time_dist = time.time() - self.__featureTimer
                    print("ZAMAN FARKI : " + str(time_dist))
                    if time_dist >= 60:

                        if len(self.__underThresholdFrameCounts) == 0:
                            self.__underThresholdFrameCounts.append(0)
                        if len(self.__underThresholdAreas) == 0:
                            self.__underThresholdAreas.append(0)
                        if len(self.__underThresholdFirstPeriodTimes) == 0:
                            self.__underThresholdFirstPeriodTimes.append(0)
                        if len(self.__underThresholdSecondPeriodTimes) == 0:
                            self.__underThresholdSecondPeriodTimes.append(0)

                        self.__featuresFrameCounts.append(sum(self.__underThresholdFrameCounts))

                        self.__featuresAreaMin.append(min(self.__underThresholdAreas))
                        self.__featuresAreaAvg.append(sum(self.__underThresholdAreas) / len(self.__underThresholdAreas))
                        self.__featuresAreaMax.append(max(self.__underThresholdAreas))

                        self.__featuresBegToMinAvg.append(sum(self.__underThresholdFirstPeriodTimes) / len(self.__underThresholdFirstPeriodTimes))
                        self.__featuresBegToMinMax.append(max(self.__underThresholdFirstPeriodTimes))

                        self.__featuresMinToEndMin.append(min(self.__underThresholdSecondPeriodTimes))
                        self.__featuresMinToEndAvg.append(sum(self.__underThresholdSecondPeriodTimes) / len(self.__underThresholdSecondPeriodTimes))
                        self.__featuresMinToEndMax.append(max(self.__underThresholdSecondPeriodTimes))

                        self.__underThreshold = []
                        self.__underThresholdFrameCounts = []
                        self.__underThresholdAreas = []
                        self.__underThresholdFirstPeriodTimes = []
                        self.__underThresholdSecondPeriodTimes = []
                        self.__featureTimer = time.time()
                else:
                    # Inserting the eye ratios in 60 seconds.
                    self.insertEyeRatiosInSecond(ear)
                    self.__featureTimer = time.time()

                # UI prints ratio, time and show image in real time.
                self.__ui.printRatio(frame, ear)
                self.__ui.printTime(frame)
                self.__ui.showImage(frame)

        self.__ui.con_thread = False

        # Write features to csv file.
        self.writeToCsv()

        # After execution release cap and destroy windows.
        cap.release()
        cv2.destroyAllWindows()

    # It checks if eye aspect ratio is under eye threshold.
    # If it is under threshold saving it.
    # If the length of under threshold greater than eye close threshold it prints that information.
    # It finds blink times and blinking frames.
    def isUnderEyeThreshold(self, ear):

        # If eye aspect ratio is under threshold then insert it to underThreshold.
        if ear < self.__eye_threshold:
            self.__underThreshold.append(ear)

            # Find the blink time first and second periods.
            self.findBlinkTimeStartingAndMiddlePoints(ear)
        else:

            # If eye aspect ratio is over threshold then evaluate the blink time periods
            #   and evaluate the blinking areas with blinking frames.
            self.evaluateBlinkTimePeriods()
            self.evaluateBlinkAreasAndBlinkFrames()

    # Scans face in 60 seconds and evaluates the max and min ratios in that frames.
    # Evaluates the average eye aspect ratio threshold and sets it.
    def scanFace60Sec(self):
        end = time.time() + 60
        while time.time() < end and self.__ui.con_thread:
            time.sleep(0.05)

        if self.__ui.con_thread:
            self.scanFaceMutex.acquire()
            newThresh = sum(self.__allEyeRatios)/len(self.__allEyeRatios) - 0.04
            self.scanFaceMutex.release()

            self.__eye_threshold = newThresh
            self.__ui.setThresh(newThresh)
            self.__chart.setEyeThresh(newThresh)
            self.__canCountBlink = True

    # Evaluates eye aspect ration and returns with eyes.
    def evaluate_ear_getEyes(self, landmarks):
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        leftEye = landmarks[lStart:lEnd]
        rightEye = landmarks[rStart:rEnd]

        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0
        return ear, leftEye, rightEye

    # Getter for landmarks as numpy array.
    def getLandmarks(self, gray, face, predictor):
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        return landmarks

    # Returns eye aspect ratio of eye.
    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    # Reads capture as frame and gets grayscale of frame.
    # Returns the faces using detector and frames.
    def read_and_detect_face(self, cap, detector):
        ret, frame = cap.read()
        if not ret:
            return None, None, None
        # frame = cv2.resize(frame, self.__ui.getScreenSize())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        return faces, frame, gray

    # Counts eye blinks if the each eyes are blinked.
    def countEyeBlinks(self, ear):
        if ear < self.__ui.getEyeThresh():
            self.__COUNTER += 1
        else:
            if self.__COUNTER >= self.__ui.getEyeConFrames():
                self.__TOTAL += 1
            self.__COUNTER = 0

    # Saves eye ratio in a second.
    def insertEyeRatiosInSecond(self, ear):
        self.scanFaceMutex.acquire()
        if ear != 0:
            self.__allEyeRatios.append(ear)
        self.scanFaceMutex.release()

    # Evaluates blink areas and blink frames.
    def evaluateBlinkAreasAndBlinkFrames(self):

        # If there is an filled list then sum the eye aspect ratios it represents the blink area
        #   and length of that list represents the how many frames are in that blink.
        if self.__underThreshold:
            blinkArea = sum(self.__underThreshold)
            frameCount = len(self.__underThreshold)
            print("BLINK FRAME SIZE : " + str(frameCount) + " BLINK AREA : {:.3f}\n".format(blinkArea))
            self.__underThresholds.append(self.__underThreshold)
            self.__underThresholdAreas.append(blinkArea)
            self.__underThresholdFrameCounts.append(frameCount)
            self.__underThreshold = []

    # Finds starting time points of blinking.
    def findBlinkTimeStartingAndMiddlePoints(self, ear):

        # For to find begin time point of blink.
        if self.__underThresholdTimeFlag:
            self.__underThresholdTimeStart = time.time()
            self.__underThresholdTimeFlag = False

        # Finding the minimum point of blink and sets the end point of first period with begin point of second period.
        if ear < self.__underThresholdMin:
            self.__underThresholdMin = ear
            underThresholdStartTimeEnd = time.time()
            underThresholdFirstPeriodTime = underThresholdStartTimeEnd - self.__underThresholdTimeStart
            self.__underThresholdFirstPeriodTimes.append(underThresholdFirstPeriodTime)
            self.__underThresholdTimeMidStart = underThresholdStartTimeEnd
            self.__underThresholdTimeMidFlag = True

    # Evaluates blink time periods.
    def evaluateBlinkTimePeriods(self):

        # If blinking is done then evaluates the second period time and prints all periods together.
        if self.__underThresholdTimeMidFlag:
            underThresholdMidTimeEnd = time.time()
            underThresholdSecondPeriodTime = underThresholdMidTimeEnd - self.__underThresholdTimeMidStart
            self.__underThresholdSecondPeriodTimes.append(underThresholdSecondPeriodTime)
            self.__underThresholdTimeMidFlag = False
            self.__underThresholdTimeFlag = True
            self.__underThresholdMin = 1

    def writeToCsv(self):
        self.__file = open("features/" + self.__video_dir + "/" + self.__video + ".csv", 'w')
        with self.__file:
            self.__fileWriter = csv.writer(self.__file)
            self.__fileWriter.writerow(self.__featureHeaders)

            for i in range(len(self.__featuresFrameCounts)):
                self.__fileWriter.writerow(["{:d}".format(self.__featuresFrameCounts[i]),
                                            "{:.3f}".format(self.__featuresAreaMin[i]),
                                            "{:.3f}".format(self.__featuresAreaAvg[i]),
                                            "{:.3f}".format(self.__featuresAreaMax[i]),
                                            "{:.3f}".format(self.__featuresBegToMinAvg[i]),
                                            "{:.3f}".format(self.__featuresBegToMinMax[i]),
                                            "{:.3f}".format(self.__featuresMinToEndMin[i]),
                                            "{:.3f}".format(self.__featuresMinToEndAvg[i]),
                                            "{:.3f}".format(self.__featuresMinToEndMax[i])])
