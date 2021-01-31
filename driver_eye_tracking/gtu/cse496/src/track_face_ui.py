import cv2
import threading
import time


# TrackFaceUI concrete class
class TrackFaceUI:

    # TrackFaceUI constructor inits UI information and starts to count time.
    def __init__(self):
        self.__FONT = cv2.FONT_HERSHEY_COMPLEX
        self.__COLOR_BLUE = (255, 0, 0)  # BLUE
        self.__COLOR_RED = (0, 0, 255)  # RED
        self.__COLOR_GREEN = (0, 255, 0)  # GREEN
        self.EYE_AR_THRESH = 0
        self.__EYE_AR_CON_FRAMES = 1
        self.__SCREEN_SIZE = (800, 600)
        self.__TIME_LABEL_LOC = (30, 30)
        self.__RATIO_LABEL_LOC = (30, 60)
        self.__BLINK_LABEL_LOC = (30, 90)
        self.__THRESH_LABEL_LOC = (30, 120)
        self.__SLEEP_STATUS_LABEL_LOC = (30, 150)
        self.__WINDOW_NAME = "Driver Eye Tracking"
        self.__WINDOW_POSITION = (0, 0)
        self.seconds = 0
        self.con_thread = True

        threading.Thread(target=self.countTimeThread).start()

    # Thread for counting time.
    def countTimeThread(self):
        now = time.time()
        while self.con_thread:
            time.sleep(1)
            cur = time.time()
            dif = cur - now

            self.seconds = int(dif)

    # Setter for threshold.
    def setThresh(self, thresh):
        self.EYE_AR_THRESH = thresh

    # Draws eyes' contours on frame.
    def drawEyesContours(self, frame, leftEye, rightEye):
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(frame, [leftEyeHull], -1, self.__COLOR_BLUE, 1)
        cv2.drawContours(frame, [rightEyeHull], -1, self.__COLOR_BLUE, 1)

    # Prints blinks on frame.
    def printBlink(self, frame, total):
        cv2.putText(frame, "Blinks : " + str(total), self.__BLINK_LABEL_LOC,
                    self.__FONT, 1, self.__COLOR_BLUE, 1)

    # Prints eye aspect ratio on frame.
    def printRatio(self, frame, ear):
        cv2.putText(frame, "Ratio  : {:.2f}".format(ear), self.__RATIO_LABEL_LOC,
                    self.__FONT, 1, self.__COLOR_BLUE, 1)

    # Prints time on frame.
    def printTime(self, frame):
        cv2.putText(frame, "Time  : " + str(self.seconds), self.__TIME_LABEL_LOC,
                    self.__FONT, 1, self.__COLOR_BLUE, 1)

    # Prints threshold on frame.
    def printThreshold(self, frame):
        cv2.putText(frame, "Threshold : {:.2f}".format(self.EYE_AR_THRESH), self.__THRESH_LABEL_LOC,
                    self.__FONT, 1, self.__COLOR_BLUE, 1)

    # Prints sleep status on frame.
    def printSleepStatus(self, frame, sleep_status):
        if sleep_status == "Sleepy":
            cv2.putText(frame, "STATUS : " + sleep_status, self.__SLEEP_STATUS_LABEL_LOC,
                        self.__FONT, 1, self.__COLOR_RED, 1)
        else:
            cv2.putText(frame, "STATUS : " + sleep_status, self.__SLEEP_STATUS_LABEL_LOC,
                        self.__FONT, 1, self.__COLOR_GREEN, 1)

    # Shows image on frame.
    def showImage(self, frame):
        cv2.imshow(self.__WINDOW_NAME, frame)

    # Checks the exit key is entered.
    def isKeyEnterOrESC(self):
        key = cv2.waitKey(1)
        isClose = key == 27 or key == 13
        self.con_thread = not isClose
        return isClose

    # Getter for font.
    def getFont(self):
        return self.__FONT

    # Getter for yellow color.
    def getColorYellow(self):
        return self.__COLOR_BLUE

    # Getter for eye threshold.
    def getEyeThresh(self):
        return self.EYE_AR_THRESH

    # Getter for eye con frames.
    def getEyeConFrames(self):
        return self.__EYE_AR_CON_FRAMES

    # Getter for screen size.
    def getScreenSize(self):
        return self.__SCREEN_SIZE

    # Getter for blink label location.
    def getBlinkLabelLoc(self):
        return self.__BLINK_LABEL_LOC

    # Getter for eye aspect ratio label location.
    def getRatioLabelLoc(self):
        return self.__RATIO_LABEL_LOC

    # Getter for window name.
    def getWindowName(self):
        return self.__WINDOW_NAME

    # Getter for window position.
    def getWindowPosition(self):
        return self.__WINDOW_POSITION

    def setConThread(self, param):
        self.con_thread = param
