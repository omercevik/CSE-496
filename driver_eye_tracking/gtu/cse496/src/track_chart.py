import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


# TrackChart concrete class
class TrackChart:

    # TrackChart constructor. Inits chart and shows it.
    def __init__(self):
        self.__ears = []
        self.__eye_thresh = 0
        self.__MAX_VALUES = 150
        self.__start = 0
        self.__end = self.__MAX_VALUES
        self.__isScanned = False
        self.__WINDOW_NAME = 'Real Time Eye Aspect Ratio Graph'
        self.__EYE_THRESH_COLOR = 'r'
        self.__LINE_COLOR = 'b'
        self.__BG_COLOR = (0, 0, 0)

        self.initPlt()
        plt.ion()
        plt.show()

    # Sets the eye threshold and enables the scan.
    def setEyeThresh(self, value):
        self.__eye_thresh = value
        self.__isScanned = True

    # Inits plot and refreshes figure.
    def initPlt(self):
        plt.clf()
        plt.xlim(0, self.__MAX_VALUES)
        plt.ylim(0, 1)
        plt.xlabel('Time')
        plt.ylabel('Eye Aspect Ratio')
        plt.title('Graph')
        plt.gcf().canvas.set_window_title(self.__WINDOW_NAME)
        plt.gca().set_facecolor(self.__BG_COLOR)
        self.plotEyeThreshold()

    # Inserts eye aspect ratio into list and inserts to chart.
    def insertTrackChart(self, value):
        self.__ears.append(value)
        if len(self.__ears) == self.__end:
            self.initPlt()
            self.__start += self.__MAX_VALUES
            self.__end += self.__MAX_VALUES
        self.plotPoints()
        self.pausePlots()

    # Plots the partial array of eye aspect ratios.
    def plotPoints(self):
        draw = self.__ears[self.__start:len(self.__ears)]
        plt.plot(draw, self.__LINE_COLOR)
        if self.__isScanned:
            self.plotEyeThreshold()
            self.__isScanned = False

    # Plots the eye threshold.
    def plotEyeThreshold(self):
        if self.__eye_thresh != 0:
            plt.plot([self.__eye_thresh]*self.__MAX_VALUES, self.__EYE_THRESH_COLOR)

    # Pauses plots for 0.00001 seconds.
    def pausePlots(self):
        plt.pause(0.00001)
