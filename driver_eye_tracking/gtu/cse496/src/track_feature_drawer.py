import csv
import matplotlib.pyplot as plt


class DrawFeature:
    def __init__(self):
        self.drawFeatures()

    def drawOnePerson(self, featureId, dirId, fileId):
        features = []
        filePath = "/home/omer/Desktop/driver_eye_tracking/gtu/cse496/features/" + str(dirId) + "/" + str(
            fileId) + ".csv"
        with open(filePath, 'r') as file:
            reader = csv.reader(file)
            index = 0
            for row in reader:
                if index == 0:
                    index += 1
                else:
                    features.append(float(row[featureId]))
        return features

    def drawFeatures(self):
        featureSize = 9
        x_labels = ["Blink Frames", "Blink Area Min", "Blink Area Avg", "Blink Area Max",
                    "Blink Begin To Min Time Avg", "Blink Begin To Min Time Max",
                    "Blink Min To End Time Min", "Blink Min To End Time Avg", "Blink Min To End Time Max"]
        labels = ["Not Sleepy", "Sleepy"]

        for directoryId in range(1, 13):
            y1 = []
            y2 = []

            for featureId in range(featureSize):
                features_0 = self.drawOnePerson(featureId, directoryId, 0)
                features_5 = self.drawOnePerson(featureId, directoryId, 5)
                features_10 = self.drawOnePerson(featureId, directoryId, 10)

                y1.append(features_0 + features_5)
                y2.append(features_10)

            fig, axs = plt.subplots(3, 3)

            bplot1 = axs[1, 0].boxplot([y1[0], y2[0]], notch=True,  # notch shape
                                       vert=True,  # vertical box alignment
                                       patch_artist=True,  # fill with color
                                       labels=labels
                                       )
            axs[1, 0].set_title(x_labels[0])
            axs[1, 0].set_ylabel("Total Frames")

            bplot2 = axs[0, 0].boxplot([y1[1], y2[1]], notch=True,  # notch shape
                                       vert=True,  # vertical box alignment
                                       patch_artist=True,  # fill with color
                                       labels=labels
                                       )
            axs[0, 0].set_title(x_labels[1])
            axs[0, 0].set_ylabel("Total Eye Aspect Ratio in Blink")

            bplot3 = axs[0, 1].boxplot([y1[2], y2[2]], notch=True,  # notch shape
                                       vert=True,  # vertical box alignment
                                       patch_artist=True,  # fill with color
                                       labels=labels
                                       )
            axs[0, 1].set_title(x_labels[2])
            axs[0, 1].set_ylabel("Total Eye Aspect Ratio in Blink")

            bplot4 = axs[0, 2].boxplot([y1[3], y2[3]], notch=True,  # notch shape
                                       vert=True,  # vertical box alignment
                                       patch_artist=True,  # fill with color
                                       labels=labels
                                       )
            axs[0, 2].set_title(x_labels[3])
            axs[0, 2].set_ylabel("Total Eye Aspect Ratio in Blink")

            bplot5 = axs[1, 1].boxplot([y1[4], y2[4]], notch=True,  # notch shape
                                       vert=True,  # vertical box alignment
                                       patch_artist=True,  # fill with color
                                       labels=labels
                                       )
            axs[1, 1].set_title(x_labels[4])
            axs[1, 1].set_ylabel("Second")

            bplot6 = axs[1, 2].boxplot([y1[5], y2[5]], notch=True,  # notch shape
                                       vert=True,  # vertical box alignment
                                       patch_artist=True,  # fill with color
                                       labels=labels
                                       )
            axs[1, 2].set_title(x_labels[5])
            axs[1, 2].set_ylabel("Second")

            bplot7 = axs[2, 0].boxplot([y1[6], y2[6]], notch=True,  # notch shape
                                       vert=True,  # vertical box alignment
                                       patch_artist=True,  # fill with color
                                       labels=labels
                                       )
            axs[2, 0].set_title(x_labels[6])
            axs[2, 0].set_ylabel("Second")

            bplot8 = axs[2, 1].boxplot([y1[7], y2[7]], notch=True,  # notch shape
                                       vert=True,  # vertical box alignment
                                       patch_artist=True,  # fill with color
                                       labels=labels
                                       )
            axs[2, 1].set_title(x_labels[7])
            axs[2, 1].set_ylabel("Second")

            bplot9 = axs[2, 2].boxplot([y1[8], y2[8]], notch=True,  # notch shape
                                       vert=True,  # vertical box alignment
                                       patch_artist=True,  # fill with color
                                       labels=labels
                                       )
            axs[2, 2].set_title(x_labels[8])
            axs[2, 2].set_ylabel("Second")

            colors = ['blue', 'green']
            for bplot in (bplot1, bplot2, bplot3, bplot4, bplot5,
                          bplot6, bplot7, bplot8, bplot9):
                for patch, color in zip(bplot["boxes"], colors):
                    patch.set_facecolor(color)

            for axs1 in axs:
                for axs2 in axs1:
                    axs2.yaxis.grid(True)

            fig.legend([bplot1["boxes"][0], bplot1["boxes"][1]], labels, loc='lower left')
            fig.tight_layout()
            plt.gcf().canvas.set_window_title("Person " + str(directoryId))
            fig.show()
        plt.show()


if __name__ == '__main__':
    DrawFeature()
