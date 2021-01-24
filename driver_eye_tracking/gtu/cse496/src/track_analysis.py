import csv
import pandas as pd
from sklearn.metrics import classification_report


class TrackAnalysis:
    def __init__(self):
        import warnings
        warnings.filterwarnings('ignore')
        self.__featureHeader = []
        self.__data = []
        self.__filePath = "/home/omer/Desktop/driver_eye_tracking/gtu/cse496/features/"
        self.__directorySize = 13
        self.__sleepSituationSize = 3
        # self.getAllData()
        self.analysis()

    def getAllData(self):
        for directory in range(1, self.__directorySize):
            file = 0
            data = []

            for i in range(self.__sleepSituationSize):
                featuresOfPerson = []
                path = self.__filePath + str(directory) + "/" + str(file) + ".csv"
                with open(path, 'r') as file_p:
                    reader = csv.reader(file_p)
                    index = 0
                    for row in reader:
                        if index == 0:
                            row.insert(0, "class")
                            self.__featureHeader = row
                            index += 1
                        else:
                            if file == 10:
                                row.insert(0, str(10))
                            else:
                                row.insert(0, str(0))
                            featuresOfPerson.append(row)
                file += 5
                data.append(featuresOfPerson)
            self.__data.append(data)

            self.writeAllDataInCsv("all_features.csv")

    def writeAllDataInCsv(self, path):
        with open(self.__filePath + path, "w") as file:

            fileWriter = csv.writer(file)
            fileWriter.writerow(self.__featureHeader)

            for firstRow in self.__data:
                for secondRow in firstRow:
                    for thirdRow in secondRow:
                        fileWriter.writerow(thirdRow)

    def analysis(self):

        data = pd.read_csv(self.__filePath + "all_features.csv")

        from sklearn.model_selection import train_test_split
        training_set, test_set = train_test_split(data, test_size=0.5, random_state=1)

        X_train = training_set.iloc[:, 1:].values
        Y_train = training_set.iloc[:, 0].values
        X_test = test_set.iloc[:, 1:].values
        Y_test = test_set.iloc[:, 0].values

        from sklearn.svm import SVC
        classifier = SVC(kernel='linear', random_state=1)
        classifier.fit(X_train, Y_train)

        Y_pred = classifier.predict(X_test)

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(Y_test, Y_pred)
        tn, fp, fn, tp = cm.ravel()

        print("Confusion Matrix : ")
        print(cm)

        print("\nConfusion Matrix Values : ")
        print("TP : " + str(tp) + "\t\tFP : " + str(fp))
        print("FN : " + str(fn) + "\t\tTN : " + str(tn))

        print("\nAccuracy Evaluation : ")
        print("(TP + TN) / (TP + TN + FP + FN) = ", end="")
        print((tp + tn) / (tp + tn + fp + fn))

        print("\nClassification Report : ")
        print(classification_report(Y_test, Y_pred))

    def printData(self, trainOrTestData, trainOrTestStr):
        dirID = 1
        for dirTrainData in trainOrTestData:
            fileID = 0
            print("************************ " + trainOrTestStr + " " + str(dirID) + " ***************************")

            for fileTrainData in dirTrainData:
                print("----------- " + trainOrTestStr + " -----------")
                for data in fileTrainData:
                    print("[Dir " + str(dirID) + "] [File " + str(fileID) + "] -> " + str(data))
                fileID += 5
                print("-----------------------------")
            dirID += 1

            print("***********************************************************")


if __name__ == '__main__':
    TrackAnalysis()