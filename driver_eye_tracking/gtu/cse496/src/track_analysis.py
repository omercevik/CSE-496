import csv
import pandas as pd


class TrackAnalysis:
    def __init__(self):
        import warnings
        warnings.filterwarnings('ignore')
        self.__featureHeader = []
        self.__data = []
        self.__filePath = "/home/omer/Desktop/driver_eye_tracking/gtu/cse496/features/"
        self.__directorySize = 13
        self.__sleepSituationSize = 3
        self.getAllData()
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

    def analysis(self):

        data = []
        for firstRow in self.__data:
            for secondRow in firstRow:
                for thirdRow in secondRow:
                    print(thirdRow)
                    data.append(thirdRow)
        data = pd.DataFrame(data)
        data.columns = self.__featureHeader

        print(data.shape)
        print(data)

        X = data.drop('class', axis=1)
        y = data['class']

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50)

        from sklearn.svm import SVC
        svclassifier = SVC(kernel='linear')
        svclassifier.fit(X_train, y_train)

        y_pred = svclassifier.predict(X_test)

        from sklearn.metrics import classification_report, confusion_matrix
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        from sklearn import metrics

        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    TrackAnalysis()
