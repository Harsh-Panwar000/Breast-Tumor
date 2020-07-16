import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

#Read the data
df = pd.read_csv("breast-cancer-wisconsin.data", sep = ",")

#Drop unnecessary columns(ID removed as it serves no purpose in classification...Bare Nuclei removed due to numerous unrecorded instances)
df.drop(["ID Number", "Bare Nuclei"], axis=1, inplace=True)

#Create a numpy array for each column
Clump_Thickness = np.array(df["Clump Thickness"])
Uniformity_of_Cell_Size = np.array(df["Uniformity of Cell Size"])
Uniformity_of_Cell_Shape = np.array(df["Uniformity of Cell Shape"])
Marginal_Adhesion = np.array(df["Marginal Adhesion"])
Single_Epithelial_Cell_Size = np.array(df["Single Epithelial Cell Size"])
Bland_Chromatin = np.array(df["Bland Chromatin"])
Normal_Nucleoli = np.array(df["Normal Nucleoli"])
Mitoses = np.array(df["Mitoses"])
Class = np.array(df["Class"])

#Create the list for the data
x = list(zip(Clump_Thickness,Uniformity_of_Cell_Shape,Uniformity_of_Cell_Size,
             Marginal_Adhesion, Single_Epithelial_Cell_Size,
             Bland_Chromatin, Normal_Nucleoli, Mitoses))
y = list(Class)

#Create the train vs test sets from the data with a 10% test size
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y , test_size=.1)

#Train the model
model = KNeighborsClassifier(n_neighbors=7)
model.fit(x_train, y_train)

#Find acurracy
acc = model.score(x_test, y_test)
print(acc)

#Print the predicted vs Actual Values
predicted = model.predict(x_test)
for case in range(len(predicted)):
    print("Predicted: " , predicted[case] , " Data: " , x_test[case] , " Actual: " , y_test[case])

