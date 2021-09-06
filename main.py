import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('creditcard_1.csv')
data.head()
data.info()

#Checking missing values
data.isnull().sum()
data.describe()

#Data visualization
# number of fraud and valid transactions 
count_classes = pd.value_counts(data['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction Distribution")
plt.xlabel("Class")
plt.ylabel("Frequency");


#Assigning the transaction class "0 = NORMAL  & 1 = FRAUD"
Normal = data[data['Class']==0]
Fraud = data[data['Class']==1]
outlier_fraction = len(Fraud)/float(len(Normal))
print()
print(outlier_fraction)
print("Fraud Cases : {}".format(len(Fraud)))
print("Valid Cases : {}".format(len(Normal)))


#Training and testing sets
X = data.iloc[:,:-1] 
y = data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 0)


"ANN"

from keras.models import Sequential
from keras.layers import Dense

print("ANN")
print()
classifier = Sequential()

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 20, epochs = 5,verbose = 1)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

"Analysis Report"
print()
print("------Classification Report------")
print(classification_report(y_pred,y_test))

print()
print("------Accuracy------")
print(f"The Accuracy Score :{(accuracy_score(y_pred,y_test)*100)}")
print()