import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\LENOVO\Downloads\logit classification.csv")

x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

# future scelling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(x_train,y_train)

y_pred  = knn_classifier.predict(x_test)

# confusion matrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# acuuracy
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

## report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)

# bias
bais = knn_classifier.score(x_test,y_test)
print(bais)

# varience
varience = knn_classifier.score(x_test,y_test)
print(varience)

# compute ROC and AUC
from sklearn.metrics import  roc_curve,roc_auc_score
y_prob= knn_classifier.predict_proba(x_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test,y_prob)
auc_score = roc_auc_score(y_test,y_pred)

print("AUC Score:",auc_score)

plt.figure()
plt.plot(fpr,tpr,color = 'blue',label='ROC curve (area = %0.2f)'%auc_score)
plt.plot([0,1],[0,1],color = 'gray',linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Recever Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid()
plt.show()

dataset1 = pd.read_csv(r"C:\Users\LENOVO\OneDrive\Desktop\final1.csv")
d2 = dataset1.copy()

dataset1 = dataset1.iloc[:,[3,4]].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
M = sc.fit_transform(dataset1)

y_pred = pd.DataFrame()

d2['y_pred1'] = knn_classifier.predict(M)

d2.to_csv('final1.csv')


