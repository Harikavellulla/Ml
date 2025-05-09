import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\LENOVO\Downloads\logit classification.csv")

x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size = 0.25,random_state=0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

'''from sklearn.preprocessing import Normalizer
sc = Normalizer()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)'''

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print(ac)

from sklearn.metrics import classification_report
cr = classification_report(y_test,y_pred)
print(cr)
# bais
bias = classifier.score(x_test,y_test)
print(bias)
# varience
variance = classifier.score(x_test,y_test)
print(variance)

variance = classifier.score(x_train,y_train)
print(variance)
# compute ROC and AUC
from sklearn.metrics import  roc_curve,roc_auc_score
y_prob= classifier.predict_proba(x_test)[:,1]

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
dataset1=dataset1.iloc[:,[3,4]].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
M = sc.fit_transform(dataset1)

y_pred1 = pd.DataFrame()

d2 ['Y_pred1'] = classifier.predict(M)

d2.to_csv('final.csv')





