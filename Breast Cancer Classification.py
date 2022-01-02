#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


cellData = pd.read_csv("bcancer.csv")


# In[4]:


cellData.drop(["Unnamed: 32", "id"], axis=1, inplace=True)


# In[5]:


cellData.head()


# In[6]:


# DATA EXPLORATION 
#
#
#
#
#
cellData.info()


# In[7]:


#Checking for the missing values
cellData.isnull().sum()


# In[8]:


cellData.isnull().sum().plot(kind='bar', color = 'red')


# In[9]:


cellData.describe()


# In[10]:


cellData['diagnosis'].value_counts()


# In[11]:


#visualizing the counts for malign and benign diagnoses
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="dark")
ax = sns.countplot(x="diagnosis", data=cellData)


# In[12]:


#distribution of each variable
cellData.hist(bins=5, figsize=(15,15));


# In[13]:


#observing the relationships between variables
sns.pairplot(cellData.loc[:,'diagnosis':'concavity_mean'], hue="diagnosis");


# In[14]:


#correlation heatmap
plt.figure(figsize=(20,20))
sns.heatmap(cellData.corr(),cmap="inferno", annot = True);


# In[15]:


benign = cellData[cellData.diagnosis == 'B']
malignant = cellData[cellData.diagnosis == 'M']


# In[16]:


#kde plots for relationships between variables for benign diagnoses
g = sns.PairGrid(benign.loc[:,'radius_mean':'concavity_mean'])
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels=8, color = 'green', fill= True);


# In[17]:


#kde plots for relationships between variables for malign diagnoses
g = sns.PairGrid(malignant.loc[:,'radius_mean':'concavity_mean'])
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels=8, color= 'red', fill=True);


# In[18]:


#dropping the attribute with the result, as well as the attribute with negligible correlation
X = cellData.drop(['diagnosis','fractal_dimension_worst'],axis=1)
y = cellData['diagnosis']


# In[19]:


X


# In[20]:


#scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# In[21]:


X


# In[22]:


#splitting the data into test and train datasets
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.25, random_state=1)


# In[23]:


#K NEAREST NEIGHBOR
#
#
#
#

#finding the optimal number of neighbors, tuning the model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, plot_confusion_matrix, classification_report, roc_curve, roc_auc_score
knnplot = []
for n in range(2,10,1):
    knnClassifier = KNeighborsClassifier(n_neighbors=n, metric='minkowski')
    knnClassifier.fit(xtrain, ytrain)
    knnyPredict = knnClassifier.predict(xtest)
    knnplot.append(accuracy_score(ytest,knnyPredict))
plt.plot(list(range(2,10,1)), knnplot)
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[24]:


#training the model and finding the accuracy score
knnOptimal = KNeighborsClassifier(n_neighbors = 8)
knnOptimal.fit(xtrain,ytrain)
knnyOptimalPred = knnOptimal.predict(xtest)
accuracy_score(ytest,knnyOptimalPred )


# In[25]:


#confusion matrix for k-nearest neighbor
plot_confusion_matrix(knnOptimal, xtest, ytest, display_labels= ['Benign', 'Malignant'], cmap = "Reds", normalize= None)
plt.title('Confusion Matrix for K-Nearest Neighbor')
plt.show()


# In[26]:


print(classification_report(ytest, knnyOptimalPred))


# In[27]:


#creating the ROC curve for k-nearest neighbor
ytestMOD = [1 if each == "M" else 0 for each in ytest]
ypredMOD = [1 if each == "M" else 0 for each in knnyOptimalPred]
fpr, tpr, _= roc_curve(ytestMOD, ypredMOD)
auc_score = roc_auc_score(ytestMOD, ypredMOD)
plt.plot(fpr, tpr, label="auc="+str(auc_score), color = 'red')
plt.box(True)
plt.title('ROC Curve for K-Nearest Neighbor')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()
roc_acu = []
roc_acu.append(auc_score)


# In[28]:


#creating precision-recall curve for k-nearest neighbor
from sklearn.metrics import auc, precision_recall_curve
precision, recall, _ = precision_recall_curve(ytestMOD, ypredMOD)
auc_score2 = auc(recall, precision)
plt.plot(recall, precision, label="auc="+str(auc_score2), color = 'red')
plt.title('Precision-Recall Curve for K-Nearest Neighbor')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()
pr_acu = []
pr_acu.append(auc_score2)


# In[29]:


### NAIVE BAYES
#
#
#
#

#training the model and finding the accuracy score

from sklearn.naive_bayes import GaussianNB
naiveBayes = GaussianNB()
naiveBayes.fit(xtrain,ytrain)
naiveBayesPred = naiveBayes.predict(xtest)
accuracy_score(ytest, naiveBayesPred)


# In[30]:


#confusion matrix for naive bayes

plot_confusion_matrix(naiveBayes, xtest, ytest, display_labels= ['Benign', 'Malignant'], cmap = "Blues", normalize= None)
plt.title('Confusion Matrix for Naive Bayes')
plt.show()


# In[31]:


print(classification_report(ytest, naiveBayesPred))


# In[32]:


#ROC curve for naive bayes
ytestMOD = [1 if each == "M" else 0 for each in ytest]
ypredNBMOD = [1 if each == "M" else 0 for each in naiveBayesPred]
fpr, tpr, _= roc_curve(ytestMOD, ypredNBMOD)
auc_score = roc_auc_score(ytestMOD, ypredNBMOD)
plt.plot(fpr, tpr, label="auc="+str(auc_score))
plt.box(True)
plt.title('ROC Curve for Naive Bayes')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()
roc_acu.append(auc_score)


# In[33]:


#precision-recall curve for naive bayes
from sklearn.metrics import auc
precision, recall, _ = precision_recall_curve(ytestMOD, ypredNBMOD)
auc_score2 = auc(recall, precision)
plt.plot(recall, precision, label="auc="+str(auc_score2))
plt.title('Precision-Recall Curve for Naive Bayes')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()
pr_acu.append(auc_score2)


# In[34]:


# RANDOM FOREST
#
#
#
#

#finding the optimal number of estimators, tuning the model
from sklearn.ensemble import RandomForestClassifier
RFPlot = []
for e in range(10,150):
    rfo = RandomForestClassifier(n_estimators = e, random_state=0)
    rfo.fit(xtrain, ytrain)
    rfoPred = rfo.predict(xtest)
    RFPlot.append(accuracy_score(ytest,rfoPred))
plt.plot(list(range(10,150)), RFPlot)
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.show()


# In[35]:


#training the model and finding the accuracy score

rfOptimal = RandomForestClassifier(n_estimators=65)
rfOptimal.fit(xtrain,ytrain)
rfOptimalPred = rfOptimal.predict(xtest)
accuracy_score(ytest, rfOptimalPred)


# In[36]:


#confusion matrix for random forest

plot_confusion_matrix(rfOptimal, xtest, ytest, display_labels= ['Benign', 'Malignant'], cmap = "Greens", normalize= None)
plt.title('Confusion Matrix for Random Forest')
plt.show()


# In[37]:


print(classification_report(ytest, rfOptimalPred))


# In[38]:


#ROC curve for random forest
ytestMOD = [1 if each == "M" else 0 for each in ytest]
ypredRFMOD = [1 if each == "M" else 0 for each in rfOptimalPred]
fpr, tpr, _= roc_curve(ytestMOD, ypredRFMOD)
auc_score = roc_auc_score(ytestMOD, ypredRFMOD)
plt.plot(fpr, tpr, label="auc="+str(auc_score), color = 'green')
plt.box(True)
plt.title('ROC Curve for Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()
roc_acu.append(auc_score)


# In[39]:


#precision-recall curve for random forest
from sklearn.metrics import auc
precision, recall, _ = precision_recall_curve(ytestMOD, ypredRFMOD)
auc_score3 = auc(recall, precision)
plt.plot(recall, precision, label="auc="+str(auc_score3), color = 'green')
plt.title('Precision-Recall Curve for Random Forest')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()
pr_acu.append(auc_score3)


# In[40]:


### SUPPORT VECTOR MACHINE
#
#
#

#training the model and finding the accuracy score for different kernel
#functions to find the best performing one for this dataset

from sklearn.svm import SVC
svmModel = SVC(kernel = "rbf").fit(xtrain, ytrain)
svmPred= svmModel.predict(xtest)
acc_rbf = accuracy_score(ytest, svmPred)
svm_comp = []
svm_comp.append(acc_rbf)
acc_rbf


# In[41]:


svmModel2 = SVC(kernel = "linear").fit(xtrain, ytrain)
svmPred2 = svmModel2.predict(xtest)
acc_rbf2 = accuracy_score(ytest, svmPred2)
svm_comp.append(acc_rbf2)
acc_rbf2


# In[42]:


svmModel3 = SVC(kernel = "poly").fit(xtrain, ytrain)
svmPred3 = svmModel3.predict(xtest)
acc_rbf3 = accuracy_score(ytest, svmPred3)
svm_comp.append(acc_rbf3)
acc_rbf3


# In[43]:


svmModel4 = SVC(kernel = "sigmoid").fit(xtrain, ytrain)
svmPred4 = svmModel4.predict(xtest)
acc_rbf4 = accuracy_score(ytest, svmPred4)
svm_comp.append(acc_rbf4)
acc_rbf4


# In[44]:


#plot showing the accuracy scores for each kernel function of SVM
svm_names = ['SVM with RBF Kernel', 'SVM with Linear Kernel', 'SVM with Poly Kernel', 'SVM with Sigmoid Kernel']
ax = sns.barplot(x= svm_names ,y = svm_comp, palette = 'BuPu')
plt.xlabel('SVM with Different Kernel Functions')
plt.ylabel('Accuracy')
plt.title("Accuracy vs SVM with Different Kernel Functions", fontsize = 20)
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate(f'{height:.2%}', (x + width/2, y + height*1.02), ha='center', fontsize = 'medium')
plt.show()


# In[45]:


#confusion matrix for SVM

plot_confusion_matrix(svmModel, xtest, ytest, display_labels= ['Benign', 'Malignant'], cmap = "Purples", normalize= None)
plt.title('Confusion Matrix for Support Vector Machine')
plt.show()


# In[46]:


print(classification_report(ytest, svmPred))


# In[47]:


#ROC curve for SVM
ytestMOD = [1 if each == "M" else 0 for each in ytest]
ypredSVMMOD = [1 if each == "M" else 0 for each in svmPred]
fpr, tpr, _= roc_curve(ytestMOD, ypredSVMMOD)
auc_score7 = roc_auc_score(ytestMOD, ypredSVMMOD)
plt.plot(fpr, tpr, label="auc="+str(auc_score7), color = 'purple')
plt.box(True)
plt.title('ROC Curve for Support Vector Machine')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()
roc_acu.append(auc_score7)


# In[48]:


#precision-recall curve for SVM
from sklearn.metrics import auc
precision, recall, _ = precision_recall_curve(ytestMOD, ypredSVMMOD)
auc_score4 = auc(recall, precision)
plt.plot(recall, precision, label="auc="+str(auc_score4), color = 'purple')
plt.title('Precision-Recall Curve for Support Vector Machine')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()
pr_acu.append(auc_score4)


# In[49]:


#comparing accuracy scores of the models
models = ['K-Nearest Neighbor','Naive Bayes','Random Forest','Support Vector Machine']
accuracies = [0.958041958041958, 0.9370629370629371, 0.951048951048951, 0.965034965034965]
plt.rcParams['figure.figsize']=13,8 
ax = sns.barplot(x=models, y = accuracies, palette = 'Set1')
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate(f'{height:.2%}', (x + width/2, y + height*1.02), ha='center', fontsize = 'medium')
plt.xlabel("Classification Models", fontsize = 15 )
plt.ylabel("Accuracy", fontsize = 15)
plt.title("Accuracy vs Classification Models", fontsize = 20)
plt.show()


# In[50]:


#comparing AUC scores from the ROC curves of all models
models = ['K-Nearest Neighbor','Naive Bayes','Random Forest','Support Vector Machine']
plt.rcParams['figure.figsize']=13,8 
ax = sns.barplot(x=models, y = roc_acu, palette = 'Set1')
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate(f'{height:.2%}', (x + width/2, y + height*1.02), ha='center', fontsize = 'medium')
plt.xlabel("Classification Models", fontsize = 15 )
plt.ylabel("AUC Score from ROC Curve", fontsize = 15)
plt.title("AUC Score from ROC Curve vs Classification Models", fontsize = 20)
plt.show()


# In[51]:


#comparing AUC scores from the Precision-Recall curves of all models

models = ['K-Nearest Neighbor','Naive Bayes','Random Forest','Support Vector Machine']
plt.rcParams['figure.figsize']=13,8 
ax = sns.barplot(x=models, y = pr_acu, palette = 'Set1')
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate(f'{height:.2%}', (x + width/2, y + height*1.02), ha='center', fontsize = 'medium')
plt.xlabel("Classification Models", fontsize = 15 )
plt.ylabel("AUC Score from Precision-Recall Curve", fontsize = 15)
plt.title("AUC Score from Precision-Recall Curve vs Classification Models", fontsize = 20)
plt.show()


# In[ ]:





# In[ ]:




