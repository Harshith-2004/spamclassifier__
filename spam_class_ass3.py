import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn import svm
# from sklearn.feature_extraction.text
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
import chardet
import re
import os 
with open('emails.csv', 'rb') as f:
    result = chardet.detect(f.read())
har = pd.read_csv('emails.csv', encoding=result['encoding'])
har.columns=['text', 'spam']
har['text']=har['text'].apply(lambda text: re.sub(r'[^a-zA-Z\s]','', text))

vectoriser = CountVectorizer(stop_words='english')
vectorised_data = vectoriser.fit_transform(har['text'])
vectorised_data = vectorised_data.toarray()
feat=vectoriser.get_feature_names_out()
print(vectorised_data.shape)
with open('featurename.txt','w') as file:
    for f in feat:
        file.write(f + '\n')
Y= np.where(har['spam']==1, 1, 0)
X_train= vectorised_data
Y_train= Y

#laplace smoothening
a_ones = np.ones_like(X_train[0])
X_train= list(X_train)
X_train.append(a_ones)
X_train.append(a_ones)
Y_train= list(Y_train)
Y_train.append(0)
Y_train.append(1)
a_ones = np.zeros_like(X_train[0])
X_train= list(X_train)
X_train.append(a_ones)
X_train.append(a_ones)
Y_train= list(Y_train)
Y_train.append(0)
Y_train.append(1)
X_train= np.array(X_train)
Y_train=np.array(Y_train)


#Gaussian Naive Bayes algorithm 
bin_mat = np.where(X_train>0,1, 0 )
phi_nb = np.sum(Y_train)/Y_train.size
print(phi_nb)
bin_mat0 = bin_mat[Y_train==0]
bin_mat1 = bin_mat[Y_train==1]
h, w = bin_mat0.shape
p0_nb= np.sum(bin_mat0, axis=0)/h
h, w = bin_mat1.shape
p1_nb= np.sum(bin_mat1, axis=0)/h
print(p1_nb)
print(p0_nb)
# X_test= np.array(X_test)
# Y_test=np.array(Y_test)
# pre = np.zeros_like(Y_test)
# bin_mat = np.where(X_test>0,1, 0 )
# h, w = bin_mat.shape
t2_nb= math.log(phi_nb/(1-phi_nb))
t2_nb+=np.sum(np.log((1-p1_nb)/(1-p0_nb)))
t1_nb=np.log((p1_nb*(1-p0_nb))/((p0_nb*(1-p1_nb))), dtype='float64')

#testing of Bernoulli's Naive Bayes algorithm!!
test_folder= 'test'
test_emails = []
num_emails = len(os.listdir(test_folder))
for i in range(1, num_emails+1):
    filename = f'email{i}.txt'
    with open(os.path.join(test_folder, filename), "r") as file:
        test_emails.append(file.read())
vectoriser1= vectoriser.transform(test_emails)
vectoriser1 = vectoriser1.toarray()
pre = np.zeros(vectoriser1.shape[0], dtype='int')
bin_mat= np.where(vectoriser1>0,1, 0 )
for i in range(vectoriser1.shape[0]):
    if np.dot(t1_nb, bin_mat[i])+t2_nb>=0:
        pre[i]=1
print("Prediction from Bernoulli's Naive-Bayes algorithm", pre)

#logestic regression:
def sigmoid(z):
    return 1/(1+np.exp(-z))
def logistic_regression(X, y, stepsize,iterations):
    n, d = X.shape
    w= np.zeros(d)
    for i in range(iterations):
        axisx.append(i+1)
        predictions = sigmoid(np.dot(X, w))  #X has the shape nxd and y as nx1
        gradient = np.dot(X.T,(y-predictions))
        w+= stepsize*gradient    
    return w

stepsize= 0.01
iterations =100
print (Y_train.shape)
axisx=[]
axisy_train=[]
weights_logreg = logistic_regression(X_train, Y_train, stepsize, iterations)

#testing for the logestic regression:
#we know at the testing of Bernoulli's Naive Bayes Algorithm we have taken files our test is stored in vectoriser1
expec= np.matmul(weights_logreg.T, vectoriser1.T)
expec= np.where(expec>0.5, 1, 0)
print("Prediction from logistic regression algorithm",expec) 

#svc classifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# Assuming you have already split your data into training and testing sets
# X_train and y_train are your training data and labels
# X_test and y_test are your testing data and labels

# Create an SVM classifier
svm_classifier = SVC(kernel='linear')  # You can specify different kernels like 'linear', 'poly', 'rbf', etc.

# Train the SVM classifier
svm_classifier.fit(X_train, Y_train)

# Make predictions on the testing data
y_pred = svm_classifier.predict(vectoriser1)
print(y_pred)


# Mixing models for all the classifiers:
# Taking the 3 models into account and analysing the 3 and returning spam if we have more than 3 models saying them as spam
# expec is from logarithmic regression, pre is from Bernoulli's Naive Bayes and y_pred is from SVC
final_expec= expec+y_pred+pre
final_expec= np.where(final_expec>=2, 1, 0)
print(final_expec)


with open("prediction.txt", "w") as f:
    # Write the first line
    f.write("spam predicted by bernoulli's naive Bayes:")
    f.write(", ".join(map(str, pre.flatten())))
    f.write("\n")
    f.write("spam predicted by Logestic Regression:")
    f.write(", ".join(map(str, expec.flatten())))
    f.write("\n")
    f.write("spam predicted by SVM:")
    f.write(", ".join(map(str, y_pred.flatten())))
    f.write("\n")
    f.write("Spam predicted by final_predicter is: ")
    f.write(", ".join(map(str, final_expec.flatten())))
    f.write("\n")


