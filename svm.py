#-------------------------------------------------------------------------
# AUTHOR: Henry Hu
# FILENAME: svm.py
# SPECIFICATION: Complete this program to simulate grid search for SVM. Test accuracy of hyperparametmers.
# FOR: CS 4210- Assignment #3
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd

#defining the hyperparameter values
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the training data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature training data and convert them to NumPy array
y_training = np.array(df.values)[:,-1] #getting the last field to create the class training data and convert them to NumPy array

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the training data by using Pandas library

X_test = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature testing data and convert them to NumPy array
y_test = np.array(df.values)[:,-1] #getting the last field to create the class testing data and convert them to NumPy array

cur_highest_accuracy = 0.0
TOTAL_TRIALS = len(y_training)

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
#--> add your Python code here

for cur_c in c: #iterates over c
    for cur_deg in degree: #iterates over degree
        for cur_kernel in kernel: #iterates kernel
           for cur_dfshape in decision_function_shape: #iterates over decision_function_shape

                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape.
                #For instance svm.SVC(c=1, degree=1, kernel="linear", decision_function_shape = "ovo")
                #--> add your Python code here
                clf = svm.SVC(C=cur_c, degree=cur_deg, kernel=cur_kernel, decision_function_shape = cur_dfshape)

                #Fit SVM to the training data
                clf.fit(X_training, y_training)

                #make the SVM prediction for each test sample and start computing its accuracy
                #hint: to iterate over two collections simultaneously, use zip()
                #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
                #to make a prediction do: clf.predict([x_testSample])
                #--> add your Python code here

                error_count = 0

                for (x_testSample, y_testSample) in zip(X_test, y_test):
                    prediction = clf.predict([x_testSample])[0]
                    if y_testSample != prediction:
                        error_count += 1

                accuracy = 1 - error_count / float(TOTAL_TRIALS)

                #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
                #with the SVM hyperparameters. Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                #--> add your Python code here

                if cur_highest_accuracy < accuracy:
                    cur_highest_accuracy = accuracy
                    print(f'Highest SVM accuracy so far: {accuracy}, Parameters: c={cur_c}, degree={cur_deg}, kernel={cur_kernel}, decision_function_shape={cur_dfshape}')
