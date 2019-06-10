#! /home/zhuo/anaconda3/envs/CECL/bin/python
# -*- coding: utf-8 -*-

from clean_data import *
from set_label import set_label
import pandas as pd
import numpy as np

# This is the main function for the cecl project

###################### Section 1 Data Preparation #########################
def data_preparation(filename):
    # read data from dataset
    all_loans = read_data(filename)

    # sort the dataset by time
    all_loans = re_organize(all_loans)

    # create test data set
    train_set, test_set = create_test_set(all_loans, 0.8)
    
    train_copy = train_set.copy()
    test_copy = test_set.copy()
    
    # delete unuseful columns
    train_copy = drop_columns(train_copy)
    test_copy = drop_columns(test_copy)
    
    # Create label and input sets
    loans = train_copy.drop("status", axis = 1)
    loans_labels = train_copy["status"].copy()
    
    # Select numeric columns 
    loans_num = loans.drop(["flag_fthb","ppmt_pnlty","st","delq_sts"], axis = 1)
    
    # Select non-numeric columns to do onehot processing
    loans_cat_1hot = loans[["flag_fthb","ppmt_pnlty","st"]]
    
##################### Section 2 Feature Engineering #########################
    
def feature_proccessing():
    
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    # Create a transformation pipeline
    # Use "mean" method to fill in the missing data 
    # Use standardsclar to scale the numeric data
    num_pipline = Pipeline([
        ('imputer', SimpleImputer(strategy = 'mean')),
        ('std_scaler', StandardScaler()),
    ])

    num_att = list(loans_num)
    cat_1hot_att = list(loans_cat_1hot)

    # Bind two pipelines 
    full_pipline = ColumnTransformer([
        ("num", num_pipline, num_att),
        ("cat_1hot", OneHotEncoder(), cat_1hot_att)
    ])

    loans_prepared = full_pipline.fit_transform(loans)

##################### Section 3 Train the Model ############################
def train_model():
    from sklearn.neural_network import MLPClassifier
    
    # Choose MLP neuro network
    clf = MLPClassifier()

    clf.fit(loans_prepared, loans_labels)
    
##################### Section 4 Validation #################################
def validation():
    
    from sklearn.model_selection import cross_val_predict

    labels_train_pred = cross_val_predict(clf, loans_prepared, loans_labels, cv = 3)
    
    from sklearn.metrics import precision_score, recall_score
    
    # Precision Score and Recall Score for the data set
    print("##############################################################################")
    print("########################## This is the training set ##########################")
    print("Precision Score:", precision_score(loans_labels, labels_train_pred))
    print("Recall Score:", recall_score(loans_labels,labels_train_pred))
    print("##############################################################################")
    
    test_prepared = full_pipline.fit_transform(test_loans)

    test_pred = cross_val_predict(clf, test_prepared, test_labels, cv =3)
    print("##############################################################################")
    print("########################## This is the test set ##############################")
    print("Precision Score:", precision_score(test_labels, test_pred))
    print("Recall Score:", recall_score(test_labels, test_pred))
    print("Test Score:", clf.score(test_prepared,test_labels))
    print("##############################################################################")
    
if __name__ == "__main__":
    filename = "LoansSample01.csv"
    data_preparation(filename)
    feature_proccessing()
    train_model()
    validation()
    
