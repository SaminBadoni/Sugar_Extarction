#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 22:20:03 2021

@author: samin
"""
import os
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
# OS independent 
sep = os.sep
cwd = os.getcwd()

def create_model():
    """
    return SVR model definition
    """
    regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    return regr

def load_data(filepath):
    """
    read csv file and return features and target values
    """
    
    data = pd.read_csv(filepath)
    orginal_efficiency = np.array(data['Efficiency'])
    orginal_efficiency  =np.reshape(orginal_efficiency,(orginal_efficiency.shape[0],1))
    
    # drop is drop based on analysis - check EDA file
    column_name_to_drop = ['Unnamed: 0','Id', 'Size(L)','OG','Color','Efficiency','SugarScale','BrewMethod','Style']
    
    data = data.drop(column_name_to_drop, axis = 1)
    return data , orginal_efficiency
    
if __name__ =="__main__":
    
    
    parser = argparse.ArgumentParser(description='Efficiency checker')
    parser.add_argument('-model_path', type=str,default="model", help='save/load model dirname')
    parser.add_argument('-check', type=str,default="train", help='train/test')
    parser.add_argument('-data_path', type=str, help='input csv file path')
    parser.add_argument('-output_file', default="result.csv", help='input csv file path')

    args = parser.parse_args()
    
    print(args.model_path)
    
    if args.check == "train":
        data,target = load_data(args.data_path)
        train_dataset, test_dataset, y_train, y_test = train_test_split( data, target , test_size=0.33, random_state=42,shuffle=True)

        model = create_model()
        model.fit(train_dataset,y_train)
        print("training score :",model.score(train_dataset,y_train))
        print("val score :",model.score(test_dataset,y_test))
        
        if not os.path.exists(cwd+sep+args.model_path):
            os.mkdir(cwd+sep+args.model_path)
        joblib.dump(model, cwd+sep+args.model_path+sep+"svr_model.sav")
        
    else:
        ##TEST DATA - can not handle null/nan value due to domain constrain 
        data,target = load_data(args.data_path)
        
        model = joblib.load(cwd+sep+args.model_path+sep+"svr_model.sav")
        predicted_value = model.predict(data)
        
        rmse = np.sqrt(MSE(target, predicted_value))
        print("RMS-test : % f" %(rmse/100)) # divide by 100 as pipeline will undo the normalization in prediction
        
        output = pd.DataFrame(zip(target,predicted_value) , columns=['Original_efficiency','predicted_efficiency'])
        output.to_csv(cwd+sep+args.output_file+".csv",index=False)
        