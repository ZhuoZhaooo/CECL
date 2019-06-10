#! /home/zhuo/anaconda3/envs/CECL/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np 

# read loans from dataset
def read_data(filename):
    
    filepath = "~/cecl_project/data/cleaned_data/" + "cleaned" + filename
    
    all_loans = pd.read_csv(filepath)
    
    return all_loans

# create a test set 
def create_test_set(all_loans, percentage):
    
    n_rows = all_loans.shape[0]
    
    s_num = int(n_rows * percentage)
    
    train_set = all_loans[:s_num]
    test_set = all_loans[s_num:]
    
    return train_set, test_set

# delete unuseful columns
def drop_columns(data):
    drop_lists = ["preharp_id_loan","id_loan",
                  "prod_type","repch_flag",
                  "cd_zero_bal","dt_zero_bal","actual_loss",
                  "dt_matr"]
    
    data_cleaned = data.drop(drop_lists, axis = 1)
    data_cleaned = data_cleaned.dropna()
    
    return data_cleaned 

# reorganize the dataset by time
def re_organize(df):
    df.set_index("svcg_cycle", inplace = True)
    df.sort_index(ascending="True", inplace = True)
    
    return df

