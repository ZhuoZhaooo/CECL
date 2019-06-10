#! /home/zhuo/anaconda3/envs/CECL/bin/python
# -*- coding: utf-8 -*-

import pandas as pd

# define a function to set labels for the dataset 
# if the delq_sts exceeds the assgined number 
# then the "status" will be True(1)
def set_label(filename, delq):
    
    filepath = "~/cecl_project/data/original_data/" + filename
    
    # read the dataset 
    df=pd.read_csv(filepath ,index_col=0)
    # create a subset whose "delq_sts" contains only numbers 
    subset_numbers = df[~(df['delq_sts'].str.contains('R', na=False))]
    # This is the subset without default
    subset_nodefault = subset_numbers[(subset_numbers['delq_sts'].astype(int) >= 0)
                                      &(subset_numbers['delq_sts'].astype(int) <= delq)]
    # This is the subset with default
    subset_default = subset_numbers[(subset_numbers['delq_sts'].astype(int) >= delq) 
                                    &(subset_numbers['delq_sts'].astype(int) <= 56)]

    # Thi is the subset with R
    subset_R = df[(df['delq_sts'].str.contains('R', na=False))]
    
    # Create a list that contains the names of loans default
    
    default_list = subset_default['id_loan'].tolist()
    
    default_list = pd.unique(default_list)

    # Get loans default and Add 1 to status
    default_loans = df[df['id_loan'].isin(default_list)]
    
    default_loans['status'] = 1
    
    # Get loans not default and Add 0 to status

    nodefault_loans = df[~(df['id_loan'].isin(default_list))]
                           
    nodefault_loans['status'] = 0
    
    # Merge the non-default loans and default loans
    all_loans = pd.concat([nodefault_loans, default_loans])
    
    # Save the data
    savepath = "~/cecl_project/data/cleaned_data/"
    
    all_loans.to_csv(path_or_buf = savepath + "cleaned" + filename)
                           
    return None