import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split






##############################################################
  ##############       Primary Function       ##############
#######  Creates, cleans dataframe, returns dataframe  #######
##############################################################
'''
acquires the data from local german credit csv
cleans and prepares the data for the exploration
returns dataframe split between train, test, and 
validate groupings
'''

def get_german_credit():
    df = acquire_german_credit()
    df = category_adjustments(df)
    train, validate, test = split_german_credit(df)
    
    return train, validate, test


#######       Acquire German Credit Data       #######
def acquire_german_credit():

    df = pd.read_csv('data/german_credit_data.csv', index_col = 0)

    
    return df


#######       Splits dataframe for scientific process       #######
def split_german_credit(df):
    #splits dataframe into two groups, group 1 and 2, 2 is the test group and will not be explored
    train_validate, test = train_test_split(df, test_size = 0.2)
    #splits group 1 further into two groups, train and validate
    train, validate = train_test_split(train_validate, test_size=0.3)
    
    return train, validate, test


#######       Engineering DataFrame       #######
def category_adjustments(df):

    df.columns = df.columns.str.lower()
    df['checking account'] = df['checking account'].replace({np.NAN:'unknown'})
    df['saving accounts'] = df['saving accounts'].replace({np.NAN:'unknown'})
    
    for col in df.columns:
        if (df[col].dtype == object):
            df[col] = df[col].astype('category')
    
    df['checking account'] = df['checking account'].cat.set_categories(['none','little', 'moderate', 'rich'], ordered = True)
    df['saving accounts'] = df['saving accounts'].cat.set_categories(
        ['none', 'little', 'moderate', 'rich', 'quite rich'], ordered = True)
    
    df['risk'] = df.risk.map({'good':1, 'bad':0})
    return df


