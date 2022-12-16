import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


# setting random seed to 7
np.random.seed(7)



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
    df = bin_features(df)
    train, validate, test = split_german_credit(df)
    
    
    return train, validate, test
    

#######       Acquire German Credit Data       #######

def acquire_german_credit():

    df = pd.read_csv('data/german_credit_data.csv', index_col = 0)
    df.columns = df.columns.str.lower()
    df['risk'] = df.risk.map({'good':1, 'bad':0})

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


    #df['checking account'] = df['checking account'].replace({np.NAN:'unknown'})
    #df['saving accounts'] = df['saving accounts'].replace({np.NAN:'unknown'})
    df['checking account'].fillna('unknown', inplace=True)
    df['saving accounts'].fillna('unknown', inplace=True)

    for col in df.columns:
        if (df[col].dtype == object):
            df[col] = df[col].astype('category')
    
    

    df['sex'] = df.sex.map({'male':1, 'female':0})
    df['job'] = df.job.map({0:'unskilled_nonresident', 1:'unskilled', 2: 'skilled', 3: 'high_skill'})
    df['job'] = df['job'].astype('category')

    df['checking account'] = df['checking account'].cat.set_categories(['unknown','little', 'moderate', 'rich'], ordered = True)
    df['saving accounts'] = df['saving accounts'].cat.set_categories(
        ['unknown', 'little', 'moderate', 'rich', 'quite rich'], ordered = True)
    df['job'] = df['job'].cat.set_categories(['unskilled_nonresident', 'unskilled', 'skilled', 'high_skill'], ordered = True)

    df['sex'] = df['sex'].astype(int)
    df['risk'] = df['risk'].astype(int)
    return df


#######       bin features in for model      #######

def bin_features(df): 

    # the first number is the minimum value of the first group, the last number is the maximum value of the last group
    # the total number of groups is the length of the list - 1
    
    #creating bins for age
    age_groups = (18,26,33,45,76)
    age_categories = ['early_life','early_established','established','older']
    df['age_groups'] = pd.cut(df['age'],age_groups, labels=age_categories)
    del df['age']

    #creating bins for loan duration
    duration_groups = (0, 11, 12, 35, 36, 100)
    duration_names = ('short_term_loan', 'one_year_loan','medium_term_loan', 'three_year_loan', 'long_term_loan')
    df['loan_duration_groups'] = pd.cut(df['duration'], duration_groups, labels = duration_names)
    del df['duration']

    return df


##############################################################
  ############       Secondary Function       ##############
######  Splits Model into prepared X and Y dataframes  ######
##############################################################

def german_credit_x_y(target):
    '''
    This function depends on the split function, being handed a dataframe
    and the target variable we are seeking to understand through prediction
    '''

    # calls split function to produce required variables
    train, validate, test = get_german_credit()

    #for i in [train, validate, test]:
    #    i = get_dummies(i)
    train = get_dummies(train)
    validate = get_dummies(validate)
    test = get_dummies(test)

    x_train = train.drop(columns=[target])
    y_train = train[target]
    
    x_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    x_test = test.drop(columns=[target])
    y_test = test[target]
    
    return x_train, y_train, x_validate, y_validate, x_test, y_test 


#######       get_dummies      #######

def get_dummies(df):
    
    #dummy_list = ['job', 'housing', 'saving accounts', 'checking account', 'purpose', 'age_groups']
    dummy_list = ['housing', 'saving accounts', 'checking account', 'purpose', 'age_groups', 'loan_duration_groups']
    del df['job']


    for items in dummy_list:
        df = pd.get_dummies(data=df, columns=[items])

    return df 