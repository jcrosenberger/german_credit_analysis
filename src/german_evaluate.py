# whole standard Data Science library
import pandas as pd
import numpy as np

#vizualization libraries
import seaborn as sns
import matplotlib.pyplot as plt

# selected functions from Data Science libraries
from scipy.stats import spearmanr, pearsonr, f_oneway

# libraries for convenience
pd.options.display.float_format = '{:,.3f}'.format




##############################################
######      Continuous Test DataFrame      ########
##############################################
def pearson_test_df(df, target_var, test_var_list):
    '''default test for continuous to continuous correlation tests. 
    Handles linear relationships well'''
    
    pearson_df = pd.DataFrame(
        {'Potential_Feature':[],
         'Coefficient' :[],
         'P-Value' : [],
         'Significance' : [],
         'Keep' : [],})

    for item in test_var_list:
        r, p_value = pearsonr(df[target_var], df[item])
        if 1 - p_value >= 0.95:
            keeper = 'Yes'
        else:
            keeper = 'No'
        
        pearson_df = pearson_df.append(
        {'Potential_Feature': item,
         'Coefficient' : r,
         'P-Value' : p_value,
         'Significance' : 1-p_value,
         'Keep' : keeper},
        ignore_index = True)
        
    return pearson_df


##############################################
######      Categorical Test DataFrame      ########
##############################################
def ftest_df(df):
    ftest_df = pd.DataFrame(
        {'Potential_Feature':[],
         'F-stat':[],
         'Significance':[],
         'Keep':[]})
    
    f, p = ftest_housing(df)
    if 1 - p >= 0.95:
        keeper = 'Yes'
    else:
        keeper = 'No'

    ftest_df = ftest_df.append(
        {'Potential_Feature':'housing',
         'F-stat':f,
         'Significance':1-p,
         'Keep':keeper},
        ignore_index = True)


    f, p = ftest_checking(df)
    if 1 - p >= 0.95:
        keeper = 'Yes'
    else:
        keeper = 'No'

    ftest_df = ftest_df.append(
        {'Potential_Feature':'checking',
         'F-stat':f,
         'Significance':1-p,
         'Keep':keeper},
        ignore_index = True)

    
    f, p = ftest_savings(df)
    if 1 - p >= 0.95:
        keeper = 'Yes'
    else:
        keeper = 'No'

    ftest_df = ftest_df.append(
        {'Potential_Feature':'savings',
         'F-stat': f,
         'Significance':1-p,
         'Keep':keeper},
        ignore_index = True)
        
    return ftest_df

#######       Categorical Test Functions       #######
def ftest_housing(df):
    #f, p = f_oneway(df[df[var] 
    
    f, p = f_oneway(df[df['housing'] == 'own'].risk, 
                   df[df['housing'] == 'rent'].risk, 
                   df[df['housing'] == 'free'].risk)
                    
    #print(f'F-statistics: {round(f, 3)}, p: {round(1-p, 3)}')
    return f, p

def ftest_savings(df):
    #f, p = f_oneway(df[df[var] 
    
    f, p = f_oneway(df[df['saving accounts'] == 'unknown'].risk, 
                   df[df['saving accounts'] == 'little'].risk, 
                   df[df['saving accounts'] == 'moderate'].risk, 
                   df[df['saving accounts'] == 'rich'].risk, 
                   df[df['saving accounts'] == 'quite rich'].risk)
                    
    #print(f'F-statistics: {round(f, 3)}, p: {round(1-p, 3)}')
    return f, p

def ftest_checking(df):
    #f, p = f_oneway(df[df[var] 
    
    f, p = f_oneway(df[df['checking account'] == 'unknown'].risk, 
                   df[df['checking account'] == 'little'].risk, 
                   df[df['checking account'] == 'moderate'].risk, 
                   df[df['checking account'] == 'rich'].risk) 
                                       
    #print(f'F-statistics: {round(f, 3)}, p: {round(1-p, 3)}')
    return f, p


###########################################################################
##########       Vizualizing Correlated Variables with Risk      ##########
###########################################################################
def correlate_viz(df, target):

    # sets size of the vizualization product
    plt.figure(figsize=(10,10))

    # DataFrame 1 - both features
    # creates a vertical heat map, correlating values in dataframe with a feature in the dataframe 
    # (the target value to be predicted)
    plt.subplot(1,2,1)
    heatmap = sns.heatmap(df.corr()[[target]].sort_values(by=target, ascending = False), vmin=-1, vmax=1, annot=True,cmap='BrBG')

    # title information
    heatmap.set_title('Features Correlating with \nRisk', fontdict={'fontsize':18}, pad=16);
