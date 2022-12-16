# Data Science Libraries
import pandas as pd
import numpy as np

# Sklearn Libraries for models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# setting random seed to 7
np.random.seed(7)


######################################################################
##########       Determine Baseline for model to beat      ##########
######################################################################

def baseline(df,var):
    ''' Method calculates the minimum odds that a model will need to beat,
    based on the target variable and the dataframe passed in
    '''

    if (df[var].value_counts(normalize=True))[0] > (df[var].value_counts(normalize=True))[1]:
        baseline = (df[var].value_counts(normalize=True))[0]
    else: 
        baseline = (df[var].value_counts(normalize=True))[1]
    return baseline



###############################################################
     ############       KNN Modelling       ##############     
  ######  Nearest Neighbor within K neighbor distance  ######
###############################################################


#######      Determine which KNN model is best to use       ########

def find_best_knn(x_train, x_validate, y_train, y_validate, df=False):
    '''
    Feed in split dataframes.
    - If df is false, then it will present a graph showing the changes between
    the nearest neighbor models as K changes
    - If df is true, then function will return a dataframe which gives the 
    resultant 
    '''
    
    
    
    k_range = range(1, 25)
    train_scores = []
    validate_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(x_train, y_train)
        train_scores.append(knn.score(x_train, y_train))
        validate_scores.append(knn.score(x_validate, y_validate))

    if df == False:
        plt.figure()
        plt.xlabel('k')
        plt.ylabel('accuracy')
        plt.plot(k_range, train_scores, label='Train')
        plt.plot(k_range, validate_scores, label='Validate')
        plt.legend()
        plt.xticks([0,5,10,15,20])
        plt.show()

    else:
        scores_dict = {'train': train_scores, 'validate': validate_scores}
        scores_df = pd.DataFrame(scores_dict)
        return scores_df


def knn_model(x, y, k=12):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x, y)
    
    y_preds = knn.predict(x)
    
    return y_preds


#########################################################################
           ############       Random Forest       ##############     
  ######  Creates N number of trees using random starting values  ######
########################################################################

def random_forest_model(x, y):
    
    rf_classifier = RandomForestClassifier(
        min_samples_leaf=10,
        n_estimators=200,
        max_depth=5, 
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        max_features='auto'
    )

    rf_classifier.fit(x, y)

    y_preds = rf_classifier.predict(x)
    
    return y_preds


#############################################################################
    ############       Gradient Boosting Classifier       ##############     
######  Creates a random forest where each tree learns from the last  ######
############################################################################

def gradient_booster_model(x, y):
    
    gradient_booster = GradientBoostingClassifier(
        learning_rate=0.1,
        max_depth = 5,
        n_estimators=200)

    gradient_booster.fit(x, y)
    
    y_preds = gradient_booster.predict(x)
    
    return y_preds


#############################################################################
    ############       Model Evaluation       ##############     
   ######  Easily evaluate models for accuracy or any other metric  ######
############################################################################

def evaluate_classification_model(model, y_train, y_preds, df=False, full= False):
    TN, FP, FN, TP = confusion_matrix(y_train,y_preds).ravel()
    ALL = TP + TN + FP + FN

    accuracy = (TP + TN)/ALL
    true_positive_rate = TP/(TP+FN)
    false_positive_rate = FP/(FP+TN)
    true_negative_rate = TN/(TN+FP)
    false_negative_rate = FN/(FN+TP)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    
    if df == False:
        return accuracy

    f1_score = 2*(precision*recall)/(precision+recall)
    
    if full == True:
        performance_df = pd.DataFrame(
                                         {'model' : [model],
                                          'accuracy' : [accuracy],
                                          'f1_score' : [f1_score],
                                          'precision' : [precision], 
                                          'recall' : [recall],
                                          'true_positive_rate' : [true_positive_rate],
                                          'false_positive_rate': [false_positive_rate], 
                                          'true_negative_rate' : [true_negative_rate], 
                                          'false_negative_rate': [false_negative_rate]
                                          })
        return performance_df

    
    
    if full == False:
        performance_df = pd.DataFrame(
                                         {'model' : [model],
                                          'accuracy' : [accuracy],
                                          'f1_score' : [f1_score],
                                          'precision' : [precision], 
                                          'recall' : [recall]
                                         })

    if df == True:
        return performance_df


def get_models(x_train, y_train, x_validate, y_validate):


    rf_y_preds_train = random_forest_model(x_train, y_train)
    rf_y_preds_val = random_forest_model(x_validate, y_validate)

    gb_y_preds_train= gradient_booster_model(x_train, y_train)
    gb_y_preds_val = gradient_booster_model(x_validate, y_validate)

    knn_y_preds_train= knn_model(x_train, y_train, k=11)
    knn_y_preds_val = knn_model(x_validate, y_validate)



    performance_df = evaluate_classification_model('random_forest', y_train, rf_y_preds_train, df=True)
    performance_df = performance_df.append(evaluate_classification_model('rf_validate', y_validate, rf_y_preds_val, df=True))
    performance_df = performance_df.append(evaluate_classification_model('gradient_booster', y_train, gb_y_preds_train, df=True))
    performance_df = performance_df.append(evaluate_classification_model('gb_validate', y_validate, gb_y_preds_val, df=True))
    performance_df = performance_df.append(evaluate_classification_model('knn', y_train, knn_y_preds_train, df=True))
    performance_df = performance_df.append(evaluate_classification_model('knn_validate', y_validate, knn_y_preds_val, df=True))
    
    return performance_df