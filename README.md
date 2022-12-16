<h1 style="color:#900C3F;font:luminary;text-align:center;"><i>German Credit Analysis </i></h1>

<h2 style="color:#900C3F;text-align:center;"> Predicting which applicants at german bank(s?) were considered to be high or low risk in a historical dataset - December 16, 2022 </h2>

<h2 style="color:#900C3F;text-align:center;"> Abstract </h2>
Examining Professor Dr. Hans Hofmann's german credit analysis dataset, I was able to build a gradient boosting ensemble model which perfectly determined which credit applicants were considered to be 'high' or 'low' risk by a german bank. This model is computationally intensive, but for a small sample of 1,000 observations, it is simple, and easy and provides excellent results in less than a second on an older MacBook Pro. 

This kind of model can be used to determine which applicant features are more and less important related to the risk result. That means this model can be used to examine the biases which banks or other creditors may have consciously, unconsciously, or within their models from the past. The more applicants we are examining, the more computational power and time needed for the results, but we can be sure of the accuracy of the model given enough time for the dataset to be sifted through.




<h1 style="color:#900C3F;">Task at hand</h1> 

### Using several data science methods detailed below, we will see if we can accurately predict german bank risk analysis for previous applicants
#### Data Science Methods include:
- splitting model into appropriate groups when recognizing their existence
- classification modeling
- feature engineering when patterns can be determined 

<h1 style="color:#900C3F;">Project Goal</h1> 

 Professor Dr. Hans Hofmann released a dataset with 1000 historical applicants for loans from german banks or perhaps from one german bank. His write up did not disambiguate this. His intention was for this dataset to be used to use 'algorithms' to predict whether those previous applicants would be given a high or low risk classification. As a low risk loan applicant, a bank will offer lower rates on loans than they would to a high risk applicant. 

 <h1 style="color:#900C3F;">Plan of action</h1>

### Acquire and prepare
1. Acquire German Credit data from a public kaggle account. Transform the data to a Pandas dataframe for manipulating using Jupyter Notebook.
2. Prepare the data for exploration and analysis. Find out if there are some values missing and find a way to handle those missing values.
3. Change the data types if needed
4. Find if there are features which can be created to simplify the exploration process.
5. Determine which observations are outliers and handle them.
6. Create a data dictionary.
7. Split the data into 3 data sets: train, validate and test data (56%, 24%, and 20% respectively)

### Explore and pre-process
1. Explore the train data set through visualizations and statistical tests. 
2. Determine which variables have statistically significant relationships with zestimates errors. 
2. Make the exploration summary and document the main takeaways.
3. Impute the missing values if needed.
4. Pick the features which can help to build a good prediction model.
5. Identify if new features must be created to improve the model's accuracy.
6. Encode the categorical variables.
7. Split dataframe into segments if those segments represent different populations.
8. Split the target variable from the data sets.


### Explore modify or create add features for model
1. Create categories from integer variables which may have too many unique numbers for classification model.
2. Determine if new features are statistically significant

### Build a classification model
1. Pick from several classification algorithms to create prediction models.
2. Create the models and evaluate regressors using Accuracy and F-1 scores on the train data set.
3. Evaluate algorithms using train and validate sets, compared to one another.
3. Determine which algorithms produced best results
5. Make predictions for the test data set.
6. Evaluate the results.

### Report results from models built

### Draw conclusions


<h1 style="color:#900C3F;">Data Dictionary</h1>


| Feature | Definition | Manipulations applied|Data Type|
|--------|-----------|-----------|-----------|
||
|||**Categorical Data**
||
|*Sex*| Identifies whether the applicant is male or female  | Male or Female | category (boolean)
|*Job*| Identifies the work stability, status of the applicant  | 4 categories: unemployed/unskilled nonresident, unskilled resident, skilled employee, management/highly qualified employee | category
|*Housing*| Identifies the applicant's living arrangements  | 3 categories: rent, own, 'for free' | category
|*Saving Accounts*| How much has the applicant saved in M2 or higher accounts | 4 categories: Little,  Moderate, Rich, Quite Rich,| category
|*Checking Account*| How much M1 money does the applicant have  | 3 categories: Little, Moderate, Rich | category
|*Purpose*| Identifies the purpose of the loan  | 11 categories: car(new), car(used), furniture, tv, home appliance, repairs, education, retraining, business, other, vacation | category
||
|||**Numerical Data**
||
|*Age*|  Age of the Applicants | Applicants range from 19 to 75 years old | integer
|*Credit Amount*|  How much credit has been extended to applicant | ranges from 250 to 18424| integer
|*Duration*|  The duration of loan sought | in months ranging from 4 to 72 | integer
||
|||**Target Data**
||
|**Risk** | **Did a German bank consider the loan to be high or low risk** | **High or Low** | **Category**


<h1 style="color:#900C3F;">Reproducability</h1>


- I wrote three custom modules to handle operations. They are contained in the src folder.
- I previously downloaded dataset from kaggle and added it to the data folder.
- Functions written in modules should work when called in presentation notebook