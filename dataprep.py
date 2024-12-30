import importlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn import set_config
from sklearn.metrics import *
from sklearn.compose import ColumnTransformer, make_column_selector,make_column_transformer,ColumnTransformer




def loadprocesseddata():
    df = loaddata()
    dfsize = df.shape
    print(dfsize)
    trainsize = int(dfsize[0]*0.6)
    testsize = int(dfsize[0]*0.2)
    dftrain, dftest, dfdev= datasplit(df,trainsize,testsize)
    dftrain=fixfloat(dftrain)
    dftest=fixfloat(dftest)
    dfdev=fixfloat(dfdev)
    Xtrain_prepared, ytrain_prepared, Xtest_prepared, ytest_prepared, Xdev_prepared, ydev_prepared= prepare_for_train(dftrain, dftest,dfdev)

    print("Train X,y shape: ", Xtrain_prepared.shape, ytrain_prepared.shape)
    print("Test X,y shape: ", Xtest_prepared.shape, ytest_prepared.shape)
    print("Dev X,y shape: ", Xdev_prepared.shape, ydev_prepared.shape)

    return Xtrain_prepared,ytrain_prepared,Xtest_prepared,ytest_prepared,Xdev_prepared,ydev_prepared


def fixfloat(df):
    df['person_emp_length'] = df['person_emp_length'].replace('?', np.nan) 
    df['person_emp_length'] = df['person_emp_length'].replace('', np.nan)
    df['person_emp_length'] = df['person_emp_length'].astype(float)

    df['loan_int_rate'] = df['loan_int_rate'].replace('?', np.nan) 
    df['loan_int_rate'] = df['loan_int_rate'].replace('', np.nan)
    df['loan_int_rate'] = df['loan_int_rate'].astype(float)

    df['loan_int_rate'] = df['loan_int_rate'].replace('?', np.nan) 
    df['loan_int_rate'] = df['loan_int_rate'].replace('', np.nan)
    df['loan_int_rate'] = df['loan_int_rate'].astype(float)

    return df

def loaddata():
    df = pd.read_csv("LoanData.csv")
    return df


def get_outlier_indices(X):  
    model = LocalOutlierFactor()
    return model.fit_predict(X)


def handle_outlier(X, y):
    outlier_ind = get_outlier_indices(X)
    return X[outlier_ind == 1], y[outlier_ind == 1]

    
def add_columns(df):

    df["LifeWithCreditHist"] = df["cb_person_cred_hist_length"] / df['person_age']
    df["WealthToAge"] = df["person_income"] / df['person_age']

  
    df["5YearLoanToIncome"] = (
        (df["loan_amnt"] * (1 + df["loan_int_rate"] / 100) ** 5) / 
        df["person_income"] * 100
    )

    return df


def replace_columns(df):
    df['person_income'] = np.log(df['person_income'] +1)
    df['loan_amnt'] = np.log(df['loan_amnt'] +1)
    df['loan_percent_income'] = np.log(df['loan_percent_income'] +1)
    df['5YearLoanToIncome'] = np.log(df['5YearLoanToIncome'] +1)

    return df

def datasplit(df, train_size, test_size):
    
    df_train, df_testdev = train_test_split(df, train_size=train_size, stratify=df['loan_status'], random_state=42)
    df_dev, df_test = train_test_split(df_testdev, test_size=test_size, stratify=df_testdev['loan_status'], random_state=42)
    
    return df_train, df_test, df_dev



def prepare_for_train(dftrain, dftest, dfdev):
    set_config(transform_output="pandas")

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('ratio_adder', FunctionTransformer(add_columns)), 
        ('log_transformer', FunctionTransformer(replace_columns)),
        ('polynomial_features', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()) 
    ])
    

    cat_pipeline = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(sparse_output=False)
    )
     

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, make_column_selector(dtype_include=np.number)),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object))
    ])
    

    ytrain = dftrain['loan_status']
    ytest = dftest['loan_status'] 
    ydev = dfdev['loan_status'] 


    Xtrain = dftrain.drop(columns=['loan_status'])
    Xtest = dftest.drop(columns=['loan_status'])
    Xdev = dfdev.drop(columns=['loan_status'])  


    Xtrain_prepared = full_pipeline.fit_transform(Xtrain)
    Xtest_prepared = full_pipeline.transform(Xtest)
    Xdev_prepared = full_pipeline.transform(Xdev)


    Xtrain_prepared, ytrain_prepared = handle_outlier(Xtrain_prepared, ytrain)

    return Xtrain_prepared, ytrain_prepared, Xtest_prepared, ytest, Xdev_prepared, ydev



