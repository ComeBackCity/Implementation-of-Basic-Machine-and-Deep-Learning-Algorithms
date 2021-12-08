#!/usr/bin/env python
# coding: utf-8

# In[11]:


# imports
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import utils


# In[12]:


# Reading data
def read_data():
    telco_data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv',
                             converters={
                                 'gender': lambda x: int(x == 'Female'),
                                 'Partner': lambda x: int(x == 'Yes'),
                                 'Dependents': lambda x: int(x == 'Yes'),
                                 'PhoneService': lambda x: int(x == 'Yes'),
                                 'MultipleLines': lambda x: int(x == 'Yes'),
                                 'OnlineSecurity': lambda x: int(x == 'Yes'),
                                 'OnlineBackup': lambda x: int(x == 'Yes'),
                                 'DeviceProtection': lambda x: int(x == 'Yes'),
                                 'TechSupport': lambda x: int(x == 'Yes'),
                                 'StreamingTV': lambda x: int(x == 'Yes'),
                                 'StreamingMovies': lambda x: int(x == 'Yes'),
                                 'PaperlessBilling': lambda x: int(x == 'Yes'),
                                 'Churn': lambda x: int(x == 'Yes'),
                                 'MonthlyCharges': lambda x: float(x)
                             })

    return telco_data


# In[13]:


# Preprocessing
def process_data(telco_data):
    telco_data.drop('customerID', axis=1, inplace=True)
    telco_data = telco_data.astype({
        'tenure': int,
        "MonthlyCharges": float,
        "TotalCharges": float
    }, errors="ignore")

    total_charges_median = (telco_data['TotalCharges'].loc[telco_data['TotalCharges'] != ' ']).median()
    telco_data['TotalCharges'].replace([' '], total_charges_median, regex=True, inplace=True)

    columns_to_encode = ['InternetService', 'Contract', 'PaymentMethod']
    for column in columns_to_encode:
        telco_data = utils.encode_and_bind(telco_data, column)

    # Move final column for better visualization
    telco_data.insert(len(telco_data.columns) - 1, 'Churn', telco_data.pop('Churn'))

    all_columns = list(telco_data.columns)
    telco_data[all_columns] = MinMaxScaler().fit_transform(telco_data[all_columns])

    return telco_data


# In[14]:


def preprocess():
    telco_data = read_data()
    telco_data = process_data(telco_data)
    return telco_data
