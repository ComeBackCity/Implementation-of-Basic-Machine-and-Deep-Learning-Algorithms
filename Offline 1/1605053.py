#!/usr/bin/env python
# coding: utf-8

# In[69]:


# imports
import math
import random

from pandas import DataFrame
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from IPython.core.display import display
import json


# In[70]:


# set random seed
random.seed(a=2)
np.random.seed(5)


# In[71]:


# information gain function
def info_gain(df: DataFrame):
    all_features = list(df.columns)
    y_feature = all_features.pop(len(all_features) - 1)
    data = df.copy()
    y = data.pop(y_feature)
    x = data
    importances = mutual_info_classif(x, y)
    info_gain_map = {
        feature: gain for feature, gain in zip(all_features, importances)
    }

    info_gain_map = {k: v for k, v in sorted(info_gain_map.items(), key=lambda item: item[1], reverse=True)}
    return list(info_gain_map.keys())


# In[72]:


# One-Hot encoding
def encode_and_bind(original_dataframe, feature_to_encode):
    return pd.get_dummies(
        original_dataframe, columns=feature_to_encode, drop_first=True
    )


# In[73]:


# pre-processor 1
def read_telco_data():
    return pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv',
                             converters={
                                 'gender': lambda x: int(x == 'Female'),
                                 'Partner': lambda x: int(x == 'Yes'),
                                 'Dependents': lambda x: int(x == 'Yes'),
                                 'PhoneService': lambda x: int(x =='Yes'),
                                 'PaperlessBilling': lambda x: int(x =='Yes'),
                                 'Churn': lambda x: int(x =='Yes'),
                             })

def process_telco_data(telco_data):
    telco_data.drop('customerID', axis=1, inplace=True)
    telco_data = telco_data.astype({
        'tenure': int,
        "MonthlyCharges": float,
        "TotalCharges": float
    }, errors="ignore")

    total_charges_median = (telco_data['TotalCharges'].loc[telco_data['TotalCharges'] != ' ']).median()
    telco_data['TotalCharges'].replace([' '], total_charges_median, regex=True, inplace=True)

    columns_to_encode = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                         'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                         'StreamingMovies', 'Contract', 'PaymentMethod']
    # for column in columns_to_encode:
    #     telco_data = encode_and_bind(telco_data, column)

    telco_data = encode_and_bind(telco_data, columns_to_encode)

    # Move final column for better visualization
    telco_data.insert(len(telco_data.columns)-1, 'Churn', telco_data.pop('Churn'))

    all_columns = list(telco_data.columns)
    telco_data[all_columns] = MinMaxScaler().fit_transform(telco_data[all_columns])

    return telco_data

def preprocess_telco_data():
    telco_data = read_telco_data()
    telco_data = process_telco_data(telco_data)
    telco_data.to_csv('telco.csv')
    return telco_data


# In[74]:


# pre processor 2
def read_adult_data(file_name, pos):
    column_names = ['C'+str(i) for i in range(15)]
    return pd.read_csv(file_name,
                         names=column_names,
                         header=None,
                         sep=' *, * ',
                         engine="python",
                         converters={
                            'C9': lambda x: float(x == 'Male'),
                            'C14': lambda x: float(x == pos)
                         })

def process_adult_data(adult_data):
    all_columns = list(adult_data.columns)
    missing_value_columns = [
        column
        for column in all_columns
        if '?' in adult_data[column].values.tolist()
    ]

    for column in missing_value_columns:
        adult_data[column].replace(['?'], adult_data[column].mode(), inplace=True)

    columns_to_encode = ['C1', 'C3', 'C5', 'C6', 'C7', 'C8', 'C13']

    # for column in columns_to_encode:
    #     adult_data = encode_and_bind(adult_data, column)

    adult_data = encode_and_bind(adult_data, columns_to_encode)

    adult_data.insert(len(adult_data.columns)-1, 'C14', adult_data.pop('C14'))

    all_columns = list(adult_data.columns)
    adult_data[all_columns] = MinMaxScaler().fit_transform(adult_data[all_columns])
    # adult_data[all_columns] = StandardScaler().fit_transform(adult_data[all_columns])

    return adult_data

def preprocess_adult_data():
    adult_data = read_adult_data('adult.csv', '>50K')
    adult_test = read_adult_data('adult.test.csv', '>50K.')

    data_size = adult_data.shape[0]
    frames = [adult_data, adult_test]
    df = pd.concat(frames)

    df = process_adult_data(df)
    adult_data, adult_test = df.iloc[0:data_size, :], df.iloc[data_size: , :]

    adult_data.to_csv('adult-data.csv')
    adult_test.to_csv('adult-test.csv')

    return adult_data, adult_test


# In[75]:


# pre processor 3
def read_cc_data():
    return pd.read_csv('creditcard.csv')

def process_cc_data(cc_data: DataFrame):
    positive_data = cc_data.loc[cc_data['Class'] == 1]
    negative_data = cc_data.loc[cc_data['Class'] == 0]

    negative_sub_data = negative_data.sample(n=20000, replace=False, random_state=5)

    frames = [positive_data, negative_sub_data]
    cc_data = pd.concat(frames)
    cc_data = cc_data.reset_index(drop=True)

    all_columns = list(cc_data.columns)
    cc_data[all_columns] = MinMaxScaler().fit_transform(cc_data[all_columns])

    return cc_data

def preprocess_cc_data():
    cc_data = read_cc_data()
    cc_data = process_cc_data(cc_data)
    cc_data.to_csv('cc.csv')
    return cc_data


# In[76]:


# loss function
def loss(y_predicted, y_actual, size):
    return mean_squared_error(y_actual, y_predicted)
    # return np.sum((y_actual - y_predicted) ** 2) / size


# In[77]:


def statistics(y_predict, y_actual):
    matrix = confusion_matrix(y_actual, y_predict)
    tn, fp, fn, tp = matrix.ravel()
    print('Accuracy is {}'.format(accuracy(y_predict, y_actual)))
    print('Sensitivity is {}'.format(tp/(tp+fn)))
    print('Specificity is {}'.format(tn/(tn+fp)))
    print('Precision is {}'.format(tp/(tp+fp)))
    print('False discovery rate {}'.format(fp/(fp+tp)))
    print('F1 score {}'.format(2*tp/(2*tp+fp+fn)))


# In[78]:


# accuracy function
def accuracy(y_predicted, y_actual):
    return accuracy_score(y_actual, y_predicted)


# In[79]:


# prediction function for determining label of hypothesis
def predict(hypothesis):
    labels = np.array([1.0 if it > 0.0 else -1.0 for it in hypothesis])
    labels = labels.reshape((labels.shape[0], 1))
    return labels


# In[80]:


# logistic regression
def train(x, y, early_terminate_threshold=0.0, learning_rate=0.0001, no_of_iterations=10000):
    no_of_data, no_of_features = x.shape
    w = np.random.rand(no_of_features, 1)
    # w = np.zeros((no_of_features, 1))
    for _ in range(no_of_iterations):
        z = x @ w
        h = np.tanh(z)
        error = loss(h, y, no_of_data)
        if error < early_terminate_threshold:
            break
        gradient = x.T @ ((y - h) * (1 - h ** 2))
        w += learning_rate * gradient / no_of_data

    return w


# In[81]:


# resample function for adaboost
def resample(x, y, w):
    indices = np.random.choice(x.shape[0], x.shape[0], replace=True, p=w )
    x_data = x[indices]
    y_data = y[indices]
    return x_data, y_data


# In[82]:


# Adaboost
def adaboost(example_x, example_y, k):
    no_of_data = example_x.shape[0]
    w = np.array([1/no_of_data] * no_of_data)
    h = []
    z = []
    for _ in range(k):
        x_data, y_data = resample(example_x, example_y, w)
        w_learn = train(x_data, y_data, early_terminate_threshold=0.8, learning_rate=0.01, no_of_iterations=10000)
        h_k = np.tanh(np.dot(example_x, w_learn))
        h_k = predict(h_k)
        error = sum(w[j] for j in range(no_of_data) if h_k[j] != example_y[j])
        if error > 0.5:
            continue

        for j in range(no_of_data):
            if h_k[j] == example_y[j]:
                w[j] = w[j] * (error / (1-error))

        w /= np.sum(w)
        h.append(w_learn)
        # z.append(math.log((1-error)/error, 2))
        z.append(np.log((1-error)/error))

    return h, z


# In[83]:


def logistic_regression_test(training_x, training_y, test_x, test_y, threshold, learning_rate=0.01, no_of_iterations = 10000):
    w_logi = train(training_x, training_y, early_terminate_threshold=threshold, learning_rate=learning_rate, no_of_iterations=no_of_iterations)
    h_logi = np.tanh(np.dot(test_x, w_logi))
    h_logi = predict(h_logi)

    h_train = np.tanh(np.dot(training_x, w_logi))
    h_train = predict(h_train)
    # print('Logistic regression accuracy {}.'.format(accuracy(h_logi, test_y)))
    print('Test set stats = ')
    statistics(y_predict=h_logi, y_actual=test_y)
    print('Train set stats = ')
    statistics(y_predict=h_train, y_actual=training_y)


# In[84]:


def adaboost_test(training_x, training_y, test_x, test_y, k):
    h_ada, z_ada = adaboost(training_x, training_y, k)

    hypo_test = np.zeros(test_y.shape)
    hypo_train = np.zeros(training_y.shape)
    for _h, _z in zip(h_ada, z_ada):
        l_test = np.tanh(np.dot(test_x, _h))
        l_train = np.tanh(np.dot(training_x, _h))
        hypo_test += _z * l_test
        hypo_train += _z * l_train

    # hypo /= sum(z_ada)

    hypo_test = predict(hypo_test)
    hypo_train = predict(hypo_train)
    print('Adaboost accuracy for test set k = {} is {}.'.format(k, accuracy(hypo_test, test_y)))
    print('Adaboost accuracy for training set k = {} is {}.'.format(k, accuracy(hypo_train, training_y)))


# In[85]:


# telco data
data = preprocess_telco_data()


# In[86]:


# data


# In[87]:


# Churn data full
# data.insert(0, 'Ones', 1.0)
# data = data.to_numpy()
#
# data_x = data[:, :-1]
# data_y = data[:, -1]
#
# data_y = np.array([1.0 if it > 0 else -1.0 for it in data_y])
# data_y = data_y.reshape((data_y.shape[0], 1))
#
# x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=10)


# In[88]:


final_column = data.columns[-1]
train_dataset, test_dataset = train_test_split(data, test_size=0.2, random_state=10)

columns = info_gain(train_dataset)
feature_cutoff = 10
columns_to_use = columns[0:feature_cutoff]
# print(columns_to_use)
columns_to_use.append(final_column)
reduced_training, reduced_test = train_dataset[columns_to_use], test_dataset[columns_to_use]

reduced_training.insert(0, 'Ones', 1.0)
reduced_training = reduced_training.to_numpy()

reduced_test.insert(0, 'Ones', 1.0)
reduced_test = reduced_test.to_numpy()

x_train = reduced_training[:, :-1]
y_train = reduced_training[:, -1]
y_train = np.array([1.0 if it > 0 else -1.0 for it in y_train])
y_train = y_train.reshape((y_train.shape[0], 1))

x_test = reduced_test[:, :-1]
y_test = reduced_test[:, -1]
y_test = np.array([1.0 if it > 0 else -1.0 for it in y_test])
y_test = y_test.reshape((y_test.shape[0], 1))


# In[89]:


# x_train


# In[90]:


# y_train


# In[91]:


# x_test


# In[92]:


# y_test


# In[93]:


logistic_regression_test(x_train, y_train, x_test, y_test, 0.5)


# In[94]:


# for i in range(1, 5):
#     adaboost_test(x_train, y_train, x_test, y_test, i*5)


# In[95]:


# adult data
training_set, test_set = preprocess_adult_data()


# In[96]:


# training_set


# In[97]:


# test_set


# In[98]:


# training_set.insert(0, 'Ones', 1.0)
# training_set = training_set.to_numpy()
# #
# test_set.insert(0, 'Ones', 1.0)
# test_set = test_set.to_numpy()
#
# x_train = training_set[:, :-1]
# y_train = training_set[:, -1]
# #
# x_test = test_set[:, :-1]
# y_test = test_set[:, -1]
# #
# y_train = np.array([1.0 if it > 0 else -1.0 for it in y_train])
# y_train = y_train.reshape((y_train.shape[0], 1))
# #
# y_test = np.array([1.0 if it > 0 else -1.0 for it in y_test])
# y_test = y_test.reshape((y_test.shape[0], 1))


# In[99]:


final_column = training_set.columns[-1]
#
columns = info_gain(training_set)
feature_cutoff = 35
columns_to_use = columns[0:feature_cutoff]
# columns_to_use = columns[-feature_cutoff:]
# print(columns_to_use)
columns_to_use.append(final_column)
reduced_training, reduced_test = training_set[columns_to_use], test_set[columns_to_use]
#
reduced_training.insert(0, 'Ones', 1.0)
reduced_training = reduced_training.to_numpy()
#
reduced_test.insert(0, 'Ones', 1.0)
reduced_test = reduced_test.to_numpy()
#
x_train = reduced_training[:, :-1]
y_train = reduced_training[:, -1]
y_train = np.array([1.0 if it > 0 else -1.0 for it in y_train])
y_train = y_train.reshape((y_train.shape[0], 1))
#
x_test = reduced_test[:, :-1]
y_test = reduced_test[:, -1]
y_test = np.array([1.0 if it > 0 else -1.0 for it in y_test])
y_test = y_test.reshape((y_test.shape[0], 1))


# In[100]:


logistic_regression_test(x_train, y_train, x_test, y_test, 0.5)


# In[101]:


# for i in range(1, 5):
#     adaboost_test(x_train, y_train, x_test, y_test, i*5)


# In[102]:


# credit card data
data = preprocess_cc_data()


# In[103]:


# data


# In[104]:


# credit card data full
# data.insert(0, 'Ones', 1.0)
# data = data.to_numpy()
#
# data_x = data[:, :-1]
# data_y = data[:, -1]
#
# data_y = np.array([1.0 if it > 0.0 else -1.0 for it in data_y])
# data_y = data_y.rescc_data()hape((data_y.shape[0], 1))
#
# x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=40)


# In[105]:


final_column = data.columns[-1]
train_dataset, test_dataset = train_test_split(data, test_size=0.2, random_state=30)
#
columns = info_gain(train_dataset)
feature_cutoff = 10
columns_to_use = columns[0:feature_cutoff]
# print(columns_to_use)
columns_to_use.append(final_column)
reduced_training, reduced_test = train_dataset[columns_to_use], test_dataset[columns_to_use]
#
reduced_training.insert(0, 'Ones', 1.0)
reduced_training = reduced_training.to_numpy()
#
reduced_test.insert(0, 'Ones', 1.0)
reduced_test = reduced_test.to_numpy()
#
x_train = reduced_training[:, :-1]
y_train = reduced_training[:, -1]
y_train = np.array([1.0 if it > 0 else -1.0 for it in y_train])
y_train = y_train.reshape((y_train.shape[0], 1))
#
x_test = reduced_test[:, :-1]
y_test = reduced_test[:, -1]
y_test = np.array([1.0 if it > 0 else -1.0 for it in y_test])
y_test = y_test.reshape((y_test.shape[0], 1))


# In[106]:


# x_train


# In[107]:


# x_test


# In[108]:


# y_train


# In[109]:


# y_test


# In[110]:


logistic_regression_test(x_train, y_train, x_test, y_test, 0.5)


# In[111]:


# for i in range(1, 5):
#     adaboost_test(x_train, y_train, x_test, y_test, i*5)

