import numpy as np
import pandas as pd

from preprocessor_1 import preprocess
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
from logistic_regression import predict, train, loss
from utils import info_gain
from datetime import datetime

startTime = datetime.now()

data = preprocess()

data.insert(0, 'Ones', 1)

training_set, test_set = train_test_split(data, test_size=0.2, random_state=44)

# info_gain(training_set)

y_training = training_set.pop('Churn').to_frame().to_numpy()
x_training = training_set.to_numpy()

y_test = test_set.pop('Churn').to_frame().to_numpy()
x_test = test_set.to_numpy()

w = train(x_training, y_training, no_of_iterations=100000)
z = np.dot(x_test, w)
h_test = np.tanh(z)
h_test = predict(h_test)
test_set_error = loss(h_test, y_test, y_test.shape[0])
print('Error rate in test set = {}'.format(test_set_error))
print('The script took {0} second !'.format(datetime.now() - startTime))
