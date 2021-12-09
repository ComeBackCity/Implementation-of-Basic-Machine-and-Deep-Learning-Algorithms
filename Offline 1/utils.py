from scipy.stats import entropy
from numpy.random import uniform
from pandas import DataFrame
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import json


# information gain library function
def info_gain(df: DataFrame):
    all_features = list(df.columns)
    y_feature = all_features.pop(len(all_features) - 1)
    data = df.copy()
    y = data.pop(y_feature)
    x = data
    importances = mutual_info_classif(x, y)
    info_gain_map = {}
    for feature, gain in zip(all_features, importances):
        info_gain_map[feature] = gain

    print(json.dumps(info_gain_map, indent=4))

# h function
# def h_func(df: DataFrame, feature: str):
#     p = len(df[df[feature] == 1.0])
#     n = df.shape[0] - p
#     q = p / (p + n)
#     h = entropy([q, 1 - q], base=2)
#     return p, n, h
#
#
# # %%
#
# # information gain
# def info_gain(X: DataFrame):
#     all_features = list(X.columns)
#     y_feature = all_features[len(all_features) - 1]
#     p, n, h_output = h_func(X, y_feature)
#     feature_gain_map = dict()
#     for feature in all_features:
#         remainder = 0
#         if feature == y_feature:
#             continue
#         values = X[feature].unique()
#         if len(values) > 2:
#             col_min = X[feature].min()
#             col_max = X[feature].max()
#             divider = uniform(col_min, col_max)
#             # divider = X[feature].median()
#             groups = [X[X[feature] <= divider], X[X[feature] > divider]]
#             for group in groups:
#                 pk, nk, h_k = h_func(group, y_feature)
#                 remainder += ((pk + nk) / (p + n)) * h_k
#         else:
#             for _, group in X.groupby([feature]):
#                 pk, nk, h_k = h_func(group, y_feature)
#                 remainder += ((pk + nk) / (p + n)) * h_k
#         feature_gain_map[feature] = h_output - remainder
#
#     print(json.dumps(feature_gain_map, indent=4))


# One-Hot encoding
def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    dummies = dummies.iloc[:, :-1]
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return res
