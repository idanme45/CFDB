import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import numpy as np
import gc
import tqdm
def nop(it, *a, **k):
    return it

tqdm.tqdm = nop
from QUERYLANG import CFQL, Instances
from preprocessing import MultiColumnEncoder
import queries
from queries import get_prediction_queries, get_prediction_query
from itertools import product
import random
import re

import projection
from tqdm.notebook import tqdm as tqdm_notebook
import pickle
from bank_train_NOT_TO_GITHUB import BinaryClassification
import warnings
warnings.filterwarnings('ignore')


def get_classifier(q):
    return int(re.findall(r'\d+', q)[-2])

try:
    models = pickle.load(open('models/Credit/Credit_models_target_mads.pkl', 'rb'))
    instances = pickle.load(open('Instances/Credit/Credit_test_instances_target_mads.pkl', 'rb'))
    train_instances = pickle.load(open('Instances/Credit/Credit_train_instances_target_mads.pkl', 'rb'))
except EOFError as e:
    print(f"EOFError: {e}")

querylang_ref = CFQL(train_instances, models)
querylang = CFQL(instances, models)

mads = {}


def get_mad(x):
    return stats.median_absolute_deviation(x)


for f in querylang_ref.instances.continuous_features:
    mads[f] = get_mad(querylang_ref.instances.instances_relation[f])


def tabular_d(x, cf, mads=mads):
    n = x.shape[1]
    n_con = len(mads)
    n_cat = n - n_con
    d_con = 0
    d_cat = 0
    for f in querylang_ref.instances.feature_names:
        v1, v2 = x[f].values[0], cf[f].values[0]
        if f in querylang_ref.instances.continuous_features:
            d_con += (abs(v1 - v2) / mads[f])
        else:
            if not v1 == v2:
                d_cat += 1
    return (n_cat / n) * d_cat + (n_con / n) * d_con


uniform_prediction='''
SELECT * 
FROM (
    SELECT *, ROW_NUMBER() OVER(partition by Gender ORDER BY RANDOM() DESC) AS 'IdInGroup'
    FROM Instances, Predictions
    WHERE Instances.instanceId = Predictions.instanceId
      AND Predictions.ClassifierId = 0
      AND Predictions.Label = 0
    ) AS T
WHERE T.'IdInGroup'<=3
'''


cfs_view = (queries.AND_NOT,{'features':['Gender']})
querylang.create_cfs_view(cf_type='CecCFs', prediction_query=uniform_prediction, cfs_query=cfs_view)


Q_efficient = f''' 
select *
FROM  my_Prediction_CFs_1, Predictions,Instances
Where Predictions.PredictionId = my_Prediction_CFs_1.PredictionId 
and Instances.InstanceId = Predictions.InstanceId
'''
output = querylang.batch_execute(Q_efficient, tabular_d, 1, max_CF_to_check=5)
print(5)
#
# features_args1 = {
#     'OR':['Gender','EDUCATION'],
#     'AND':['EDUCATION','Marital_status'],
#     'AND_NOT':['Marital_status', 'EDUCATION'],
#     'NOT_AND':['Gender','Marital_status'],
#     'IMPLIES': {'f1':'Marital_status', 'f2':'EDUCATION'},
#     'Q7':{'f1':'Amount','f2':'EDUCATION'},
#     'Q8':{'f1':'Marital_status','f2':'AGE','vals':[21,22,23]},
#     'Q9':{'f1':'AGE','min_val':35.5,'max_val':39.5},
#     'Q10':{'f1':'EDUCATION','val':'university'}
# }
#
# def get_CFs_Queries(features):
#     myCfsQueries =[None,
#                (queries.OR,{'features':features['OR']}),
#                (queries.AND,{'features':features['AND']}),
#                (queries.AND_NOT,{'features':features['AND_NOT']}),
#                (queries.NOT_AND,{'features':features['NOT_AND']}),
#                (queries.IMPLIES,features['IMPLIES']),
#                (queries.Q7,features['Q7']),
#                (queries.Q8,features['Q8']),
#                (queries.Q9,features['Q9']),
#                (queries.Q10,features['Q10'])
#                   ]
#     return myCfsQueries
#
# def full_batch_evaluation(size, metric, t, hashing, atom_implication):
#     querylang = QueryLang(instances, models)
#     myPredictionQueries = [queries.get_prediction_query(size=size,classifier_id=0,prediction=1,target=instances.outcome_name,label=0)]
#     myCfsQueries =  get_CFs_Queries(features_args1)
#     myCfTypes = ['CecCFs']
#     for cf_type, prediction_q, cfs_q in product(myCfTypes, myPredictionQueries, myCfsQueries):
#         querylang.create_cfs_view(cf_type=cf_type, prediction_query=prediction_q, cfs_query=cfs_q,verbose=False)
#     myQuery = ' Union '.join([f'SELECT * FROM  my_cfs_{i}' for i in range(1,len(myCfsQueries)+1)])
#     return querylang.batch_execute(myQuery,metric, t, hashing, atom_implication)
#
# def l0_metric(x,y):
#     x = x.to_numpy().reshape(-1)
#     y = y.to_numpy().reshape(-1)
#     return sum(x!=y)
#
# o = full_batch_evaluation(1, l0_metric, 5, hashing=True, atom_implication=True)