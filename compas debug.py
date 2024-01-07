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
    models = pickle.load(open('models/Compas/compas_models_target_mads_v2.pkl', 'rb'))
    instances = pickle.load(open('Instances/Compas/compas_test_instances_target_mads.pkl', 'rb'))
except EOFError as e:
    print(f"EOFError: {e}")

querylang = CFQL(instances, models)


size = 25
myPredictionQueries = get_prediction_queries(size=size,target=instances.outcome_name,classifier_ids=range(len(models)))


myCfsQueries =[None,
               (queries.OR,{'features':['Gender','CrimeDegree']}),
               (queries.AND,{'features':['Gender','CrimeDegree']}),
               (queries.AND_NOT,{'features':['Race', 'Age', 'Gender', 'CrimeDegree']}),
               (queries.NOT_AND,{'features':['Race', 'CrimeDegree']}),
               (queries.IMPLIES,{'f1':'CrimeDegree', 'f2':'PriorsCount'}),
               (queries.Q7,{'f1':'PriorsCount','f2':'CrimeDegree'}),
               (queries.Q8,{'f1':'Race','f2':'PriorsCount','vals':[0,1,2]}),
               (queries.Q9,{'f1':'PriorsCount','min_val':2,'max_val':5}),
               (queries.Q10,{'f1':'Race','val':'Native_American'})]

#myCfsQueries =[(queries.Q8,{'f1':'balance','f2':'age','vals':[18,19,20]}),]
myCfTypes = ['CecCFs','GrowingSpheresCFs','DiverseCFs']
#myCfTypes = ['GrowingSpheresCFs']
for cf_type, prediction_q, cfs_q in product(myCfTypes, myPredictionQueries, myCfsQueries):
    if cf_type == 'DiverseCFs' and get_classifier(prediction_q) == 2:
        continue
    if cf_type != 'DiverseCFs' and get_classifier(prediction_q) == 3:
        continue
    querylang.create_cfs_view(cf_type=cf_type, prediction_query = prediction_q, cfs_query=cfs_q)

gc.collect()
myQuery = f'''
SELECT *
FROM  my_cfs_76
'''
querylang.execute(myQuery, parallel=False, verbose = True)