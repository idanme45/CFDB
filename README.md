# Counterfactual Query Language (CFQL) Framework

## Overview

The Counterfactual Query Language (CFQL) framework is designed to facilitate the generation and execution of Counterfactual  for interpretable machine learning explanations. It provides a flexible and expressive way to query counterfactuals based on specific criteria.

### Requirements 
networkx
z3-solver
tqdm
dice_ml
torch
pandas
numpy
concurrent
sqlite3
sqlparse

## Usage:

### Import the necessary modules:
```python
   from QUERYLANG import CFQL, Instances
   import ....
```
### Set up your machine learning models and instances.

```python
cfql = CFQL(instances, models)
```

### Define a prediction query using SQL syntax.
```python
prediction_query = '''
SELECT Predictions.PredictionId 
    FROM Instances, Predictions
    WHERE Instances.instanceId = Predictions.instanceId
      AND Predictions.ClassifierId = 0
      AND Predictions.Label = 0
'''
```

### Define a counterfactual search query 
#### This part demonstare loading CFQL queries of interest. By doing so, users can instantiate these queries and leverage the capabilities of CFQL for generating counterfactual explanations based on their requirements


```python
# counterfactuals that do not change the gender and  the race.
cfs_not_and = (queries.NOT_AND, {'features': ['gender', 'race']})

# This view focuses on instances where the model's prediction is 0 and the classifier ID is 0.
# The counterfactuals generated will not modify the 'gender' and 'race' features.
cfql.create_cfs_view(cf_type='CecCFs', prediction_query=prediction_query, cfs_query=cfs_not_and)
```

##### The `create_cfs_view` method generates interpretable counterfactual view and prints their corresponding name. These views can then be queried to retrieve insightful information about counterfactuals.

### execute the query

```python
Q = """ 
SELECT * 
FROM view_name
"""

# Execute the query using the regular execute
cfql.execute(Q)
```


### Alternatively, execute the query using the batch version

- **Threshold:** The threshold parameter allows you to set a limit on the number of counterfactuals retrieved for each query. It facilitates the re-use of pre-computed counterfactuals, improving efficiency.

- **Metric:** The metric parameter enables you to select a specific evaluation metric for assessing the quality of the generated counterfactuals. It plays a crucial role in determining how well the counterfactuals align with the desired characteristics.

```python
cfql.batch_execute(Q, metric=l0_metric, threshold=2)
```


