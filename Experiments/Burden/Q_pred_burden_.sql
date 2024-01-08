SELECT T.PredictionId
FROM (SELECT *, ROW_NUMBER() OVER(partition by gender, race ORDER BY RANDOM() DESC) AS idx
     FROM Instances AS I, Predictions AS P
     WHERE I.instanceId = P.instanceId
       AND P.Label = 'unfavorable' ) AS T
WHERE T.idx<=503

/* A query that selects a sample of 503 instances from
each group (gender, race) (503 being the size of the smallest group) to address data
imbalance */