SELECT *
FROM CFs AS C, Prediction_CFs AS PC,
     Instances AS I
WHERE CFs.CfId = PC.CfId
  AND PC.PredictionId = pred.PredictionId
  AND I.InstanceId = pred.InstanceId
  AND C.Race = I.Race
  AND C.Gender = I.Gender

/* A query that requests counterfactuals (CFs) without altering the race and gender attributes */