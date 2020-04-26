# citation-analysis
Project repo for Computational Linguistics Team Laboratory at the University of Stuttgart


### Evaluation
we plan to implement and use ***f1_score*** metric for evaluation of our classifier

> F1 score is a weighted average of Precision and Recall(or Harmonic Mean between Precision and Recall).  
> The formula for F1 Score is:
> F1 = 2 * (precision * recall) / (precision + recall)

```python
eval.metrics.f1_score(y_true, y_pred, labels, average)
```
#### Parameters:
**y_true** : 1-d array or list of gold class values  
**y_pred** : 1-d array or list of estimated values returned by a classifier  
**labels** : list of labels/classes  
**average**: string - [None, 'micro', 'macro'] 
