# Meet knn-sampler

knn-sampler is a non parametrical conditional density function estimator and sampler.
It is very useful when you need to estimate errors in a regression problem, since instead
of returning the regressor pointiwse prediction, it returns the predicted values distribution.

two estimators are avalible:

[x] KNNSampler - Based o KNN search algorithm (unsupervised)
[x] ForestSampler - Based on Forest algorithms (supervised)