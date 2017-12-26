K-fold cross-validation
======

Cross-validation, sometimes called rotation estimation, is a model validation technique for assessing how the results of a statistical analysis will generalize to an independent data set. 
It is mainly used in settings where the goal is prediction, and one wants to estimate how accurately a predictive model will perform in practice. One round of cross-validation involves partitioning a sample of data into complementary subsets, performing the analysis on one subset (called the training set),
and validating the analysis on the other subset (called the validation set or testing set). To reduce variability, multiple rounds of cross-validation are performed using different partitions, and the validation results are averaged over the rounds.

Cross-validation is important in guarding against testing hypotheses suggested by the data, especially where further samples are hazardous, costly or impossible to collect 

![Kfold](https://upload.wikimedia.org/wikipedia/commons/f/fc/4fold3class.jpg)


Statistical properties
======

Suppose we choose a measure of fit F, and use cross-validation to produce an estimate F* of the expected fit EF of a model to an independent data set drawn from the same population as the training data. If we imagine sampling multiple independent training sets following the same distribution, the resulting values for F* will vary. The statistical properties of F* result from this variation.

The cross-validation estimator F* is very nearly unbiased for EF. The reason that it is slightly biased is that the training set in cross-validation is slightly smaller than the actual data set (e.g. for LOOCV the training set size is n − 1 when there are n observed cases). In nearly all situations, the effect of this bias will be conservative in that the estimated fit will be slightly biased in the direction suggesting a poorer fit. In practice, this bias is rarely a concern.

The variance of F* can be large.[8][9] For this reason, if two statistical procedures are compared based on the results of cross-validation, it is important to note that the procedure with the better estimated performance may not actually be the better of the two procedures (i.e. it may not have the better value of EF). Some progress has been made on constructing confidence intervals around cross-validation estimates,[8] but this is considered a difficult problem.


Head distribution ration on supervision process related with the final precision
======

![Headistribution](https://raw.github.com/rmaestre/K-fold-cross-validation/master/results/results.png)
