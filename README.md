### Gene Contribution on Breast Cancer Survival
This repository is the implementation of an exercise for the _Statistics for Big Data_ course for the MSc in Data Science of Athens University of Economics and Business.

We are provided with a dataframe that includes information about 78 deceased patients who were suffering from breast cancer. In particular, for each patient we have: 

* Their survival (in months)
* their age
* an indication about their ERP
* part of their genetic profile (~25.000 genes)

The aim of this analysis is to identify the genes that lead to the most significant contribution with 
regards to survival. To carry out this task we will be using LASSO and Logistic Regression.

More details to come.


#### TO-DO
* `glmnet` with alpha=1 actually translates to Lasso. There is no need to create a logistic regression model afterwards. We can use the predict function with `type='coefficients'` to get the coefficient estimates. Then we can run `predict` again, this time with `type='response'` in order to get predictions for the test data.
* Moooore
