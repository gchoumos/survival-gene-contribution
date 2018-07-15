
# TODO
#-----
# 1. Install libraries if needed.
# 2. Path of setwd
# 3. Use a threshold for NaNs of any observation so that sample 54 for example gets dropped automatically.
# 4. No need to create new data frames for the train and test data as we can just use the train and test
#    indices instead. Have a look at this and change it if it applies. 
set.seed(27)
setwd("/home/gchoumo/Dropbox/my_docs/msc_ds/trimester_6/statistics_for_big_data/project/survival-gene-contribution/")

# I manually updated the spreadsheet. I created a new row where I keep the value of the survival in months.
# In addition I removed the "gene name" column as it had too many NaNs and it wasn't of any actual use.
# Then I saved it as a csv and I also removed the spaces (not really worth mentioning I guess).

# It turns out that sample 54 holds too many NaN values, so I guess we should drop it.
# I dropped it

# Now in R we are able to read the csv
data = read.table('all_edited.csv',sep=',',header=FALSE,na.strings='NaN')

# The previous has 24481 rows (which are actually the predictors). Let keep only the complete
# cases, ie. the ones that don't have any null values.
data = data[complete.cases(data),]


# Keep the first 100 rows in a small dataframe so that we can easily process it for now.
#data_small = head(data,100)
# The below is ugly and temporary.
data_small = data

# The resulting dataframe included 21907 rows. This means that ~ 2500 predictors (genes) were thrown.
# Now we can transpose it. Later do the same for the big one too.
df_small = data.frame(t(data_small))


# The header names should be the first row values (the following 2 lines are doing it - R sucks)
colnames(df_small) = as.character(unlist(df_small[1,]))
df_small =  df_small[-1,]

# After all the previous there are no incomplete cases (the following should return 0)
sum(!complete.cases(df_small))

# Convert the Survival_Months to integer
df_small$Survival_Months = as.integer(as.character(df_small$Survival_Months))
# Now convert it to a factor if the months are > 60 (more than 5 years)
df_small$Survival_Months = as.factor(ifelse(df_small$Survival_Months>60,1,0))

# Convert all the rest of the columns to numeric (doubles)
for (i in c(2:ncol(df_small))) {
    df_small[,i] = as.numeric(as.character(df_small[,i]))
}

# Split into train and test (I'll get 80% for train - no special thought behind this though)

train = sample(1:nrow(df_small),as.integer(0.8*nrow(df_small)))
test = (-train)

# Get the train data and test data (for the small dataframe now)
train_data = df_small[train,]
test_data = df_small[test,]

# Load the glmnet library
library(glmnet)

# Let's now split the predictors and the target. Get the train data (for later) and all
#x_train = model.matrix(Survival_Months~.,train_data)
#y = train_data$Survival_Months
x_train = as.matrix(train_data[,-1])
y_train = train_data$Survival_Months

x_test = as.matrix(test_data[,-1])
y_test = test_data$Survival_Months

x = as.matrix(df_small[,-1])
y = df_small$Survival_Months


###
## HERE ON ISL
###

# Let's keep a lambda vector to test various values
#lambdas = 10^seq(10,-2,length=20)

# The lasso model
#lasso_model = glmnet(x[train,],y[train], alpha=1, lambda=lambdas, intercept=FALSE)


# We are setting family to binomial. This actually equals to logistic regression.
# Then, setting alpha to 1 means that we are performing lasso.
# The default measure will be the deviance. You can override it to mean squared error for example
# by passing also type.measure = 'mse'
lasso_cv <- cv.glmnet(x,y,alpha=1,family='binomial',intercept=FALSE)
plot(lasso_cv)

# Get the best value of lambda
bestl = lasso_cv$lambda.1se
bestl

# Get the indices of the features that lasso kept
coeffs = which(ifelse(coef(lasso_cv,s=bestl) > 0,TRUE,FALSE))
coeffs_names = rownames(coeffs)[which(coeffs!=0)]
coeffs_names

# Time to fit a logistic regression using the coefficients that lasso suggested.
# Get a new matrix with the informative predictors only
x_train_inf = x_train[,c(coeffs_names)]
x_test_inf = x_test[,c(coeffs_names)]
logreg = glm(y_train~.,data=cbind(x_train_inf,y_train),family=binomial)

# Predictions
preds = ifelse(predict(logreg,x_test_inf,type='response') > 0.5,1,0)

# Create a confusion matrix
table(preds,y_test)
