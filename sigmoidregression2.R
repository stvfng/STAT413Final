library(ISLR)
data(Default)
set.seed(1)

##Recode response to numeric
Default$binary_def <- ifelse(Default$default == 'Yes', 1, 0)

##Benchmark using pre-packaged logistic regression
glm(default ~ balance, family = "binomial", data = Default)

##Dataset with only the intercept term and any predictors used
##Note: for this method of logistic regression the data MUST
##Be presented in this format, an n x k+1 matrix, n = num
##of observations and k being number of predictors
##The first column MUST be a vector in Rn of all 1s to
##represent learning the intercept
prd <- cbind(rep(1, times = nrow(Default)), Default$balance)

##Response as a matrix
##Even though the response
##Is only a column it still 
##Needs to be of the matrix type
response <- as.matrix(Default$binary_def)

##Randomly split data into training and testing 
idx <- sample(1:nrow(prd), .5 * nrow(prd), replace = F)
training <- prd[idx,]
training_response <- response[idx]
testing <- prd[-idx, ]
testing_response <- response[-idx]

##Define logistic function
logistic <- function(z) {
  #Note that this implementation assumes that 1
  #is contained within x, that is, in an n-dim
  #feature space, x is an Rn+1 vector including 
  #the intercept 
  as.numeric(1/(1 + exp(-z)))
}

##Define cross-entropy loss for logistic, hyperbolic tangent, and probit,
##The loss functions are all the same and can be minimized using optim()
##from built-in stats package. Note that only parameter is weights, which
##is the beta-vector. All 3 functions minimize the log cross-entropy loss
##on the training set, which is hard-coded into the function
logistic_cost <- function(weights) {
  num_obvs <- nrow(training)
  pred <- logistic(training %*% weights)
  log_loss <- sum((-training_response * log(pred)) - ((1 - training_response) * log(1 - pred)))
  return(log_loss / num_obvs)
}

tanh_cost <- function(weights) {
  num_obvs <- nrow(training)
  pred <- 1/2 * tanh(training %*% weights) + 1/2
  log_loss <- sum((-training_response * log(pred)) - ((1 - training_response) * log(1 - pred)))
  return(log_loss / num_obvs)
}

probit_cost <- function(weights) {
  num_obvs <- nrow(training)
  pred <- pnorm(training %*% weights)
  log_loss <- sum((-training_response * log(pred)) - ((1 - training_response) * log(1 - pred)))
  return(log_loss / num_obvs)
}

##Use gradient descent or other optimization 
##Algorithm using optim(). This finds the wrights
##Which minimize the log-loss and can be accessed
##Through test$par
test <- optim(c(0,0), fn = logistic_cost)
test2 <- optim(c(0,0), fn = tanh_cost)
test3 <- optim(c(0,0), fn = probit_cost)

##Predict on the test set and set up the confusion matrix
logistic_predictions <- ifelse(logistic(testing %*% test$par) > 0.5, 1, 0)
logistic_confusion <- table(logistic_predictions, testing_response)
logistic_confusion
logistic_fscore <- logistic_confusion[2, 2]/(logistic_confusion[2,2]+0.5*(logistic_confusion[1,2] + logistic_confusion[2,1]))
logistic_fscore #43.5-

tanh_predictions <- ifelse(tanh(testing %*% test2$par) > 0.5, 1, 0)
tanh_confusion <- table(tanh_predictions, testing_response)
tanh_confusion
tanh_fscore <- tanh_confusion[2, 2]/(tanh_confusion[2,2]+0.5*(tanh_confusion[1,2] + tanh_confusion[2,1]))
tanh_fscore #20.225

probit_predictions <- ifelse(pnorm(testing %*% test3$par) > 0.5, 1, 0)
probit_confusion <- table(probit_predictions, testing_response)
probit_confusion
probit_fscore <- probit_confusion[2, 2]/(probit_confusion[2,2]+0.5*(probit_confusion[1,2] + probit_confusion[2,1]))
probit_fscore #41.112


##============================================
##============================================
##Cross Validation
##============================================
##Define general logistic function
general <- function(m, z) {
  as.numeric(1/(1 + exp(-m * z)))
}
##Define different values of m to be cross validated over
slopes <- c(1e-2, 1e-1, 1, 1e1, 1e2)

##Randomly shuffle the rows in training
shuffle <- sample(nrow(training))
shuffle_training <- training[shuffle, ]
shuffle_response <- training_response[shuffle]
##Split into five folds for five fold CV
folds <- cut(seq(1, nrow(training)), breaks = 5, labels = F)

##Matrix to hold all of the test errors 
error_mat <- matrix(NA, nrow = 5, ncol = 5)

##Error rate function
err <- function(tbl) {
  (tbl[1,2] + tbl[2, 1]) / sum(tbl)
}

##Perform cross-validation
for(i in 1:5) {
  ##Segment data
  test_idx <- which(folds == i, arr.ind = T)
  test_cv <- shuffle_training[test_idx, ]
  train_cv <- shuffle_training[-test_idx, ]
  test_response_cv <- shuffle_response[test_idx]
  train_response_cv <- shuffle_response[-test_idx]
  
  ##Define/train ALL 5 COST FUNCTIONS i hate R so much
  logistic_cost_001 <- function(weights) {
    num_obvs <- nrow(train_cv)
    pred <- general(1e-2, train_cv %*% weights)
    log_loss <- sum((-train_response_cv * log(pred)) - ((1 - train_response_cv) * log(1 - pred)))
    return(log_loss / num_obvs)
  }
  
  logistic_cost_01 <- function(weights) {
    num_obvs <- nrow(train_cv)
    pred <- general(1e-1, train_cv %*% weights)
    log_loss <- sum((-train_response_cv * log(pred)) - ((1 - train_response_cv) * log(1 - pred)))
    return(log_loss / num_obvs)
  }
  
  logistic_cost_1 <- function(weights) {
    num_obvs <- nrow(train_cv)
    pred <- general(1, train_cv %*% weights)
    log_loss <- sum((-train_response_cv * log(pred)) - ((1 - train_response_cv) * log(1 - pred)))
    return(log_loss / num_obvs)
  }
  
  logistic_cost_10 <- function(weights) {
    num_obvs <- nrow(train_cv)
    pred <- general(10, train_cv %*% weights)
    log_loss <- sum((-train_response_cv * log(pred)) - ((1 - train_response_cv) * log(1 - pred)))
    return(log_loss / num_obvs)
  }
  
  logistic_cost_100 <- function(weights) {
    num_obvs <- nrow(train_cv)
    pred <- general(100, train_cv %*% weights)
    log_loss <- sum((-train_response_cv * log(pred)) - ((1 - train_response_cv) * log(1 - pred)))
    return(log_loss / num_obvs)
  }
  
  ##Again, for all 5 cost functions, optimize
  optim1 <- optim(c(0,0), fn = logistic_cost_001)
  optim2 <- optim(c(0,0), fn = logistic_cost_01)
  optim3 <- optim(c(0,0), fn = logistic_cost_1)
  optim4 <- optim(c(0,0), fn = logistic_cost_10)
  optim5 <- optim(c(0,0), fn = logistic_cost_100)
  
  ##Perform predictions/confusion matrices for all five scenarios
  predict_001 <- ifelse(general(1e-2, test_cv %*% optim1$par) > 0.5, 1, 0)
  predict_01 <- ifelse(general(1e-1, test_cv %*% optim2$par) > 0.5, 1, 0)
  predict_1 <- ifelse(general(1e0, test_cv %*% optim3$par) > 0.5, 1, 0)
  predict_10 <- ifelse(general(1e1,test_cv %*% optim4$par) > 0.5, 1, 0)
  predict_100 <- ifelse(general(1e2, test_cv %*% optim5$par) > 0.5, 1, 0)
  
  table1 <- table(predict_001, test_response_cv)
  table2 <- table(predict_01, test_response_cv)
  table3 <- table(predict_1, test_response_cv)
  table4 <- table(predict_10, test_response_cv)
  table5 <- table(predict_100, test_response_cv)
  
  ##Calculate error rate and add to table
  errs <- c(err(table1), err(table2), err(table3), err(table4), err(table5))
  error_mat[i,] <- errs
  
  print("done")
}

#Then examine error_matrix...
View(error_mat)
