set.seed(1)

###In this exercise we will be simulating data
###Which standard logistic regression
###Is expected to perform poorly in
###Do the other sigmoid improve performance?
###Probably not 

##Simulate Data
x1 <- rnorm(10000)
x2 <- rnorm(10000)
x3 <- rnorm(10000)
y <- ifelse(I(x1^2) - I(x2^2) + I(x3^2) > 0, 1, 0)
y_bad <- as.matrix(y)
xmat_bad <- cbind(rep(1, times = 10000), x1, x2, x3)

##Split into 50/50 training-testing
idx <- sample(1:10000, .5 * 10000, replace = F)
training <- xmat_bad[idx,]
training_response <- y_bad[idx]
testing <- xmat_bad[-idx, ]
testing_response <- y_bad[-idx]

##Define logistic function
logistic <- function(z) {
  #Note that this implementation assumes that 1
  #is contained within x, that is, in an n-dim
  #feature space, x is an Rn+1 vector including 
  #the intercept 
  as.numeric(1/(1 + exp(-z)))
}

##glm() benchmark
glm(y ~ x1 + x2 + x3, family = 'binomial')

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
test <- optim(c(0,0,0,0), fn = logistic_cost)
test2 <- optim(c(0,0,0,0), fn = tanh_cost)
test3 <- optim(c(0,0,0,0), fn = probit_cost)

##Predict on the test set and set up the confusion matrix
logistic_predictions <- ifelse(logistic(testing %*% test$par) > 0.5, 1, 0)
table(logistic_predictions, testing_response)
#True positive = 3563
#False positive = 1437
#False negative = 0
#Fscore = .83
#Classification error = 1437/5000

tanh_predictions <- ifelse(tanh(testing %*% test2$par) > 0.5, 1, 0)
table(tanh_predictions, testing_response)
#0 F-score
#1437/5000 Classification error

probit_predictions <- ifelse(pnorm(testing %*% test3$par) > 0.5, 1, 0)
table(probit_predictions, testing_response)
#Fscore = 0.83
#1437/5000 classification error 