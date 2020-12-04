## Logistic Regression Simulation
library(ggplot2)
set.seed(1)
x <- rnorm(1000)
xb <- 2*x
p <- 1/(1+exp(-xb))
y <- rbinom(n = 1000, size = 1, prob = p)
set <- data.frame(cbind(x, y))
set

ggplot(data = set) + 
  aes(x, y) + 
  geom_point(col = 'blue', alpha = 1/5) +
  geom_line(aes(y = p), size = 1.5)


##Benchmark using pre-packaged logistic regression
glm(y~x, family = "binomial", data = set)

##Dataset with only the intercept term and any predictors used
##Note: for this method of logistic regression the data MUST
##Be presented in this format, an n x k+1 matrix, n = num
##of observations and k being number of predictors
##The first column MUST be a vector in Rn of all 1s to
##represent learning the intercept
prd <- cbind(rep(1, times = nrow(set)), set$x)

##Response as a matrix
##Even though the response
##Is only a column it still 
##Needs to be of the matrix type
response <- as.matrix(set$y)

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
##Compute F score
logistic_predictions <- ifelse(logistic(testing %*% test$par) > 0.5, 1, 0)
logistic_confusion <- table(logistic_predictions, testing_response)
logistic_confusion
logistic_fscore <- logistic_confusion[2, 2]/(logistic_confusion[2,2]+0.5*(logistic_confusion[1,2] + logistic_confusion[2,1]))
logistic_fscore
logistic_error <- (logistic_confusion[1, 2] + logistic_confusion[2, 1])/nrow(testing)
logistic_error

tanh_predictions <- ifelse(tanh(testing %*% test2$par) > 0.5, 1, 0)
tanh_confusion <- table(tanh_predictions, testing_response)
tanh_confusion
tanh_fscore <- tanh_confusion[2, 2]/(tanh_confusion[2,2]+0.5*(tanh_confusion[1,2] + tanh_confusion[2,1]))
tanh_fscore
tanh_error <- (tanh_confusion[1, 2] + tanh_confusion[2, 1])/nrow(testing)
tanh_error

probit_predictions <- ifelse(pnorm(testing %*% test3$par) > 0.5, 1, 0)
probit_confusion <- table(probit_predictions, testing_response)
probit_confusion
probit_fscore <- probit_confusion[2, 2]/(probit_confusion[2,2]+0.5*(probit_confusion[1,2] + probit_confusion[2,1]))
probit_fscore
probit_error <- (logistic_confusion[1, 2] + logistic_confusion[2, 1])/nrow(testing)
probit_error

ggplot() + aes(c('Logistic', 'Hyperbolic Tangent', 'Probit'), c(logistic_error, tanh_error, probit_error)) +
  geom_col(fill = 'lightblue', width = 0.3) +
  xlab('Sigmoid Function') +
  ylab('Classification Error') +
  ylim(c(0, 0.4)) +
  ggtitle('Classification Error by Sigmoid Function')

ggplot() + aes(c('Logistic', 'Hyperbolic Tangent', 'Probit'), c(logistic_fscore, tanh_fscore, probit_fscore)) +
  geom_col(fill = 'lightblue', width = 0.3) +
  xlab('Sigmoid Function') +
  ylab('F Score') +
  ylim(c(0, 1)) +
  ggtitle('F Score by Sigmoid Function')





