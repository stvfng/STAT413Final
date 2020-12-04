##Figure 3 on report 


xrange <- seq(-5,5, length.out = 1e3)
general_logistic <- function(k = 1, z){
  as.numeric(1 / (1 + exp(-k*z)))
}
plot(0, 0, xlim = c(-5, 5), ylim = c(0,1), type = 'n', xlab = 'Value of predictor', ylab = 'Predicted probability of success')
lines(xrange, general_logistic(k = 0, z = xrange), col = 'red')
lines(xrange, general_logistic(k = 1e6, z = xrange), col = 'blue')
lines(xrange, general_logistic(k = 1, z = xrange), col = 'goldenrod3')
legend("bottomright", legend = c('c2 = 0', 'c2 = 1', 'c2 = 1e6'), col=c('red', 'goldenrod3', 'blue'), pch=1)

