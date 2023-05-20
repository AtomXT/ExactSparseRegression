library(glmnet)
N <- seq(80,220,5)
results = data.frame(matrix(ncol = 4, nrow = 0))
colnames(results) <- c('n','A','F','time')
for (n in N){
  print(n)
  p <- 2000
  k <- 10
  rho <- 0
  general_w <- TRUE
  
  
  data <- read.data(n, p, k, rho, general_w)
  X = data$X
  Y = data$Y
  w = data$w
  start_time <- Sys.time()
  fit <- glmnet(X, Y)
  
  lambdas <- fit$lambda[fit$df >= k]
  end_time <- Sys.time()
  TIME <- round(as.numeric(end_time - start_time, units="secs"), 4)
  lambda <- lambdas[1]
  w1 <- coef(fit, s = lambda)[2:(p+1)]
  correct <- t(sign(w)) %*% sign(w1)
  tpr <- correct/k
  fpr <- sum(w1[w==0]!=0)/(p-k)
  result <- list(A=tpr, F=fpr, time=TIME)
  print(result)
  results[nrow(results)+1, ] = c(n, tpr, fpr, TIME)
}

results

# write.csv(results, "lasso_results_figure1.csv", row.names = FALSE)
