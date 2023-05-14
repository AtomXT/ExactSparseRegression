library(glmnet)

all.data.path <- "C:/Users/xu-to/OneDrive - Northwestern University/Candidacy/Algorithm/data"

n <- 10000
p <- 1000
k <- 50
rho <- 0.1
data.name <- paste("data", n,p,k,rho, sep="_")
data.path <- paste(data.name, ".txt", sep="")
Y.name <- paste("Y", n,p,k,rho, sep="_")
Y.path <- paste(Y.name, ".txt", sep="")
w.name <- paste("w_true", n,p,k,rho, sep="_")
w.path <- paste(w.name, ".txt", sep="")
X <- as.matrix(read.table(paste(all.data.path, data.path, sep='/'), header=FALSE))
Y <- as.matrix(read.table(paste(all.data.path, Y.path, sep='/'), header=FALSE))
w <- as.matrix(read.table(paste(all.data.path, w.path, sep='/'), header=FALSE))

start_time <- Sys.time()
fit <- glmnet(X, Y)
lambdas <- fit$lambda[fit$df == k]
end_time <- Sys.time()
TIME <- round(as.numeric(end_time - start_time, units="secs"), 1)
lambda <- lambdas[1]
w1 <- coef(fit, s = 0.9)[2:(p+1)]
correct <- t(w) %*% sign(w1)
print(correct/k)
print(TIME)

