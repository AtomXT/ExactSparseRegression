
setwd("C:/Users/xu-to/OneDrive - Northwestern University/Candidacy/Algorithm/lasso")
all.data.path <- "C:/Users/xu-to/OneDrive - Northwestern University/Candidacy/Algorithm/data"


read.data <- function(n, p, k, rho, general_w)
{
  if (general_w) {
    data.name <- paste("general_data", n,p,k,rho, sep="_")
    Y.name <- paste("general_Y", n,p,k,rho, sep="_")
    w.name <- paste("general_w_true", n,p,k,rho, sep="_")
  } else {
    data.name <- paste("data", n,p,k,rho, sep="_")
    Y.name <- paste("Y", n,p,k,rho, sep="_")
    w.name <- paste("w_true", n,p,k,rho, sep="_")
  }
  
  
  data.path <- paste(data.name, ".txt", sep="")
  Y.path <- paste(Y.name, ".txt", sep="")
  w.path <- paste(w.name, ".txt", sep="")
  X <- as.matrix(read.table(paste(all.data.path, data.path, sep='/'), header=FALSE))
  Y <- as.matrix(read.table(paste(all.data.path, Y.path, sep='/'), header=FALSE))
  w <- as.matrix(read.table(paste(all.data.path, w.path, sep='/'), header=FALSE))
  return(list(X=X,Y=Y,w=w))
}






