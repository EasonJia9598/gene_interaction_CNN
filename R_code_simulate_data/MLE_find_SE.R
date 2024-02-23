library(ape)
library(evolCCM)
library(progress)

set.seed(123)

t_s = 100 # tree size
sim_tree = rtree(n = t_s) # generate a random 100-tip tree
# rescale branch length to 0.1
sim_tree$edge.length = sim_tree$edge.length / mean(sim_tree$edge.length) * 0.1


EstimateCCM_updated_version <- function(profiles, phytree, ip=0.1, pen=0.5,  ...){
  ### Main function for parameter estimation based on `ace()` from `ape` package.
  if (nrow(profiles) != length(phytree$tip.label)){
    stop("the profile matrix size doesn't match the number of tips on the tree.")
  }
  
  if (ncol(profiles)>7){
    warning("It may take much longer to estimate a large community. Consider to split it into small subcommunities.")
  }
  
  #print('Trigger program 1 ')
  
  profiles = profiles[phytree$tip.label,]
  n = ncol(profiles)
  lvls = apply(expand.grid(replicate(n, c(0,1), simplify = F)),1,paste0, collapse="")
  x <- apply(profiles, 1, paste0, collapse="")
  x = factor(x, levels = lvls)
  nl <- nlevels(x)
  x <- as.integer(x)
  
  #print('Trigger program 2')
  
  # extract information from tree
  phy=phytree
  nb.tip <- length(phy$tip.label)
  nb.node <- phy$Nnode
  liks <- matrix(0, nb.tip + nb.node, nl)
  TIPS <- 1:nb.tip
  liks[cbind(TIPS, x)] <- 1
  phy <- reorder(phy, "postorder")
  e1 <- phy$edge[, 1]
  e2 <- phy$edge[, 2]
  EL <- phy$edge.length
  
  ## construct likelihood function and estimate parameters
  dev <- function(p) {
    if (any(is.nan(p)) || any(is.infinite(p)))
      return(1e+50)
    comp <- numeric(nb.tip + nb.node)
    C = matrix(0, nrow=n, ncol=n)
    
    # assign initial values
    p0s = p[1:n]
    gldifs = p[(n+1):(2*n)]
    C[upper.tri(C, diag=F)] = p[(2*n+1):length(p)]
    diag(C) = gldifs
    C[lower.tri(C)] = t(C)[lower.tri(C)]
    Q = ConstructQunsym(C, p0s)
    Q[Q == Inf] <- 1e13
    Q[Q == -Inf] <- -1e13
    diag(Q) <- -rowSums(Q)
    decompo <- eigen(Q)
    lambda <- decompo$values
    GAMMA <- decompo$vectors
    invGAMMA <- solve(GAMMA)
    #pb <- progress_bar$new(total = nb.node)
    for (i in seq(from = 1, by = 2, length.out = nb.node)) {
      j <- i + 1L
      anc <- e1[i]
      des1 <- e2[i]
      des2 <- e2[j]
      v.l <- GAMMA %*% diag(exp(lambda * EL[i])) %*%
        invGAMMA %*% liks[des1, ]
      v.r <- GAMMA %*% diag(exp(lambda * EL[j])) %*%
        invGAMMA %*% liks[des2, ]
      v <- v.l * v.r
      # change
      liks[anc,] <- v
      # end of change
      comp[anc] <- sum(v)
      #pb$tick()
    }
    #nonNeg_comp <- ifelse(comp <= 0, 1e-13, comp)
    
    stable <- GAMMA[,abs(lambda)<1e-10]
    dev_value <- -2 * log(sum(v *  stable / sum(stable)))
    
    #print(paste('sum(GAMMA[, 1])', sum(stable)))
    #print(paste('lambda', lambda))
    #print(paste('GAMMA',stable))
    #print(paste('v * GAMMA[, 1]', v * stable))
    #print(paste('v', v))
    #print(paste('dev_value', dev_value))
    
    
    #dev_value <- -1 * sum(log(nonNeg_comp[-TIPS]))
    penalty = pen*sum( p^2)
    dev_value = dev_value + penalty
    if (is.na(dev_value))
      Inf
    else dev_value
    
  }
  
  #print('Trigger program 3 ')
  
  # estimate the rates
  np=n*(n+1)/2 + n
  obj <- list()
  if (length(ip) == 1){
    ip = rep(ip, length.out = np)
  }
  
  #print('Trigger program nlm')
  iter.history <- capture.output(
    out <- nlminb(ip, function(p) dev(p),
                  control = list(...)), split=T)
  
  obj$loglik <- -out$objective/2
  obj$rates <- out$par
  #print('Trigger program nlm 2')
  out.nlm <- try(nlm(function(p) dev(p), p = obj$rates, iterlim = 1, stepmax = 0, hessian = TRUE), silent = TRUE)
  #print('Trigger program') 
  Cestimate = matrix(0, nrow=n, ncol=n)
  Cestimate[upper.tri(Cestimate,diag=F)] = obj$rates[(2*n+1):length(obj$rates)]
  Cestimate[lower.tri(Cestimate)] = t(Cestimate)[lower.tri(Cestimate)]
  diag(Cestimate) = obj$rates[(n+1):(2*n)]
  
  return(list(alpha = obj$rates[1:n], B = Cestimate, nlm.par=out$par, nlm.converge = out$convergence, nlm.hessian=out.nlm$hessian,dev=dev))
  
}

# construct Q matrix
ConstructQunsym <- function(C, c0s){
  n = nrow(C)
  mutdif = diag(C)
  diag(C) = 0
  statesM = expand.grid(replicate(n, c(0,1), simplify = F))
  Q = matrix(0, nrow=nrow(statesM), ncol=nrow(statesM))
  rownames(Q) <- colnames(Q)<- apply(statesM, 1, paste0, collapse=",")
  
  ### convert binary statesM to {-1, 1}
  for (i in 1:nrow(statesM)) {
    statei = as.numeric(statesM[i, ])
    
    repi =  matrix(rep(statei,length(statei)), nrow=length(statei), byrow = T)
    
    diag(repi) = 1 -  diag(repi)
    rowlable = paste0(statei, collapse=",")
    collable = apply(repi,1, paste0, collapse=",")
    
    diag(repi) = statei
    # same states:
    sm = (repi == statei)
    dm = (!sm)
    
    qs = (c0s - ifelse(statei==0,-1,1)*mutdif)  - diag(sm %*% C) + diag(dm %*% C)
    qs = exp(qs)
    Q[rowlable, collable] = qs
  }
  
  return(Q)
}





n_runs = 50 # number of simulations

n = 5 # number of genes in community

nrun  = n_runs



network_type_rates = 0.5
line = c(1, 0,	1,	0,	0,	1,	0,	0	,0, 1) * network_type_rates
two_triangles = c(1	,1	,1,	0,	0,	1,	0,	0,	1,	1) * network_type_rates
star = c(0	,1,	1,	0,	0,	1,	0	,0	,1,	0) * network_type_rates
full_connected = c(1,	1,	1,	1,	1,	1,	1,	1,	1,	1) * network_type_rates

types_name = c("line", "two_triangles", "star", "full_connected")
b12_matrix_data <- cbind(line, two_triangles, star, full_connected)

folder_saving_name = '100_new_version/'

print("Start to create folders")
for (type_index in 1:4){
  saving_folder_path = paste0("/scratch/zjia/data/test_data/prediction/", folder_saving_name, types_name[type_index])
  dir.create(saving_folder_path)
  
}

print("Finish creating folders")

for (type_index in 1:4){
  pb <- progress_bar$new(format = "[:bar] :current/:total (:percent) eta: :eta", total = n_runs)
  # save results
  rates = matrix(NA, nrow=n_runs, ncol=(n + n + choose(n,2)))
  profiles = matrix(NA, t_s, n * n_runs)
  estP <- matrix(NA, nrow=n_runs, ncol=(n + n + choose(n,2)))
  estSE <- matrix(NA, nrow=n_runs, ncol=(n + n + choose(n,2)))
  covRates <- c()
  
  for (i in 1:n_runs){
    # set parameters
    alpha = runif(n, -0.5, 0.5)
    #alpha = c(-0.47762521 ,-0.14494056,  0.30724990 ,-0.35663234,  0.02839111)
    B <- matrix(0, n, n)
    # Fixed rate
    #b12 = rnorm(choose(n,2), -0.5, 0.5)
    #b12 = runif(choose(n,2), -1, 1)
    #b12 <- ifelse(runif(length(b12), 0, 1) < 0.3, 0, b12)
    b12 = b12_matrix_data[,type_index]
    B[upper.tri(B)] <- b12
    B[lower.tri(B)] <- t(B)[lower.tri(B)]
    diag(B) = runif(n, -0.3,0.3)
    #diag(B) = c(0.14553070  ,0.08843931, -0.13292997 ,-0.19241466  ,0.12495385)
    # save rates
    rates[i,] = c(alpha, diag(B), B[upper.tri(B)]) 
    # simulate profiles
    simDF = SimulateProfiles(sim_tree, alpha, B)
    profiles[, i*n - (n-1):0] = as.matrix(simDF) # store profiles
    
    #print('start to estimate')
    aE <- EstimateCCM_updated_version(profiles=simDF, phytree=sim_tree)
    #print('Estimated finish')
    estP[i,] <- c(aE$alpha, diag(aE$B), aE$B[upper.tri(aE$B)])
    #profiles[i, ] = as.vector(as.matrix(simDF)) # store profiles
    paE <- ProcessAE(aE)
    estSE[i,] <- paE$hessianSE
    # negative or very large convergence rates mean not good convergence
    covRates <- c(covRates, paE$rate)
    #print(i)
    pb$tick(1)
  }
  
  #boxplot(estP[, -c(1:(ncol(estP)-10))], main=paste0("Estimation of ",n_runs," simulations"))
  
  #print(head(rates))
  #print(head(profiles))
  

  saving_folder_path = paste0("/scratch/zjia/data/test_data/prediction/",folder_saving_name, types_name[type_index])
  write.csv(rates, paste0(saving_folder_path, "/rates_data.csv"), quote=F, row.names=F)
  write.csv(profiles,  paste0(saving_folder_path, "/profiles_data.csv"), quote=F, row.names=F)
  write.csv(estP,  paste0(saving_folder_path, "/estP.csv") , quote=F, row.names=F)
  write.csv(estSE,  paste0(saving_folder_path, "/estSE.csv"), quote=F, row.names=F)
}
#boxplot(estP, main=paste0("Estimation of ",nrun," simulations"))
#points(1:length(trueP), trueP, pch=8, col="red")






