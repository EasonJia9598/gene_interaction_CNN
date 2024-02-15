library(knitrProgressBar)
library(parallel)
library(ape)
# install.packages("devtools")
#library(devtools)
#devtools::install_github("beiko-lab/evolCCM")

# install.packages("progress")
library(progress)




start_number_of_genes = 5
end_number_of_genes = 5

simulation_profiles <- function(file_id, n, t_s, per_runs, folder_path, all_names) {
  library(evolCCM)
  library(ape)
  library(knitrProgressBar)
  
  # if we need to have abitrary tree strcture
  
  file_id = as.integer(file_id)
  
  if (file_id == 1){
    progress_file_path <- paste0(folder_path, "/progress_file.log")
    set_kpb <- set_progress_mp(progress_file_path)
  }
  
  runs_in_one_loop = 2500
  generations = per_runs / runs_in_one_loop 
  
  line = c(1, 0,	1,	0,	0,	1,	0,	0	,0, 1) * 2
  two_triangles = c(1	,1	,1,	0,	0,	1,	0,	0,	1,	1) * 2
  star = c(0	,1,	1,	0,	0,	1,	0	,0	,1,	0) * 2
  full_connected = c(1,	1,	1,	1,	1,	1,	1,	1,	1,	1) * 2

  
  for (number_of_files in 1:generations){
    rates = matrix(NA, nrow = runs_in_one_loop, ncol=(n + n + choose(n,2)))
    profiles = matrix(NA, t_s, n * runs_in_one_loop)
    
    for (i in 1:runs_in_one_loop){
      if (file_id == 1){
        update_progress(set_kpb)
      }
      
      t_s = 100 # tree size
      sim_tree = rtree(n = t_s) # generate a random 100-tip tree
      # rescale branch length to 0.1
      sim_tree$edge.length = sim_tree$edge.length / mean(sim_tree$edge.length) * 0.1
      
      
      # set parameters
      alpha = runif(n, -0.5, 0.5)
      B <- matrix(0, n, n)
      #b12 = rnorm(choose(n,2), -0.5, 1)
      # Ass some probability mass at zero
      # Estimate a sparse network,
      # A mixture distribitoni of point mass at zero and uniform(-1, 1) 
      # p = 0.3 to be 0, and 0.7 stay the same
      
      b12 = runif(choose(n,2), -1, 1)
      #b12 <- ifelse(runif(length(b12), 0, 1) < 0.3, 0, b12)
      
      #b12 = star
      
      
      B[upper.tri(B)] <- b12
      B[lower.tri(B)] <- t(B)[lower.tri(B)]
      diag(B) = runif(n, -0.3,0.3)
      # save rates
      rates[i,] = c(alpha, diag(B), B[upper.tri(B)]) 
      # simulate profiles
      simDF = SimulateProfiles(sim_tree, alpha, B)
      profiles[, i*n - (n-1):0] = as.matrix(simDF) # store profiles
    }
    
    
    # print(head(rates))
    # print(head(profiles))
    rates <- data.frame(rates)
    colnames(rates) <- all_names
    
    # Construct the file name with variable n and timestamp
    #rates_file_name <- sprintf("%s/%d_rates_data_%d_genes_%d_records.csv", folder_path, file_id * 100 + number_of_files, n, runs_in_one_loop)
    #profiles_file_name <- sprintf("%s/%d_profiles_data_%d_genes_%d_records.csv", folder_path, file_id * 100 + number_of_files,  n, runs_in_one_loop)
    
    rates_file_name <- sprintf("%s/raw_data/rates/%d_rates.csv", folder_path, file_id * 100 + number_of_files)
    profiles_file_name <- sprintf("%s/raw_data/profiles/%d_profiles.csv", folder_path, file_id * 100 + number_of_files)
    
    write.csv(rates, rates_file_name, quote=F, row.names=F)
    write.csv(profiles, profiles_file_name, quote=F, row.names=F)
    print("\nSaving Success!")
  
  }
}


# kpb_watch <- watch_progress_mp(2000, watch_location = "/scratch/zjia/multiprocess/progress_file.log")




set.seed(78506572)
#set.seed(123)
t_s = 100 # tree size
#sim_tree = rtree(n = t_s) # generate a random 100-tip tree
# rescale branch length to 0.1
#sim_tree$edge.length = sim_tree$edge.length / mean(sim_tree$edge.length) * 0.1


for (number_of_gens in start_number_of_genes:end_number_of_genes){
  n_runs = 40 # number of simulations
  per_runs = 50000 # number of simulations per run
  n = number_of_gens # number of genes in community
  # save results
  print(n)
  
  timestamp <- format(Sys.time(), "%Y_%m_%d_%H_%M_%S")
  
  # Specify the path for the new folder
  folder_path <-sprintf("/scratch/zjia/multiprocess/uniform_random_tree_%d_genes_%d_records_%s", n, n_runs * per_runs, timestamp)
  raw_data_path <- paste0(folder_path, '/raw_data')
  rates_path <- paste0(folder_path,'/raw_data/rates')
  profile_path <-paste0(folder_path, '/raw_data/profiles')
  
  # Create the folder
  dir.create(folder_path)
  dir.create(raw_data_path)
  dir.create(rates_path)
  dir.create(profile_path)
  
  
  # Check if the folder was created successfully
  if (file.exists(folder_path)) {
    cat("Folder created successfully at:", folder_path, "\n")
  } else {
    cat("Failed to create folder at:", folder_path, "\n")
  }
  
  # Generate alpha names
  alpha_names <- paste0("alpha_", 1:n)
  
  # Generate beta names
  beta_names <- character(choose(n, 2) + n)
  count <- 1
  
  # Add terms like beta_11, beta_22, ..., beta_nn
  for (i in 1:n) {
    beta_names[count] <- paste0("beta_", i, i)
    count <- count + 1
  }
  
  # Add remaining beta terms
  for (i in 1:(n-1)) {
    for (j in (i+1):n) {
      beta_names[count] <- paste0("beta_", i, j)
      count <- count + 1
    }
  }
  
  # Combine alpha and beta names into a single list, removing empty strings
  all_names <- c(alpha_names, beta_names[beta_names != ""])
  
  
  # Define parameters
  params <- list()
  
  # Create 40 sets of parameters where x and y are random numbers
  for (i in 1:n_runs) {
    params[[i]] <- list(file_id = i, n = n, t_s = t_s, per_runs = per_runs, folder_path = folder_path, all_names = all_names)
  }
  
  # Initialize cluster
  cl <- makeCluster(detectCores())
  
  # Export function to all cluster nodes
  clusterExport(cl, c("simulation_profiles"))
  
  # Execute function in parallel
  results <- parLapply(cl, params, function(params) {
    simulation_profiles(params$file_id, params$n, params$t_s, params$per_runs, params$folder_path, params$all_names)
  })
  
  # Stop cluster
  stopCluster(cl)
  
}
