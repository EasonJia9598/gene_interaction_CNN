library(knitrProgressBar)
# install.packages("knitrProgressBar")
library(parallel)
library(ape)
# install.packages("ape")
library(gplots)
# install.packages("gplots")
library(evolCCM)
# install.packages("devtools")
#library(devtools)
#devtools::install_github("beiko-lab/evolCCM")
# install.packages("progress")
library(progress)
library(dendextend)

random_tree_tips = 0 # 1 for random tree tips, 0 for fixed tree tips
per_runs = 100 # number of simulations per run
runs_in_one_loop = 100
t_s = 100 # tree size
log_name = 'Darwin_scenerios'

random_seeds = c(78572, 12321, 23123, 26675)
random_seed = random_seeds[3]

print(random_seed)
set.seed(random_seed)



start_number_of_genes = 5
end_number_of_genes = 5

sim_tree = rtree(n = t_s) # generate a random 100-tip tree
# rescale branch length to 0.1
sim_tree$edge.length = sim_tree$edge.length / mean(sim_tree$edge.length) * 0.1

simulation_profiles <- function(runs_in_one_loop, random_tree_tips, sim_tree2, file_id, n, t_s, per_runs, folder_path, all_names) {
  # R parallel environment uses its independent libraries
  library(evolCCM)
  library(ape)
  library(knitrProgressBar)
  library(dendextend)
  
  # if we need to have abitrary tree strcture
  
  file_id = as.integer(file_id)
  
  if (file_id == 1){
    progress_file_path <- paste0(folder_path, "/progress_file.log")
    set_kpb <- set_progress_mp(progress_file_path)
  }
  
  
  generations = per_runs / runs_in_one_loop 
  
  #line = c(1, 0,	1,	0,	0,	1,	0,	0	,0, 1) * 2
  #two_triangles = c(1	,1	,1,	0,	0,	1,	0,	0,	1,	1) * 2
  #star = c(0	,1,	1,	0,	0,	1,	0	,0	,1,	0) * 2
  #full_connected = c(1,	1,	1,	1,	1,	1,	1,	1,	1,	1) * 2

  
  for (number_of_files in 1:generations){
    rates = matrix(NA, nrow = runs_in_one_loop, ncol=(n + n + choose(n,2)))
    # Normal Profile
    if (random_tree_tips == 0){
      profiles = matrix(NA, t_s, n * runs_in_one_loop)
    }else{
      profiles = matrix(0.5, t_s, n * runs_in_one_loop)
    }
    

    # Profiles with Padding
    for (i in 1:runs_in_one_loop){
        if (file_id == 1){
            update_progress(set_kpb)
        }
        
        if (random_tree_tips == 0){
            # Normal Profile
            sim_tree = rtree(n = t_s) # generate a random 100-tip tree
        }else{
            # Profile with padding
            number_of_tips_random = sample(400:1000, 1)
            sim_tree = rtree(n = number_of_tips_random) # generate a random 100-tip tree with padding
        }
        # rescale branch length to 0.1
        sim_tree$edge.length = sim_tree$edge.length / mean(sim_tree$edge.length) * 0.1
        
        # assigning parameter values.
        # The parameters should be in a reasonable scale otherwise the simulated profiles may have all 0s or 1s.
        alpha <- runif(n, -0.5, 1) # assign n random intrinsic rates.
        B <- matrix(0, n, n)
        type = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0) * 0.5
        #b12 = runif(choose(n,2), -1.5, 1.5)
        b12 = type
        B[upper.tri(B)] <- b12
        B[lower.tri(B)] <- t(B)[lower.tri(B)]
        diag(B) <- runif(n, -0.5,0.5) # assign half the difference between gain and loss rates for each gene

        rates[i,] = c(alpha, diag(B), B[upper.tri(B)]) 
        # simulate 5 random profiles
        simDF <- SimulateProfiles(sim_tree, alpha, B)

        # change a random clade of a Darwin's scenario to 1's
        Darwin_simDF  <- simDF
        dend = TreeToDend(sim_tree)

        repeat{
            end_repeat = 0
            # Cut the dendrogram to get a sub-branch
            cutting_height <- runif(1, min = 0.2, max = 2)
            sub_branch <- cutree(dend, h = cutting_height)  # Adjust h as needed for the height
            number_of_branches = max(sub_branch)
            for(i_branches in 1:number_of_branches) {
                random_clade_index = sample(1:max(sub_branch), 1)
                sub_branch_df = as.data.frame(sub_branch)
                # Specify the row names to pick
                rows_to_pick <- row.names(sub_branch_df)[which(sub_branch_df$sub_branch == random_clade_index)]
                # Subset the dataframe based on the specified row names
                if (length(rows_to_pick) > 10 && length(rows_to_pick)  < 30){
                    end_repeat = 1
                    break
                }
            }
            if (end_repeat == 1){
                break
            }
        }

        total_row_names = rownames(Darwin_simDF)

        Darwin_simDF[rows_to_pick, 1:2] = 1
        rows_not_to_pick <- total_row_names[!(total_row_names %in% rows_to_pick)]
        Darwin_simDF[rows_not_to_pick, 1:2] = 0

        print(as.matrix(Darwin_simDF))

        if (random_tree_tips == 0){
            # Normal simulate profiles
            profiles[, i*n - (n-1):0] = as.matrix(Darwin_simDF) # store profiles
        }else{
            # Profile with Padding 
            padding_matrix = matrix(0.5, t_s, n)
            padding_matrix[1:number_of_tips_random, ] <- as.matrix(Darwin_simDF)
            profiles[, i*n - (n-1):0] = padding_matrix # store profiles
        }
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


# kpb_watch <- watch_progress_mp(2000, watch_location = " /scratch/h/honggu/zeshengj/CNN/data/simulations_data/progress_file.log")

#sim_tree = rtree(n = t_s) # generate a random 100-tip tree
# rescale branch length to 0.1
#sim_tree$edge.length = sim_tree$edge.length / mean(sim_tree$edge.length) * 0.1


for (number_of_gens in start_number_of_genes:end_number_of_genes){
  n_runs = detectCores() # number of simulations
  n = number_of_gens # number of genes in community
  # save results
  print(paste("Number of Genes:", n))
  print(paste("Number of Cores:", detectCores()))
  
  timestamp <- format(Sys.time(), "%Y_%m_%d_%H_%M_%S")
  
 

  

  if (random_tree_tips == 0){
    # Specify the path for the new folder
    folder_path <-sprintf("/scratch/h/honggu/zeshengj/CNN/data/simulations_data/%d_%s_normal_tips_%d_genes_tree_seed_%d_%d_records_%s",t_s, log_name, n, random_seed, n_runs * per_runs, timestamp)
  }else{
    # Specify the path for the new folder
    folder_path <-sprintf("/scratch/h/honggu/zeshengj/CNN/data/simulations_data/%d_%s_padding_tips_%d_genes_tree_seed_%d_%d_records_%s",t_s,log_name, n, random_seed, n_runs * per_runs, timestamp)
  }

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
    params[[i]] <- list(runs_in_one_loop = runs_in_one_loop, random_tree_tips = random_tree_tips, sim_tree = sim_tree, file_id = i, n = n, t_s = t_s, per_runs = per_runs, folder_path = folder_path, all_names = all_names)
  }
  
  # Initialize cluster
  cl <- makeCluster(n_runs)
  
  # Export function to all cluster nodes
  clusterExport(cl, c("simulation_profiles"))
  
  # Execute function in parallel
  results <- parLapply(cl, params, function(params) {
    simulation_profiles(params$runs_in_one_loop, params$random_tree_tips, params$sim_tree, params$file_id, params$n, params$t_s, params$per_runs, params$folder_path, params$all_names)
  })
  
  # Stop cluster
  stopCluster(cl)
  
}
