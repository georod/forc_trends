#=================================================================================
# Trend Breakpoint Analysis using MODIS data - Using Bfast
#=================================================================================
# 2024-06-24
# Peter R.

# Notes:
#  - This script finds breaks (breakpoints) in MODIS EVI (250 m) time series
#  - This code is for running parallel processing in a high performance computer. It takes a couple of hours to complete.
#  - The computing environment is set with a separate SLURM bash file. 



start.time <- Sys.time()
start.time


#=================================
# Load libraries
# ================================

library(bfast)
library(foreach)
library(doParallel)

#install.packages("stlplus")
library(stlplus)




#=================================
# File paths and folders
# ================================

#setwd("~/projects/scripts/forc_trends") 
#setwd("C:/Users/Peter R/Documents/st_trends_for_c/algonquin")

dataf <- "~/projects/data" # data folder

outf3 <- paste0(dataf, "/forc_trends_pj/algonquin/output_h5p/EVI_500m/bfast/")

folder1 <- "EVI_500m"




#========================================
# Parallel processing settings
#========================================

# Use the environment variable SLURM_CPUS_PER_TASK to set the number of cores.
# This is for SLURM. Replace SLURM_CPUS_PER_TASK by the proper variable for your system.
# Avoid manually setting a number of cores.
ncores = Sys.getenv("SLURM_CPUS_PER_TASK") 

registerDoParallel(cores=ncores)# Shows the number of Parallel Workers to be used
print(ncores) # this how many cores are available, and how many you have requested.
#getDoParWorkers()# you can compare with the number of actual workers



#======================================
# Time series loop
#======================================

timeSleep <- 3

rTs2 <- readRDS(paste0("~/projects/data/forc_trends_pj/algonquin/output_h5p/", folder1, "/bfast/output10/rTs2.rds")) # Time series created in separate script


#-----------------------------------
# Run BFast
#-----------------------------------
  
    
allBrksbF0 <-  foreach (i=1:ncol(rTs2), .inorder=TRUE) %dopar% 
      
      {
        
        print(i) 
        
        bF0 <- try( bfast(
          rTs2[, i], 
          h = 0.05, 
          season= c("harmonic"), #c("dummy") # harmonic 
          max.iter = 10,
          breaks = NULL,
          hpc = "foreach",
          level=0.01, # significance level
          decomp = c("stlplus"),
          type = "OLS-MOSUM"
          
        ))
        
        if(class(bF0) == "try-error") { 
          # This is to jump pixels with all NA values. The number of NAs needs to match the vars found below in 'else {...}'
          
        as.data.frame(cbind(i, matrix(rep(NA, 9), nrow=1, ncol=9, byrow=FALSE)))
          
          
        } else {
          
          # forest cells with data
          
          # if trend break then
          if (bF0$nobp[[1]]==FALSE) {
            # all trend breaks 
            # var names: pixel, break, # of observations, # of interactions, time lower bound, time estimate, time upper bound, metric value right before break, metric values right after break, magnitude of break & direction
            as.data.frame(cbind(i, bF0$nobp[[1]], bF0$output[[length(bF0$output)]]$bp.Vt$nobs, length(bF0$output), 
                                matrix(cbind(time(rTs2[, i])[as.numeric(names(bF0$output[[length(bF0$output)]]$bp.Vt$y)[ifelse(as.vector(bF0$output[[length(bF0$output)]]$ci.Vt$confint) < 1, 1, ifelse(as.vector(bF0$output[[length(bF0$output)]]$ci.Vt$confint) > 460, 460, as.vector(bF0$output[[length(bF0$output)]]$ci.Vt$confint) ))])], as.vector(bF0$Mags) ), 
                                       nrow = length(bF0$output[[length(bF0$output)]]$bp.Vt$breakpoints), ncol = 6, byrow = FALSE, dimnames = NULL)))
            
            
          } else {
            # if no trend break then
            
            as.data.frame(cbind(i, 1, NA, NA, matrix(rep(NA, 6), nrow=1, ncol=6, byrow=FALSE)))
            
            
          }
          
          
        }
        
  
} 
  


Sys.sleep(timeSleep)

print(paste0("object length: ", length(allBrksbF0)))

saveRDS(allBrksbF0, paste0(outf3, "output10", "/", "allBrksbF0.rds")) #Save object
  

end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

print("done")

