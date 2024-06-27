#=================================================================================
# Trend Analysis using MODIS data - Classify Trend shifts using EVI
# 
#=================================================================================
# 2024-06-24
# Peter R.

start.time <- Sys.time()
start.time

# Notes: 
# - This script is to classify trends using EVI
# - This code is for running parallel processing in a high performance computer. It takes a couple of hours to complete.


#=================================
# Load libraries
# ================================

# install.packages(c("strucchangeRcpp", "bfast"))
library(bfast)
library(foreach)
library(doParallel)

#install.packages("stlplus")
library(stlplus)


#=================================
# File paths and folders
# ================================

#setwd("~/projects/scripts/forc_trends") 

dataf <- "~/projects/data" # data folder


# Output folders
outf3 <- paste0(dataf, "/forc_trends_pj/EVI_250m/bfast01/")  # Note: h=0.5 run. Change back when done



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
# Time series
#======================================

timeSleep <- 3

rTs2 <- readRDS("~/projects/EVI_250m/bfast/period10/rTs2.rds") # Time series created in separate script


  
#=====================================================
# BFAST_01: Classify breaks into different classes
#=====================================================
  
   
brksbF0_01_Class <- foreach (i=1:ncol(rTs2), .inorder=TRUE) %dopar%  
     
      {
      
         print(i) # print cell #
      
        bF0_01  <- try(bfast01( 
                              rTs2[, i],
                              formula = NULL,
                              test = "OLS-MOSUM",
                              level = 0.01, 
                              aggregate = all,
                              trim = NULL,
                              bandwidth = 0.05, # this bandwith is equivalent to h in bfast. 
                              functional = "max",
                              order = 3,
                              lag = NULL,
                              slag = NULL,
                              na.action = na.omit, #na.pass is not a valid option here
                              reg = c("lm"), # c("lm", "rlm"),
                              stl = "none",
                              sbins = 1
          
                        ) )
        
        if(class(bF0_01) == "try-error") {
          
        list( as.data.frame(cbind(i, matrix(rep(NA,2) , nrow=1, ncol=2, byrow=FALSE)) ),
          
          as.data.frame(cbind(i, matrix(rep(NA,7), nrow=1, ncol=7, byrow=FALSE))))
          
        } else {
                 
          bF1_01_Class <- bfast01classify(
          bF0_01,
          alpha = 0.05, 
          pct_stable = 5,
          typology = c("standard")
		  
            )
          
         list(as.data.frame(cbind(i, bF0_01$breaks, bF0_01$data[bF0_01$breakpoints, 1])),
          as.data.frame(cbind(i, bF1_01_Class)))
            
        }
        
       
      
    }
  
  
Sys.sleep(timeSleep)
    
        

print(paste0("length: ", length(brksbF0_01_Class)))


saveRDS(brksbF0_01_Class, paste0(outf3, "period7", "/", "brksbF0_01_Class_sv2.rds")) # Save object

end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken 


print("done")
