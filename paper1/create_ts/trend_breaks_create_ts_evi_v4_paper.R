#=================================================================================
# Trend Analysis using BFAST & MODIS data - Create time series object 
#=================================================================================
# 2024-06-24
# Peter R.

# Notes:
#   - The aim of this script is to produce BFAST inputs using MODIS EVI (250 m, 16-days)
#   - Filtering of pixels was done based on suggestion by Samanta et al. 2010


start.time <- Sys.time()
start.time


#=================================
# Load libraries
# ================================
# install.packages(c("strucchangeRcpp", "bfast"))
library(terra)
library(sf)
#library(bfast)

library(foreach)

library(remotes)

#install.packages("stlplus")
library(stlplus)

library(MODIS)

library(raster) # Install raster after terra to avoid package issues
#library(sqldf)


#=================================
# File paths and folders
# ================================

setwd("~/projects/scripts/forc_trends") 

dataf <- "~/projects/data" # data folder


# MODIS files with original sinusoidal proj for period 2001-2023
fpath <- paste0(dataf,"/modis/algonquin_v2/modistsp/VI_16Days_250m_v61/EVI")

# Land cover
fpath2 <- "~/projects/def-mfortin/georod/data/ont_out/ont_CA_forest_VLCE2_2003.tif"

# QA pixels folder
fpath3_1 <- paste0(dataf,"/modis/algonquin_v2/modistsp/VI_16Days_250m_v61/QA_qual")
fpath3_2 <- paste0(dataf,"/modis/algonquin_v2/modistsp/VI_16Days_250m_v61/QA_usef")
fpath3_3 <- paste0(dataf,"/modis/algonquin_v2/modistsp/VI_16Days_250m_v61/QA_adj_cld")
fpath3_4 <- paste0(dataf,"/modis/algonquin_v2/modistsp/VI_16Days_250m_v61/QA_aer")
fpath3_5 <- paste0(dataf,"/modis/algonquin_v2/modistsp/VI_16Days_250m_v61/QA_mix_cld")
fpath3_6 <- paste0(dataf,"/modis/algonquin_v2/modistsp/VI_16Days_250m_v61/QA_shd")
fpath3_7 <- paste0(dataf,"/modis/algonquin_v2/modistsp/VI_16Days_250m_v61/QA_BRDF")
fpath3_8 <- paste0(dataf,"/modis/algonquin_v2/modistsp/VI_16Days_250m_v61/QA_land_wat")
fpath3_9 <- paste0(dataf,"/modis/algonquin_v2/modistsp/VI_16Days_250m_v61/QA_snow_ice")


# CSV to reclassify land cover values to keep only forest pixels
fpath4 <- "./misc/ca_forest_vlce2_lcover_type1_values.csv"

# Path to vector
shp1 <- "./misc/shp/algonquin_envelope_500m_buff_v1.shp"


# Output folders
outf3 <- paste0(dataf, "/forc_trends_pj/algonquin/output_h5p/EVI_250m/bfast/") 


# Start and end point of time series
start20yr <- c(1)
end20yr <- c(460)

periods1 <- cbind(c(start20yr), c(end20yr))
periods1Labs <- c("period10")
periods1 <- matrix(periods1[1,], nrow=1)



#======================================
# Time series loop
#======================================

#Sensibility test for determining the effect of different period lenght (modifiable time unit problem)
dir.create(paste0(outf3, periods1Labs[1])) 

# To leave out 2001-2002 EVI data (which have very sparse data for the study area) start after 2002, startI=26
startI <- 26
timeSleep <- 3


for (z in 1:nrow(periods1) ) {
  
  
  # Path to raster files
  # The year 2003 starts at position i=23 for EVI. This may be different for other indices
  rastfiles <- list.files(path=fpath, pattern = "*EVI*", full.names = TRUE)
  rastfiles <- rastfiles[startI:length(rastfiles)] 
  
  

  rastfilesNames <- list.files(path=fpath, pattern = "*EVI*", full.names = FALSE)
  rastfilesNames <- rastfilesNames[startI:length(rastfilesNames)] 

  
  rastfilesQA1 <- list.files(path=fpath3_1, pattern = "*_QA_qual*", full.names = TRUE)
  #length(rastfilesQA)
  rastfilesQA1 <- rastfilesQA1[startI:length(rastfilesQA1)] # 

  
  rastfilesQA2 <- list.files(path=fpath3_2, pattern = "*_QA_usef*", full.names = TRUE)
  #length(rastfilesQA)
  rastfilesQA2 <- rastfilesQA2[startI:length(rastfilesQA2)] # 

  
  rastfilesQA3 <- list.files(path=fpath3_3, pattern = "*_QA_adj*", full.names = TRUE)
  #length(rastfilesQA)
  rastfilesQA3 <- rastfilesQA3[startI:length(rastfilesQA3)] # 

  
  rastfilesQA4 <- list.files(path=fpath3_4, pattern = "*_QA_aer*", full.names = TRUE)
  #length(rastfilesQA)
  rastfilesQA4 <- rastfilesQA4[startI:length(rastfilesQA4)] # 

  
  rastfilesQA5 <- list.files(path=fpath3_5, pattern = "*_QA_mix_cld*", full.names = TRUE)
  #length(rastfilesQA)
  rastfilesQA5 <- rastfilesQA5[startI:length(rastfilesQA5)] # 

  
  rastfilesQA6 <- list.files(path=fpath3_6, pattern = "*_QA_shd*", full.names = TRUE)
  #length(rastfilesQA)
  rastfilesQA6 <- rastfilesQA6[startI:length(rastfilesQA6)] # 

  
  rastfilesQA7 <- list.files(path=fpath3_7, pattern = "*_QA_BRDF*", full.names = TRUE)
  #length(rastfilesQA)
  rastfilesQA7 <- rastfilesQA7[startI:length(rastfilesQA7)] # 

  
  rastfilesQA8 <- list.files(path=fpath3_8, pattern = "*_QA_land_wat*", full.names = TRUE)
  #length(rastfilesQA)
  rastfilesQA8 <- rastfilesQA8[startI:length(rastfilesQA8)] # 
 
  
  rastfilesQA9 <- list.files(path=fpath3_9, pattern = "*_QA_snow_ice*", full.names = TRUE)
  #length(rastfilesQA)
  rastfilesQA9 <- rastfilesQA9[startI:length(rastfilesQA9)] # 

  
  
  #--------------------------------------
  # Produce raster files objects
  #--------------------------------------
  
  rastfiles <- rastfiles[periods1[z,1]:periods1[z, 2]]
  rastfilesNames <- rastfilesNames[periods1[z,1]:periods1[z, 2]]
  
  
  rastfilesQA1 <- rastfilesQA1[periods1[z,1]:periods1[z, 2]]
  rastfilesQA2 <- rastfilesQA2[periods1[z,1]:periods1[z, 2]]
  rastfilesQA3 <- rastfilesQA3[periods1[z,1]:periods1[z, 2]]
  rastfilesQA4 <- rastfilesQA4[periods1[z,1]:periods1[z, 2]]
  rastfilesQA5 <- rastfilesQA5[periods1[z,1]:periods1[z, 2]]
  rastfilesQA6 <- rastfilesQA6[periods1[z,1]:periods1[z, 2]]
  
  rastfilesQA7 <- rastfilesQA7[periods1[z,1]:periods1[z, 2]]
  rastfilesQA8 <- rastfilesQA8[periods1[z,1]:periods1[z, 2]]
  rastfilesQA9 <- rastfilesQA9[periods1[z,1]:periods1[z, 2]]
  
  
  #--------------------------------------
  # Read data
  #--------------------------------------
  
  # Read in raster stack
  r1 <- terra::rast(rastfiles)
  
  Sys.sleep(timeSleep)
  
  
  #--------------------
  # Read in vector
  # transform shp1 to raster projection
  vpolyList1 <- list()
  
  for (y in 1:length(shp1)) {
    
    temp1 <- vect(st_read(shp1[y]))
    vpolyList1[[y]] <- project(temp1,r1)
    
  }
  
  #--------------------
  # Crop time series
  r2 <- crop(r1, vpolyList1[[1]])
  
  Sys.sleep(timeSleep)
  
  
  #--------------------
  # Read in QA pixels
  
  # Note: QA values range from 1 to 15.
  # See https://lpdaac.usgs.gov/documents/621/MOD13_User_Guide_V61.pdf for more info
  
  r1QAL <- list(terra::rast(rastfilesQA1),terra::rast(rastfilesQA2),terra::rast(rastfilesQA3), terra::rast(rastfilesQA4), terra::rast(rastfilesQA5), terra::rast(rastfilesQA6),
                 terra::rast(rastfilesQA7), terra::rast(rastfilesQA8), terra::rast(rastfilesQA9) )
  
  Sys.sleep(timeSleep)
  
  
  # Crop time series

  r2QACL <- list()
  
  r2QACL <- foreach (j = 1:length(r1QAL)) %do%
    crop(r1QAL[[j]],vpolyList1[[1]])
  

  Sys.sleep(timeSleep)
  
  # Reclassify QA pixel, keep only good ones. See Samantha2010
  ## from-to-becomes
  m1 <- c(2, 3, NA) # MODLAND_QA. Include only QA scores from 0 & 1, rest NA.
  m2 <- c(7, 15, NA) # VI usefulness. Include only QA scores from 1 to 9, rest NA. This is 2 levels more conservative than Samanta et al. 2010
  m3 <- c(1, NA) # Include only QA scores 0, rest NA. Adjacent cloud detected, flag values must be equal to 0.
  m4 <- c(0, NA, 3, NA) # Aerosol Quantity. Include only QA scores 1, 2, rest NA. Exclude Climatology, High Aerosols TO FIX
  m5 <- c(1,NA) # Include only QA scores 0, rest NA. Mixed clouds  flag values must be equal to 0.
  m6 <- c(1, NA) # Include only QA scores 0, rest NA. Possible shadow flag values must be equal to 0.
  
  m7 <- c(1, NA) # Include only QA scores 0, rest NA. Atmosphere BRDF correction flag values must be equal to 0.
  m8 <- c(0, 0, NA, 2, 7, NA) # Include only QA scores 0, rest NA. Land/Water masks flag values must be equal to 1, rest NA.
  m9 <- c(1, NA) # Include only QA scores 0, rest NA. Possible snow/ice flag values must be equal to 0.
  
  
  
  rclMList <- list(matrix(m1, ncol=3, byrow=TRUE), matrix(m2, ncol=3, byrow=TRUE), matrix(m3, ncol=2, byrow=TRUE),
                   matrix(m4, ncol=2, byrow=TRUE),  matrix(m5, ncol=2, byrow=TRUE), matrix(m6, ncol=2, byrow=TRUE),
                   matrix(m7, ncol=2, byrow=TRUE), 
                   matrix(m8, ncol=3, byrow=TRUE),
                   matrix(m9, ncol=2, byrow=TRUE)
                   )
  
  r2QACLC <- list()
  r2QACLC <- foreach (j = 1:length(r2QACL)) %do%
    classify(r2QACL[[j]], rclMList[[j]], include.lowest=TRUE, right=TRUE)
  

  # create SpatRasterDataset
  r2QAsds <- sds(r2QACLC)
  
  r2QAall <- app(r2QAsds, 'sum', na.rm=TRUE) 
  
  # replace integer values with a new single value
  r2QAallMask <- terra::ifel(r2QAall >=0, 1, r2QAall) 

  
  
  #-----------------------------------
  # Clean pixels, remove noisy pixels
  #-----------------------------------
  
  # this creates a list of rasters. The foreach function creates a list
  r1CL <- foreach (j= 1:nlyr(r2)) %do%
    mask(r2[[j]], r2QAallMask[[j]])
  
  Sys.sleep(timeSleep)
  
  # Read list of cleaned rasters
  r1C <- rast(r1CL)
  #rm(r1CL)
  
  
  # ---------------------
  # Read in Land cover
  # ---------------------
  # Read land cover 2003
  rLc <- terra::rast(fpath2)
  
  rLc <- crop(rLc, project(vpolyList1[[1]],crs(rLc) )) # project vector rather than raster
  
  # Reclassify Land cover to keep only forest
  LcType1 <- read.csv(fpath4)
  
  rclM2 <- as.matrix(LcType1[,c(1,3)]) # Note: I am also including wetland-treed
  rLc2 <- classify(rLc,rclM2)
  
  
  
  #-------------------------------
  # Mask out non-forest pixels
  #-------------------------------
  
  rLc3 <- project(rLc2, crs(r1C),  method='near', threads=TRUE)
  
  rLc3 <- resample(rLc3 , r1C, method='near', threads=TRUE) 
  
  terra::ext(rLc3) <- terra::ext(r1C)
  r3 <- mask(r1C, rLc3, maskvalue=0)
  r4 <- mask(r3, rLc3, maskvalue=NA)
  

  r1Br <- raster::brick(r4) # I need to convert SpatRast object to raster object as Bfast needs the latter
  

  #-----------------------------------
  # Dates & Time series
  #-----------------------------------
  #Dates for period 2003-01-01 to 2022-12-31
  #this shows the human readable dates. E.g. 2003_001 as "2003-01-01"
  rDates <- MODIS::extractDate(paste0((sapply(strsplit(rastfilesNames, '_'), "[", 3)), (sapply(strsplit(rastfilesNames, '_'), "[", 4))),1, 7, asDate = TRUE)$inputLayerDates
 
  
  # Create dataframe and transpose
  t1 <- t(as.data.frame(r1Br))
  
  # time-series
  rTs2 <- bfastts(t1, dates= rDates, type = c("16-day"))
  
  saveRDS(rTs2, paste0(outf3, periods1Labs[1], "/", "rTs2.rds")) 
  
  
} # end of very long loop


end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

print(paste0("rows: ", nrow(rTs2), "cols: ", ncol(rTs2)))
print("done")
