#=============================================================
# Download MODIS data
#=============================================================

# 2023-04-01

# Peter R.

# Notes:
# 1) Use this code to download MODIS data
# 2) An web browser application will open at http://127.0.0.1:6703
# 3) Load sample json file to load key options used
# 4) The version number needs to be chosen manual (choose version: 061)
# 5) Load bounding box shapefile (algonquin_envelope_500m_buff_v1.shp)
# 6) Alternatively, a new json file can be created and saved with the application 
 
#install.packages('MODIStsp')

library(MODIStsp)
MODIStsp()

