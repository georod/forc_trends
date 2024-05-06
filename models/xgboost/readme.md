# XGBoost models 

## 2024-04-01
## Peter R.

- For a few weeks, I looked at models with varying values of bfast's h parameter. For this I created several sets of CSV files:
 - forest_evi_breaks_negative_v3, forest_evi_breaks_positive_v3
 - forest_evi_breaks_negative_h2p_v3, forest_evi_breaks_positive_h2p_v3
 - forest_evi_breaks_negative_h10p_v3, forest_evi_breaks_positive_h10p_v3
- I have to finalize and pick one version of the models and create the corresponding plots for the paper. I have chosen to keep the h=1 year models
- I am only using XGB models for bfast standard algorithm, that is for EVI breaks (not trends)
- Originally, the intention was to assess the relationship between drivers and negative breaks. Given that most of the negative breaks I found could not be match to driver data (space-time match), I decided to use climate data to explore the breaks.
- While doing a seasonal exploration, I realized that the confidence intervals of the time of break was quite wide for a big portion of breaks. In some cases, these CIs were more that two years in width.
- In order to understand what was causing there very wide CIs, I explored with modifying the value of h and created some plots.
- In sum, my final version of XGB models will use high quality breaks only, use h=0.5 (as originally established, ~ 1 year od data points), and VIFplus variable set. These XGB models also seems to have a relatively good R-square-adjusted
- I have cleaned this github repo.  I have placed a copy here: C:\Users\Peter R\Documents\st_trends_for_c\github  This represents a copy before I re-organized my files on 2024-04-01.  
- I reorganized my files to keep my default GitHub repo streamlined
- Moving forward, I think I can run all my XGBoost models locally. No need to spin up DRAC.
- More info here: C:\Users\Peter R\Documents\st_trends_for_c\algonquin\version3\xgboost_readme_v1.md

