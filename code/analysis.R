#### This code replicates the analysis in Vesco et al.
#### Produced by P. Vesco, last updated July 3, 2025

#The script sets up the model formulas, fits the models, calculates the LOO-CV statistics, creates performance tables and runs the counterfactual analysis presented in the manuscript and SM

rm(list = ls(all = TRUE))

library(randomForest)
library(posterior)
library(ggridges)
library(patchwork)
library(future)
library(Metrics)
library(LongituRF)
library(splitstackshape)
library(gpboost)
library(groupdata2)
library(rsample)
library(progress)
library(caret)
library(utils)
library(itsadug)
library(splitTools)
library(additive)
library(cvTools)
library(groupdata2)
library(workflows)
library(broom.mixed)
library(xtable)
library(glmmTMB)
library(Amelia)
library(lme4)
library(validate)
library(tidyverse)
library(HH)
library(skimr)
library(recipes)
library(bayestestR)
library(knitr)
library(patchwork)
library(tidybayes)
library(mice)
library(gt)
library(scales)
library(sjPlot)
library(ggmap)
library(RColorBrewer)
library(robustlmm)
library(countrycode)
library(bayesplot)
#Sys.setenv(GITHUB_PAT = "mygithubtoken")
#remotes::install_github("paul-buerkner/brms")
library(brms)
library(bayestestR)
library(loo)
library(see)
library(cmdstanr)
library(remotes)
library(rlang)
library(ggbeeswarm)
library(extrafont)
library(HDInterval)
library(texreg)
library(dbarts)
library(cowplot)
library(magick)
library(webshot2)
library(fixest)              
library(modelsummary)


font_import()
loadfonts(device="win")

SEEDNUM = 352024
TRAIN_MODELS <- FALSE # To fit the models set this to TRUE
GDP <-  FALSE ## As it is set up currently, the script runs the analysis for the MAIN SPECIFICATION presented in the manuscript

###To run the tests presented in the Supplementary Material, you need to set the appropriate test to TRUE 

AFFECTED <- FALSE
AGG <-   FALSE
CUTOFF <- FALSE
DEAD <- FALSE
ECONTEST <- FALSE
FE <- TRUE
INTER <- FALSE 
NOBNG <- FALSE
NOCHI <- FALSE
NOGDP <- FALSE
NOIND <- FALSE
NOLOC <-   FALSE
NOMMR <-  FALSE
NORE <- FALSE
NOYEAR <- FALSE
OUTLIER <- FALSE
SPLIT <- FALSE

#### DEFITRUE#### DEFINE RESULTS FOLDER ####
#setwd("political_development_flood")

setwd("/Users/paola.vesco/ViEWS Dropbox/Paola Vesco/paola/projects/polimpact/flood_paper/")
RESULT_FOLDER <- "results/panel"

if(GDP){ RESULT_FOLDER <- paste(RESULT_FOLDER, paste0("GDP"), sep = "/")} 
if(FE){ RESULT_FOLDER <- paste(RESULT_FOLDER, paste0("FE"), sep = "/")} 
if(AFFECTED){ RESULT_FOLDER <- paste(RESULT_FOLDER, paste0("affected"), sep = "/")}
if(DEAD){ RESULT_FOLDER <- paste(RESULT_FOLDER, paste0("deaths"), sep = "/")}
if(CUTOFF){ RESULT_FOLDER <- paste(RESULT_FOLDER, paste0("cutoff"), sep = "/")}
if(OUTLIER){ RESULT_FOLDER <- paste(RESULT_FOLDER, paste0("outlier"), sep = "/")}
if(NOGDP) { RESULT_FOLDER <- paste(RESULT_FOLDER, paste0("nogdp"), sep = "/")}
if(AGG) { RESULT_FOLDER <- paste(RESULT_FOLDER, paste0("aggregate"), sep = "/")}
if(NOLOC){ RESULT_FOLDER <- paste(RESULT_FOLDER, paste0("nolocal"), sep = "/")}
if(NOMMR){ RESULT_FOLDER <- paste(RESULT_FOLDER, paste0("nomyanmar"), sep = "/")}
if(NOIND){ RESULT_FOLDER <- paste(RESULT_FOLDER, paste0("noindia"), sep = "/")}
if(NOCHI){ RESULT_FOLDER <- paste(RESULT_FOLDER, paste0("nochina"), sep = "/")}
if(NOBNG){ RESULT_FOLDER <- paste(RESULT_FOLDER, paste0("nobangladesh"), sep = "/")}
if(ECONTEST){ RESULT_FOLDER <- paste(RESULT_FOLDER, paste0("econtest"), sep = "/")}
if(INTER){ RESULT_FOLDER <- paste(RESULT_FOLDER, paste0("interactions"), sep = "/")}
if(NORE){ RESULT_FOLDER <- paste(RESULT_FOLDER, paste0("norandomeffects"), sep = "/")}
if(NOYEAR){ RESULT_FOLDER <- paste(RESULT_FOLDER, paste0("noyeartrends"), sep = "/")} 
if(SPLIT){ RESULT_FOLDER <- paste(RESULT_FOLDER, paste0("random_split"), sep = "/")} 

results_path <- paste0(getwd(),"/",RESULT_FOLDER)
models_path <- paste0(results_path,"/bayes_models/")
plots_path <- paste0(models_path,"plots/")
tables_path <- paste0(models_path,"tables/")

results_path
models_path
plots_path
tables_path

dir.create(results_path)
dir.create(models_path)
dir.create(plots_path)
dir.create(tables_path)



####LOAD THE DATA####

df <- readRDS('data/data_final.rds')

##Filter by affected population higher than 0 across all specifications
df = subset(df, df$population_affected > 0)

## For random effects or fixed effects
df$gwcode <- factor(df$gwcode)
df$continent <- factor(df$continent)

#Invert the scale of exclusion so that it is interpreted as inclusion

df$v2xpe_exlsocgr <- 1 - df$v2xpe_exlsocgr


if(FE){
  df$year <- factor(df$year)
}

if(DEAD){
  df <- subset(df, df$dead_w > 0)
}

if(AFFECTED){
  df = subset(df, df$population_affected > 1000)
}


if(OUTLIER){
  df <-  subset(df, dead_w < 79828)
}

if(NOMMR){
  df <-  subset(df, df$gwcode != 775)
}

if(NOIND){
  df <-  subset(df, df$gwcode != 750)
}

if(NOCHI){
  df <-  subset(df, df$gwcode != 710)
}

if(NOBNG){
  df <-  subset(df, df$gwcode != 771)
}


#log non-index variables
log_vars <- c("duration", "nevents_sum10", "population_affected", "wdi_gdppc", "hdi_l1", "brd_12mb", "decay_brds_c")

df <- df %>% 
  mutate(across(all_of(log_vars), .fns = function(x) log(x+1)))
df$dead_w <- as.integer(df$dead_w)

df <- df %>%
  mutate(tropical_flood_dummy = as.integer(tropical_flood > 0))


##define formulas for models

if(NOGDP){
  v_baseline <- c("dfo_severity", "duration", "nevents_sum10", "population_affected", "rugged", "tropical_flood")
}else if (NOLOC){
  v_baseline <- c("dfo_severity", "duration", "nevents_sum10", "population_affected", "rugged", "tropical_flood", "wdi_gdppc")
} else if(FE) {
  v_baseline <- c("dfo_severity", "duration", "nevents_sum10", "population_affected", "hdi_l1", "wdi_gdppc", "tropical_flood", "gwcode")
}else {
  v_baseline <- c("dfo_severity", "duration", "nevents_sum10", "population_affected", "rugged", "hdi_l1", "wdi_gdppc", "tropical_flood")
}


preds <- c(v_baseline, "year", "brd_12mb", "decay_brds_c", "v2xpe_exlsocgr", 
           "e_wbgi_vae", "e_wbgi_gee", "v2x_rule", "v2x_polyarchy", "hdi_l1", "wdi_gdppc", "continent")

all <- c("dead_w", preds)

if(CUTOFF){
  train <- df %>% dplyr::filter(year >= 2000 & year <= 2010) %>% dplyr::select(all)
  test <- df %>% dplyr::filter(year >= 2011) %>% dplyr::select(all)
}else if (SPLIT) {
  smp_size <- floor(0.8 * nrow(df))
  train_ind <- sample(seq_len(nrow(df)), size = smp_size)
  train <- df[train_ind, ] %>% dplyr::select(all)
  test <- df[-train_ind, ] %>% dplyr::select(all)
}else{
  train <- df %>% dplyr::filter(year >= 2000 & year <= 2014) %>% dplyr::select(all)
  test <- df %>% dplyr::filter(year >= 2015) %>% dplyr::select(all)
  
}

############TRAIN THE MODELS#################

#train the models

if(TRAIN_MODELS){
  if (FE){
    formulas <- list(
      baseline = (dead_w ~ dfo_severity + duration + nevents_sum10 + 
                    population_affected  + hdi_l1 + wdi_gdppc + 
                    I(tropical_flood > 0) | gwcode + year),
      conflict_country = (dead_w ~ dfo_severity + duration + nevents_sum10 + 
                            population_affected   + hdi_l1 + wdi_gdppc +
                            I(tropical_flood > 0) + decay_brds_c | gwcode + year),
      conflict_sub = (dead_w ~ dfo_severity + duration + nevents_sum10 +
                        population_affected + hdi_l1  + wdi_gdppc +
                        I(tropical_flood > 0) + brd_12mb | gwcode + year),
      accountability = (dead_w ~ dfo_severity + duration + nevents_sum10 +
                          population_affected  + hdi_l1 + wdi_gdppc +
                          I(tropical_flood > 0) + e_wbgi_vae | gwcode + year),
      goveff = (dead_w ~ dfo_severity + duration + nevents_sum10 +
                  population_affected + hdi_l1  + wdi_gdppc +
                  I(tropical_flood > 0) + e_wbgi_gee | gwcode + year),
      inclusion = (dead_w ~ dfo_severity + duration + nevents_sum10 +
                     population_affected  + hdi_l1 + wdi_gdppc +
                     I(tropical_flood > 0) + v2xpe_exlsocgr | gwcode + year),
      ruleoflaw = (dead_w ~ dfo_severity + duration + nevents_sum10 +
                     population_affected + hdi_l1 + wdi_gdppc +
                     I(tropical_flood > 0) + v2x_rule | gwcode + year), 
      poly = (dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + v2x_polyarchy  | gwcode + year))
  }else if(INTER) {
    plan(multicore, workers = 4, future.seed = TRUE)
    # Set the number of cores for parallel processing
    options(mc.cores = parallel::detectCores())
    formulas <- list(
      baseline = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + year +  (1 + nevents_sum10 | continent)),
      conflict_country = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + decay_brds_c + wdi_gdppc*decay_brds_c + year + (1 + nevents_sum10 | continent)),
      conflict_sub = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + brd_12mb  + wdi_gdppc*brd_12mb + year  + (1 + nevents_sum10 | continent)),
      accountability = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + e_wbgi_vae + wdi_gdppc*e_wbgi_vae + year + (1 + nevents_sum10 | continent)),
      goveff = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + e_wbgi_gee + wdi_gdppc*e_wbgi_gee + year + (1 + nevents_sum10 | continent)),
      inclusion = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  +v2xpe_exlsocgr + wdi_gdppc*v2xpe_exlsocgr + year + (1 + nevents_sum10 | continent)),
      ruleoflaw = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + v2x_rule + wdi_gdppc*v2x_rule + year + (1 + nevents_sum10 | continent)))
  }else if (AGG) {
    # Define the model formulas based on your variables
    plan(multicore, workers = 4, future.seed = TRUE)
    # Set the number of cores for parallel processing
    options(mc.cores = parallel::detectCores())
    formulas <- list(
      baseline = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + year + (1 + nevents_sum10 | continent)),
      conflict = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + brd_12mb + decay_brds_c + year + (1 + nevents_sum10 | continent)),
      democracy = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + e_wbgi_vae + v2xpe_exlsocgr + year + (1 + nevents_sum10 | continent)),
      instqual = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + e_wbgi_gee  + v2x_rule + year + (1 + nevents_sum10 | continent)),
      poly = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + v2x_polyarchy + year + (1 + nevents_sum10 | continent)),
      all = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + e_wbgi_vae + v2xpe_exlsocgr  + e_wbgi_gee + v2x_rule + decay_brds_c + brd_12mb + year + (1 + nevents_sum10 | continent))
    )
  } else if(NOLOC) {
    # Define the model formulas based on your variables
    plan(multicore, workers = 4, future.seed = TRUE)
    # Set the number of cores for parallel processing
    options(mc.cores = parallel::detectCores())
    formulas <- list(
      baseline = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged  + wdi_gdppc + I(tropical_flood >0)  + year +  (1 + nevents_sum10 | continent)),
      conflict_country = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged  + wdi_gdppc + I(tropical_flood >0)  + decay_brds_c + year + (1 + nevents_sum10 | continent)),
      accountability = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged  + wdi_gdppc  + I(tropical_flood >0)  + e_wbgi_vae + year + (1 + nevents_sum10 | continent)),
      goveff = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged  + wdi_gdppc + I(tropical_flood >0)  + e_wbgi_gee + year + (1 + nevents_sum10 | continent)),
      inclusion = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged  + wdi_gdppc  + I(tropical_flood >0)  +v2xpe_exlsocgr + year + (1 + nevents_sum10 | continent)),
      ruleoflaw = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged  + wdi_gdppc + I(tropical_flood >0)  + v2x_rule + year + (1 + nevents_sum10 | continent)))
  } else if (ECONTEST){
    plan(multicore, workers = 4, future.seed = TRUE)
    # Set the number of cores for parallel processing
    options(mc.cores = parallel::detectCores())
    formulas <- list(
      gdpbase = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + wdi_gdppc + hdi_l1  + I(tropical_flood >0)  + year + (1 + nevents_sum10 | continent)),
      gdpconf = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + wdi_gdppc + hdi_l1  + I(tropical_flood >0) +  year +  brd_12mb + decay_brds_c + (1 + nevents_sum10 | continent))
    )
  }else if (NORE){
    # Define the model formulas based on your variables
    plan(multicore, workers = 4, future.seed = TRUE)
    # Set the number of cores for parallel processing
    options(mc.cores = parallel::detectCores())
    formulas <- list(
      baseline = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + year ),
      conflict_country = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + hdi_l1 + wdi_gdppc + rugged + I(tropical_flood >0)   + decay_brds_c + year),
      conflict_sub = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + hdi_l1 + wdi_gdppc + rugged + I(tropical_flood >0)  + brd_12mb  + year),
      accountability = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected   + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + e_wbgi_vae + year),
      goveff = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + e_wbgi_gee + year),
      inclusion = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + v2xpe_exlsocgr),
      ruleoflaw = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + v2x_rule + year)
    )
  }else if (NOYEAR){
    # Define the model formulas based on your variables
    plan(multicore, workers = 4, future.seed = TRUE)
    # Set the number of cores for parallel processing
    options(mc.cores = parallel::detectCores())
    formulas <- list(
      baseline = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)   + (1 + nevents_sum10 | continent)),
      conflict_country = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged  + hdi_l1 + wdi_gdppc+ I(tropical_flood >0)   + decay_brds_c  + (1 + nevents_sum10 | continent)),
      conflict_sub = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + hdi_l1 + wdi_gdppc+ I(tropical_flood >0)  + brd_12mb   + (1 + nevents_sum10 | continent)),
      accountability = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected  + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + e_wbgi_vae  + (1 + nevents_sum10 | continent)),
      goveff = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + e_wbgi_gee  + (1 + nevents_sum10 | continent)),
      inclusion = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + v2xpe_exlsocgr  + (1 + nevents_sum10 | continent)),
      ruleoflaw = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + v2x_rule  + (1 + nevents_sum10 | continent))
    )
  } else if (NOGDP) {
    # Define the model formulas based on your variables
    plan(multicore, workers = 4, future.seed = TRUE)
    # Set the number of cores for parallel processing
    options(mc.cores = parallel::detectCores())
    formulas <- list(
      baseline = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged  + I(tropical_flood >0)  + year + (1 + nevents_sum10 | continent)),
      conflict_country = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + I(tropical_flood >0)   + decay_brds_c + year + (1 + nevents_sum10 | continent)),
      conflict_sub = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + I(tropical_flood >0)  + brd_12mb  + year + (1 + nevents_sum10 | continent)),
      accountability = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected  + rugged  + I(tropical_flood >0)  + e_wbgi_vae + year + (1 + nevents_sum10 | continent)),
      goveff = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged  + I(tropical_flood >0)  + e_wbgi_gee + year + (1 + nevents_sum10 | continent)),
      inclusion = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged  + I(tropical_flood >0)  + v2xpe_exlsocgr + year + (1 + nevents_sum10 | continent)),
      ruleoflaw = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + I(tropical_flood >0)  + v2x_rule + year + (1 + nevents_sum10 | continent)),
      poly = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + I(tropical_flood >0)  + v2x_polyarchy + year + (1 + nevents_sum10 | continent))
    ) } else {
      # Define the model formulas based on your variables
      plan(multicore, workers = 4, future.seed = TRUE)
      # Set the number of cores for parallel processing
      options(mc.cores = parallel::detectCores())
      formulas <- list(
        baseline = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged  + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + year + (1 + nevents_sum10 | continent)),
        conflict_country = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)   + decay_brds_c + year + (1 + nevents_sum10 | continent)),
        conflict_sub = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + brd_12mb  + year + (1 + nevents_sum10 | continent)),
        accountability = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected  + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + e_wbgi_vae + year + (1 + nevents_sum10 | continent)),
        goveff = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged  + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + e_wbgi_gee + year + (1 + nevents_sum10 | continent)),
        inclusion = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + v2xpe_exlsocgr + year + (1 + nevents_sum10 | continent)),
        ruleoflaw = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + v2x_rule + year + (1 + nevents_sum10 | continent)),
        poly = bf(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged + hdi_l1 + wdi_gdppc + I(tropical_flood >0)  + v2x_polyarchy + year + (1 + nevents_sum10 | continent)))
    }
  
  # Fit the models using the 'brm' function from the 'brms' package
  if (FE) {
    models <- map(formulas, ~fepois(.x, data = df))
    walk2(models, names(models), ~saveRDS(.x, paste0(models_path, .y, "_model.rds")))
  }else if (NOBNG) { #Due to convergence issues for models omitting Bangladesh (smaller sample) increase iterations
    models <- map(formulas, ~brm(
      formula = .x,
      data = train,
      family = negbinomial(),
      chains = 12,             
      iter = 15000,            
      warmup = 7500,          
      control = list(
        adapt_delta = 0.999999,  
        max_treedepth = 25,      
        step_size = 0.0001        
      ),
      backend = "cmdstanr",
      threads = threading(3),   
      save_pars = save_pars(all = TRUE),
      seed = SEEDNUM,
      init = "0",              
      refresh = 200           
    ))
  }else {
    models <- map(formulas, ~brm(
      formula = .x,
      data = train,
      family = negbinomial(),
      chains = 8,
      iter = 8000,           
      warmup = 4000,         
      control = list(
        adapt_delta = 0.9999,  
        max_treedepth = 15     
      ),
      backend = "cmdstanr",
      threads = threading(4),
      init = "0",   
      save_pars = save_pars(all = TRUE),
      seed = SEEDNUM
    ))
  }
  
  # Save the models as RDS files
  walk2(models, names(models), ~saveRDS(.x, paste0(models_path, .y, "_model.rds")))
}

# Re train fit accountability for NOBNG since it did not converge!


#### Load the  models from saved RDS files
if(AGG){
  fit_baseline <- readRDS(paste0(models_path, "baseline_model.rds"))
  fit_conflict <- readRDS(paste0(models_path, "conflict_model.rds"))
  fit_democracy <- readRDS(paste0(models_path, "democracy_model.rds"))
  fit_instqual <- readRDS(paste0(models_path, "instqual_model.rds"))
  fit_poly <- readRDS(paste0(models_path, "poly_model.rds"))
  fit_all <- readRDS(paste0(models_path, "all_model.rds"))
} else if (NOLOC){
  fit_baseline <- readRDS(paste0(models_path, "baseline_model.rds"))
  fit_conflict_country <- readRDS(paste0(models_path, "conflict_country_model.rds"))
  fit_accountability <- readRDS(paste0(models_path, "accountability_model.rds"))
  fit_goveff <- readRDS(paste0(models_path, "goveff_model.rds"))
  fit_inclusion <- readRDS(paste0(models_path, "inclusion_model.rds"))
  fit_ruleoflaw <- readRDS(paste0(models_path, "ruleoflaw_model.rds"))
} else if(ECONTEST){
  fit_gdpbase <- readRDS(paste0(models_path, "gdpbase_model.rds"))
  fit_gdpconf <- readRDS(paste0(models_path, "gdpconf_model.rds"))
} else if(GDP){
  fit_baseline <- readRDS(paste0(models_path, "baseline_model.rds"))
  fit_conflict_country <- readRDS(paste0(models_path, "conflict_country_model.rds"))
  fit_conflict_sub <- readRDS(paste0(models_path, "conflict_sub_model.rds"))
  fit_accountability <- readRDS(paste0(models_path, "accountability_model.rds"))
  fit_goveff <- readRDS(paste0(models_path, "goveff_model.rds"))
  fit_inclusion <- readRDS(paste0(models_path, "inclusion_model.rds"))
  fit_ruleoflaw <- readRDS(paste0(models_path, "ruleoflaw_model.rds"))
  fit_poly <- readRDS(paste0(models_path, "poly_model.rds"))
}else{
  fit_baseline <- readRDS(paste0(models_path, "baseline_model.rds"))
  fit_conflict_country <- readRDS(paste0(models_path, "conflict_country_model.rds"))
  fit_conflict_sub <- readRDS(paste0(models_path, "conflict_sub_model.rds"))
  fit_accountability <- readRDS(paste0(models_path, "accountability_model.rds"))
  fit_goveff <- readRDS(paste0(models_path, "goveff_model.rds"))
  fit_inclusion <- readRDS(paste0(models_path, "inclusion_model.rds"))
  fit_ruleoflaw <- readRDS(paste0(models_path, "ruleoflaw_model.rds"))
}


#Check convergence 

#Figure S6.

library(rstan)

if(GDP){
  summary(fit_baseline) #Print Rhat, ESS etc
  loo(fit_baseline) #Print K-statistics 
  
  posterior_samples <- as_draws_df(fit_baseline)
  # Trace plot for parameters
  trace_plot <- mcmc_trace(posterior_samples, pars = c("b_Intercept", "b_dfo_severity", "b_duration", "b_nevents_sum10",  "b_rugged", "b_Itropical_flood>0TRUE",  "b_wdi_gdppc", "b_hdi_l1", "b_year"))
  
  # Assuming the parameter names in your model are as follows
  param_names <- c("Intercept", "Flood severity", "Flood duration", 
                   "Past flood events", "Rugged terrain", "Tropical flood", "National GDP per capita", "Local HDI", "Year")
  
  # Customizing the labels to match your parameters
  custom_labels <- c(
    b_Intercept = "Intercept",
    b_dfo_severity = "Flood severity",
    b_duration = "Flood duration",
    b_nevents_sum10 = "Past flood events",
    b_rugged = "Rugged terrain",
    `b_Itropical_flood>0TRUE` = "Tropical flood",
    `b_wdi_gdppc` ="National GDP per capita",
    `b_hdi_l1` ="Local HDI",
    b_year = "Year"
  )
  
  generate_parameter_plots <- function(parameter_name, custom_label) {
    
    hist_plot <- mcmc_hist(
      posterior_samples,
      pars = parameter_name
    ) +
      theme_minimal() +
      theme(axis.line = element_line(), 
            axis.ticks = element_line(),
            panel.grid.minor = element_blank(),
            plot.background = element_rect(fill = "white", color = "white"),
            panel.background = element_rect(fill = "white", color = "white"),
            text = element_text(family = "Arial", size = 12), 
            plot.title = element_text(size = 12, hjust = 0.5),
            axis.title = element_text(size = 12),
            axis.text = element_text(size = 12),
      ) +
      labs(
        y = "Frequency",
        x = "Value",
        title = custom_label
      )
    
    
    trace_plot <- mcmc_trace(
      posterior_samples,
      pars = parameter_name,
      facet_args = list(labeller = as_labeller(custom_labels))
    ) +
      theme_minimal() +
      theme(axis.line = element_line(), 
            axis.ticks = element_line(),
            panel.grid.minor = element_blank(),
            plot.background = element_rect(fill = "white", color = "white"),
            panel.background = element_rect(fill = "white", color = "white"),
            text = element_text(family = "Arial", size = 12), 
            plot.title = element_text(size = 12, hjust = 0.5),
            axis.title = element_text(size = 12),
            axis.text = element_text(size = 12),
            strip.text = element_text(size = 12), 
            legend.position = "none"
      ) +
      labs(
        y = "Value",
        x = "Iteration",
        title = custom_label
      )
    
    
    combined_plot <- hist_plot + trace_plot
    
    return(combined_plot)
  }
  
  combined_plots <- list()
  for (param in names(custom_labels)) {
    combined_plots[[param]] <- generate_parameter_plots(param, custom_labels[[param]])
  }
  
  all_plots_combined <- wrap_plots(combined_plots, ncol = 1)
  
  print(all_plots_combined)
  
  ggsave(paste0(plots_path, "convergence_fitbaseline.png"), plot = all_plots_combined, 
         width = 8, height = 15,dpi = 350)
  
}



###ANALYSIS AND VISUALISATION####

###relative performance 

#combine all models
# Create a list of all models
if(AGG){
  all_models <- list(
    fit_baseline,
    fit_conflict,
    fit_democracy, 
    fit_instqual,
    fit_poly, 
    fit_all)
} else if(ECONTEST) {
  all_models <- list(
    fit_gdpbase,
    fit_gdpconf
  )
}else if(NOLOC){
  all_models <- list(
    fit_baseline,
    fit_conflict_country,
    fit_accountability,
    fit_goveff,
    fit_inclusion,
    fit_ruleoflaw)
} else if (GDP){
  all_models <- list(
    fit_baseline,
    fit_conflict_country,
    fit_conflict_sub,
    fit_accountability,
    fit_goveff,
    fit_inclusion,
    fit_ruleoflaw, 
    fit_poly)
}else {
  all_models <- list(
    fit_baseline,
    fit_conflict_country,
    fit_conflict_sub,
    fit_accountability,
    fit_goveff,
    fit_inclusion,
    fit_ruleoflaw)
}

# Name the list for convenience
if(AGG){names(all_models) <- c(
  "Baseline",
  "Conflict",
  "Democracy",
  "Institutional quality",
  "Electoral democracy", 
  "All"
)} else if(ECONTEST){
  names(all_models) <- list(
    "Baseline", 
    "Baseline w conflict")
}else if(NOLOC){
  names(all_models) <- c(
    "Baseline",
    "Conflict history",
    "Accountability",
    "Gov. effectiveness",
    "Inclusion",
    "Rule of law"
  )
}else if (GDP){
  names(all_models) <- c(
    "Baseline",
    "Conflict history",
    "Local conflict",
    "Accountability",
    "Gov. effectiveness",
    "Inclusion",
    "Rule of law", 
    "Electoral democracy"
  )
}else {
  names(all_models) <- c(
    "Baseline",
    "Conflict history",
    "Local conflict",
    "Accountability",
    "Gov. effectiveness",
    "Inclusion",
    "Rule of law"
  )
}

# Now, create a tibble that contains all  models
fit_df <- tibble(
  fits = all_models,
  mname = names(all_models))

# Create a list that directly references the fits in fit_df for easier access
models <- fit_df$fits
names(models) <- fit_df$mname


## Regression tables
custom_labels <- c(e_wbgi_vae = "Accountability",
                   v2xpe_exlsocgr = "Inclusion",
                   e_wbgi_gee = "Gov. effectiveness",
                   v2x_rule = "Rule of law",
                   decay_brds_c = "Conflict history",
                   brd_12mb = "Local conflict",
                   v2x_polyarchy = "Electoral democracy",
                   population_affected = "Exposed population",
                   dfo_severity = "Flood severity",
                   duration = "Flood duration",
                   nevents_sum10 = "Past flood events",
                   rugged = "Rugged terrain",
                   tropical_flood = "Tropical flood",
                   wdi_gdppc = "National GDP per capita",
                   hdi_l1 = "Local HDI", 
                   "Itropical_flood>0TRUE" = "Tropical flood", 
                   year = "Year")

ordered_labels_fe <- c(population_affected = "Exposed population",
                    dfo_severity = "Flood severity",
                    duration = "Flood duration",
                    nevents_sum10 = "Past flood events",
                    rugged = "Rugged terrain",
                    "Itropical_flood>0TRUE" = "Tropical flood",
                    wdi_gdppc = "National GDP per capita",
                    hdi_l1 = "Local HDI",
                    e_wbgi_vae = "Accountability",
                    v2xpe_exlsocgr = "Inclusion",
                    e_wbgi_gee = "Gov. effectiveness",
                    v2x_rule = "Rule of law",
                    decay_brds_c = "Conflict history",
                    brd_12mb = "Local conflict",
                    v2x_polyarchy = "Electoral democracy", 
                    "FE: gwcode" = "Country fixed effects",
                    "FE: factor(year)" = "Year fixed effects"
)

ordered_labels <- c(population_affected = "Exposed population",
                    dfo_severity = "Flood severity",
                    duration = "Flood duration",
                    nevents_sum10 = "Past flood events",
                    rugged = "Rugged terrain",
                    "Itropical_flood>0TRUE" = "Tropical flood",
                    wdi_gdppc = "National GDP per capita",
                    hdi_l1 = "Local HDI",
                    e_wbgi_vae = "Accountability",
                    v2xpe_exlsocgr = "Inclusion",
                    e_wbgi_gee = "Gov. effectiveness",
                    v2x_rule = "Rule of law",
                    decay_brds_c = "Conflict history",
                    brd_12mb = "Local conflict")

custom_fe_rows <- tibble::tibble(
  term = c("Country fixed effects", "Year fixed effects"),
  "Accountability"    = c("X", "X"),
  "Inclusion"         = c("X", "X"),
  "Gov. effectiveness"= c("X", "X"),
  "Rule of law"       = c("X", "X"),
  "Conflict history"  = c("X", "X"),
  "Local conflict"    = c("X", "X"),
  "Baseline"          = c("X", "X")
)
# Named colors
colors <- c(
  'e_wbgi_vae' = "#8eade8",
  'v2xpe_exlsocgr' = "#3c61a3",
  'e_wbgi_gee' = "#f2aed8",
  'v2x_rule' = "#c556d1",
  'decay_brds_c' = "#f5b342",
  'brd_12mb' = "#f0843c",
  "v2x_polyarchy" = "#5df0e1",
  'population_affected' = "#CCCCCC",
  'dfo_severity' = "#CCCCCC",
  'duration' = "#CCCCCC",
  'nevents_sum10' = "#CCCCCC",
  'rugged' = "#CCCCCC",  # Not in plot, but for completeness
  'tropical_flood' = "#CCCCCC",  # Not in plot, but for completeness
  "wdi_gdppc" = "#CCCCCC",
  "hdi_l1"= "#CCCCCC"
)

library(rstan)
if(GDP){
    # 1) extract as before, but make `model` a factor
    coefs_df <- lapply(names(models), function(mn){
      broom.mixed::tidy(
        models[[mn]],
        effects  = "fixed",
        conf.int = TRUE
      ) %>%
        mutate(model = mn)  # character for now
    }) %>%
      bind_rows() %>%
      # turn `model` into a factor in the desired order
      mutate(model = factor(model, levels = names(models))) %>%
      # exponentiate
      mutate(
        IRR     = exp(estimate), # Convert log coefficients to IRRs
        CI_low  = exp(conf.low), 
        CI_high = exp(conf.high) 
      ) 
    
    # 2) build the two-line cell
    table_df <- coefs_df %>%
      filter(term %in% names(ordered_labels)) %>%
      mutate(term = factor(term, levels = names(ordered_labels))) %>%
      arrange(term, model) %>%
      mutate(
        cell = sprintf(
          "%.3f \\\\n[%.3f, %.3f]",
          IRR, CI_low, CI_high
        )
      ) %>%
      dplyr::select(model, term, cell) %>%
      pivot_wider(
        names_from  = model,
        values_from = cell
      ) %>%
      mutate(
        term_label = ordered_labels[as.character(term)]
      ) %>%
      # now select in exactly the `models` order:
      dplyr::select(term_label, all_of(names(models)))
    
    # 3) print
    library(kableExtra)
    kable(
      table_df,
      format    = "latex",
      booktabs  = TRUE,
      align     = c("l", rep("c", length(models))),
      caption   = "\\textbf{Negative binomial models with random effects (in-sample)}",
      label     = "tab:gdp",
      escape    = FALSE,
      linesep   = ""
    ) %>%
      kable_styling(
        latex_options = "hold_position",
        font_size     = 8
      ) %>%
      add_header_above(c(" " = 1, "Models" = length(models)))
  }



if (FE) {
  models <- list(
    "Baseline"           = fit_baseline,
    "Accountability"     = fit_accountability,
    "Inclusion"          = fit_inclusion,
    "Gov. effectiveness" = fit_goveff,
    "Rule of law"        = fit_ruleoflaw,
    "Conflict history"   = fit_conflict_country,
    "Local conflict"     = fit_conflict_sub
  )
  
  modelsummary::msummary(
    models,
    coef_map    = ordered_labels_fe,
    coef_order  = names(ordered_labels_fe),
    stars       = c('***' = .01, '**' = .05, '†' = .1),
    add_rows    = custom_fe_rows,
    gof_omit    = 'FE:|IC|Log.Lik|Std.Errors|AIC|BIC|Deviance|R2',
    statistic   = c("conf.int", "p.value"),  # Add p-value here
    fmt         = function(x) sprintf("%.3f", x),  # Format to 3 decimals
    exponentiate = TRUE,  
    output      = paste0(tables_path, "regression_table_FE.tex"),
    title       = "Poisson Fixed Effects Models (IRR)"
  )
}

if(FE){
  # Variables to plot, in the same order as ordered_labels
  plot_vars <- c(
    "dfo_severity", "duration", "nevents_sum10", "population_affected", 
    "hdi_l1", "wdi_gdppc", 
    "e_wbgi_vae", "v2xpe_exlsocgr", "e_wbgi_gee", "v2x_rule", "decay_brds_c", "brd_12mb"
  )
  
  # Custom display order for y-axis (top to bottom in plot)
  ordered_labels <- c(
    "Accountability",
    "Inclusion",
    "Gov. effectiveness",
    "Rule of law",
    "Conflict history",
    "Local conflict",
    "Exposed population",
    "Flood severity",
    "Flood duration",
    "Past flood events",
    "Local HDI",
    "National GDP per capita"
  )
  
  
  
  # Mapping from model names to variable added in that model
  added_vars <- c(
    Accountability = "e_wbgi_vae",
    Inclusion = "v2xpe_exlsocgr",
    "Gov. effectiveness" = "e_wbgi_gee",
    "Rule of law" = "v2x_rule",
    "Conflict history" = "decay_brds_c",
    "Local conflict" = "brd_12mb"
  )
  
  # Baseline variables (present in Baseline model)
  v_baseline <- c(
    "dfo_severity", "duration", "nevents_sum10", "population_affected", "rugged", "hdi_l1", "wdi_gdppc", "tropical_flood"
  )
  
  # Ordered pretty labels for y axis
  custom_labels <- c(
    population_affected = "Exposed population",
    dfo_severity = "Flood severity",
    duration = "Flood duration",
    nevents_sum10 = "Past flood events",
    hdi_l1 = "Local HDI",
    wdi_gdppc = "National GDP per capita",
    e_wbgi_vae = "Accountability",
    v2xpe_exlsocgr = "Inclusion",
    e_wbgi_gee = "Gov. effectiveness",
    v2x_rule = "Rule of law",
    decay_brds_c = "Conflict history",
    brd_12mb = "Local conflict"
  )
  # Stack all model coefs, add model name and original variable
  coefs <- purrr::map2_dfr(models, names(models), ~
  broom::tidy(.x) %>%
    mutate(model = .y, varname = term)
)

  coefs_plot <- coefs %>%
  filter(
    (model == "Baseline" & varname %in% v_baseline) |
      (model != "Baseline" & varname == added_vars[model])
  ) %>%
  mutate(
    term_label = custom_labels[varname],
    color_var = varname,
    
    # Use raw log-coefficient for confidence interval
    estimate_log = estimate,
    estimate = exp(estimate_log),
    ymin = exp(estimate_log - 1.96 * std.error),
    ymax = exp(estimate_log + 1.96 * std.error)
  )


  # Ensure correct factor order for the y axis
  coefs_plot$term_label <- factor(coefs_plot$term_label, levels = rev(ordered_labels))
  
  ggplot(coefs_plot, aes(x = term_label, y = estimate, color = color_var)) +
    geom_point(size = 3) +
    labs(shape = "Clipped CI") +
    geom_errorbar(aes(ymin = ymin, ymax = ymax), width = 0.15) +
    scale_y_log10(
      breaks = c(0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000),
      labels = c("0.0001", "0.001", "0.01", "0.1", "1", "10", "100", "1000")
    ) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "gray40") +
    scale_color_manual(values = colors, guide = "none") +
    coord_flip() +
    labs(
      x = NULL,
      y = "Incidence rate ratio" # Or "Incidence rate ratio (exp(coef))" if you want to plot exp(estimate)
    ) +
    theme_minimal(base_family = "Arial") +
    theme(
      axis.title = element_text(size = 16, color = "black"),
      axis.text = element_text(size = 14, color = "black"),
      plot.title = element_text(size = 16, face = "bold"),
      panel.grid.minor = element_blank(),
      plot.background = element_rect(fill = "white", color = "white"),
      panel.background = element_rect(fill = "white", color = "white")
    )
  ggsave(filename = paste0(plots_path, "coef_plot_FE.png"), dpi = 350)
}

#Diagnostics

#Fig S6b

if(GDP) {diagnostics <- summary(fit_baseline)
print(diagnostics)

print(round(diagnostics$fixed %>% dplyr::select(Rhat, Bulk_ESS, Tail_ESS), 0))}



####CONDITIONAL EFFECTs###########

##### Generate conditional effects

set.seed(SEEDNUM)

if(AGG | ECONTEST | FE) {print('skip!')
} else{
  #posterior epred
  cfx_pred_accountability <- conditional_effects(fit_df$fits$Accountability, method = "posterior_epred", plot = FALSE, prob = 0.8) %>% plot(plot = FALSE)
  cfx_pred_inclusion <- conditional_effects(fit_df$fits$Inclusion, method = "posterior_epred", plot = FALSE, prob = 0.8) %>% plot(plot = FALSE)
  cfx_pred_goveff <- conditional_effects(fit_df$fits$`Gov. effectiveness`, method = "posterior_epred", plot = FALSE, prob = 0.8) %>% plot(plot = FALSE)
  cfx_pred_ruleoflaw <- conditional_effects(fit_df$fits$`Rule of law`, method = "posterior_epred", plot = FALSE, prob = 0.8) %>% plot(plot = FALSE)
  cfx_pred_conflict_country <- conditional_effects(fit_df$fits$`Conflict history`, method = "posterior_epred", plot = FALSE, prob = 0.8) %>% plot(plot = FALSE)
  if(NOLOC){print("No local conflict!")}else {cfx_pred_conflict_sub <- conditional_effects(fit_df$fits$`Local conflict`, method = "posterior_epred", plot = FALSE, prob = 0.8) %>% plot(plot = FALSE)}
  if(GDP) {cfx_pred_poly <- conditional_effects(fit_df$fits$`Electoral democracy`, method = "posterior_epred", plot = FALSE, prob = 0.8) %>% plot(plot = FALSE)}else{print("No Poly!")}
  cfx_pred_baseline <- conditional_effects(fit_df$fits$Baseline, method = "posterior_epred", plot = FALSE, prob = 0.8) %>% plot(plot = FALSE)
  

  # Adjust y-axis limits 
  
  max_y_conflict <- max(quantile(cfx_pred_conflict_country$decay_brds_c$data$upper__, 0.80))
  max_y_conflict
  
  # Replace 'your_data' with your actual data frame or object containing plot data
  common_y_limits <- c(0, max_y_conflict)
  
  
  
  #Function to get range of x variable
  get_x_range <- function(plot_data, common_y_limits, x_variable) {
    data_within_y <- subset(plot_data, lower__ >= common_y_limits[1] & upper__ <= common_y_limits[2])
    return(range(data_within_y[[x_variable]], na.rm = TRUE))
  }
  
  #Function to generate plot
  
  generate_plot <- function(plot_data, x_variable, x_label, color, common_y_limits) {
    x_range <- get_x_range(plot_data, common_y_limits, x_variable)
    
    plot <- ggplot(plot_data) +
      geom_rug(data = train, aes(x = !!sym(x_variable)), sides = "b") +
      geom_ribbon(aes(x = !!sym(x_variable), y = estimate__, ymin = lower__, ymax = upper__), fill = color) +
      geom_line(aes(x = !!sym(x_variable), y = estimate__), color = "black", size = 0.8) +
      labs(x = x_label, y = "Flood mortality") +
      xlim(x_range) +
      coord_cartesian(ylim = common_y_limits) +  # Use this instead of ylim()
      theme_minimal() +
      theme(
        axis.line = element_line(), 
        axis.ticks = element_line(),
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill = "white", color = "white"),
        panel.background = element_rect(fill = "white", color = "white"),
        axis.title = element_text(size = 14, color = "black", family="Arial"),
        axis.text = element_text(size = 12, color = "black", family="Arial"),
        legend.text = element_text(size = 12, color = "black", family="Arial"),
        legend.title = element_text(size = 12, color = "black", family="Arial"),
        plot.tag = element_text(size = 14, color = "black", family="Arial")
      )
    
    return(plot)
  }
  
  # List of variables with corresponding data
  
  if(NOGDP){variables <- list(
    `e_wbgi_vae` = cfx_pred_accountability$e_wbgi_vae[[1]],
    `v2xpe_exlsocgr` = cfx_pred_inclusion$v2xpe_exlsocgr[[1]],
    `e_wbgi_gee` = cfx_pred_goveff$e_wbgi_gee[[1]],
    `v2x_rule` = cfx_pred_ruleoflaw$v2x_rule[[1]],
    `decay_brds_c` = cfx_pred_conflict_country$decay_brds_c[[1]],
    `brd_12mb` = cfx_pred_conflict_sub$brd_12mb[[1]],
    `population_affected` = cfx_pred_baseline$population_affected[[1]],
    `dfo_severity` = cfx_pred_baseline$dfo_severity[[1]],
    `duration` = cfx_pred_baseline$duration[[1]],
    `nevents_sum10` = cfx_pred_baseline$nevents_sum10[[1]],
    `rugged` = cfx_pred_baseline$rugged[[1]],
    `tropical_flood` = cfx_pred_baseline$tropical_flood[[1]]
  )}else if (NOLOC) {variables <- list(
    `e_wbgi_vae` = cfx_pred_accountability$e_wbgi_vae[[1]],
    `v2xpe_exlsocgr` = cfx_pred_inclusion$v2xpe_exlsocgr[[1]],
    `e_wbgi_gee` = cfx_pred_goveff$e_wbgi_gee[[1]],
    `v2x_rule` = cfx_pred_ruleoflaw$v2x_rule[[1]],
    `decay_brds_c` = cfx_pred_conflict_country$decay_brds_c[[1]],
    `population_affected` = cfx_pred_baseline$population_affected[[1]],
    `dfo_severity` = cfx_pred_baseline$dfo_severity[[1]],
    `duration` = cfx_pred_baseline$duration[[1]],
    `nevents_sum10` = cfx_pred_baseline$nevents_sum10[[1]],
    `rugged` = cfx_pred_baseline$rugged[[1]],
    `tropical_flood` = cfx_pred_baseline$tropical_flood[[1]],
    `wdi_gdppc` = cfx_pred_baseline$wdi_gdppc[[1]]
  )}else if(GDP){variables <- list(
    `e_wbgi_vae` = cfx_pred_accountability$e_wbgi_vae[[1]],
    `v2xpe_exlsocgr` = cfx_pred_inclusion$v2xpe_exlsocgr[[1]],
    `e_wbgi_gee` = cfx_pred_goveff$e_wbgi_gee[[1]],
    `v2x_rule` = cfx_pred_ruleoflaw$v2x_rule[[1]],
    `decay_brds_c` = cfx_pred_conflict_country$decay_brds_c[[1]],
    `brd_12mb` = cfx_pred_conflict_sub$brd_12mb[[1]],
    `v2x_polyarchy` = cfx_pred_poly$v2x_polyarchy[[1]],
    `population_affected` = cfx_pred_baseline$population_affected[[1]],
    `dfo_severity` = cfx_pred_baseline$dfo_severity[[1]],
    `duration` = cfx_pred_baseline$duration[[1]],
    `nevents_sum10` = cfx_pred_baseline$nevents_sum10[[1]],
    `rugged` = cfx_pred_baseline$rugged[[1]],
    `tropical_flood` = cfx_pred_baseline$tropical_flood[[1]],
    `wdi_gdppc` = cfx_pred_baseline$wdi_gdppc[[1]],
    `hdi_l1` = cfx_pred_baseline$hdi_l1[[1]]
  )} else {variables <- list(
    `e_wbgi_vae` = cfx_pred_accountability$e_wbgi_vae[[1]],
    `v2xpe_exlsocgr` = cfx_pred_inclusion$v2xpe_exlsocgr[[1]],
    `e_wbgi_gee` = cfx_pred_goveff$e_wbgi_gee[[1]],
    `v2x_rule` = cfx_pred_ruleoflaw$v2x_rule[[1]],
    `decay_brds_c` = cfx_pred_conflict_country$decay_brds_c[[1]],
    `brd_12mb` = cfx_pred_conflict_sub$brd_12mb[[1]],
    `population_affected` = cfx_pred_baseline$population_affected[[1]],
    `dfo_severity` = cfx_pred_baseline$dfo_severity[[1]],
    `duration` = cfx_pred_baseline$duration[[1]],
    `nevents_sum10` = cfx_pred_baseline$nevents_sum10[[1]],
    `rugged` = cfx_pred_baseline$rugged[[1]],
    `tropical_flood` = cfx_pred_baseline$tropical_flood[[1]],
    `wdi_gdppc` = cfx_pred_baseline$wdi_gdppc[[1]],
    `hdi_l1` = cfx_pred_baseline$hdi_l1[[1]]
  )} 
  
  
  if(NOGDP){labels <- c(
    e_wbgi_vae = "Accountability",
    v2xpe_exlsocgr = "Inclusion",
    e_wbgi_gee = "Gov. effectiveness",
    v2x_rule = "Rule of law",
    decay_brds_c = "Conflict history",
    brd_12mb = "Local conflict",
    population_affected = "Exposed population",
    dfo_severity = "Flood severity",
    duration = "Flood duration",
    nevents_sum10 = "Past flood events",
    rugged = "Rugged terrain",
    tropical_flood = "Tropical flood"
  )
  }else if(NOLOC) {labels <- c(
    e_wbgi_vae = "Accountability",
    v2xpe_exlsocgr = "Inclusion",
    e_wbgi_gee = "Gov. effectiveness",
    v2x_rule = "Rule of law",
    decay_brds_c = "Conflict history",
    population_affected = "Exposed population",
    dfo_severity = "Flood severity",
    duration = "Flood duration",
    nevents_sum10 = "Past flood events",
    rugged = "Rugged terrain",
    tropical_flood = "Tropical flood", 
    wdi_gdppc = "National GDP per capita"
  )
  }else if(GDP) {labels <- c(
    e_wbgi_vae = "Accountability",
    v2xpe_exlsocgr = "Inclusion",
    e_wbgi_gee = "Gov. effectiveness",
    v2x_rule = "Rule of law",
    decay_brds_c = "Conflict history",
    brd_12mb = "Local conflict",
    v2x_polyarchy = "Electoral democracy",
    population_affected = "Exposed population",
    dfo_severity = "Flood severity",
    duration = "Flood duration",
    nevents_sum10 = "Past flood events",
    rugged = "Rugged terrain",
    tropical_flood = "Tropical flood",
    wdi_gdppc = "National GDP per capita",
    hdi_l1 = "Local HDI"
  )
  }else{labels <- c(
    e_wbgi_vae = "Accountability",
    v2xpe_exlsocgr = "Inclusion",
    e_wbgi_gee = "Gov. effectiveness",
    v2x_rule = "Rule of law",
    decay_brds_c = "Conflict history",
    brd_12mb = "Local conflict",
    population_affected = "Exposed population",
    dfo_severity = "Flood severity",
    duration = "Flood duration",
    nevents_sum10 = "Past flood events",
    rugged = "Rugged terrain",
    tropical_flood = "Tropical flood",
    wdi_gdppc = "National GDP per capita",
    hdi_l1 = "Local HDI"
  )
  }
  
  # Loop to generate plots for each variable
  all_plots <- list()
  for (x_variable in names(variables)) {
    plot_data <- variables[[x_variable]]
    plot_label <- labels[[x_variable]]
    plot_color <- colors[x_variable]
    plot <- generate_plot(plot_data, x_variable, plot_label, plot_color, common_y_limits)
    all_plots[[x_variable]] <- plot
  }
  
  # Optional: Display or save plots
  for (name in names(all_plots)) {
    print(all_plots[[name]])
    ggsave(filename = paste0(plots_path, name, ".png"), plot = all_plots[[name]], width = 10, height = 8)
  }
  
  # Combine the plots
  
  
  # Combine plots
  if(NOLOC){combined_plot <- wrap_plots(all_plots$e_wbgi_vae, all_plots$e_wbgi_gee, all_plots$decay_brds_c, 
                                        all_plots$v2xpe_exlsocgr, all_plots$v2x_rule, ncol = 3)  
  }else{combined_plot <- wrap_plots(all_plots$e_wbgi_vae, all_plots$e_wbgi_gee, all_plots$decay_brds_c,
                                    all_plots$v2xpe_exlsocgr, all_plots$v2x_rule, all_plots$brd_12mb, ncol = 3)
  }
  
  combined_plot
  
  combined_plot <- combined_plot & 
    plot_annotation(
      tag_levels = list(c("Democracy", "Institutional quality", "Conflict"))
    ) & 
    theme(
      plot.tag.position = c(0.5, 1),  # Adjust the tag position globally
      plot.tag = element_text(size = 16, face = 'bold', hjust = 0.3, vjust = -0.8),  # Adjust hjust for better centering
      plot.margin = unit(c(15, 10, 10, 10), "pt")  # Define the margin with explicit units
    )
  # poly 
  
  if (NOLOC) {
    combined_plot_all <- wrap_plots(all_plots$e_wbgi_vae,  all_plots$e_wbgi_gee, all_plots$decay_brds_c, 
                                    all_plots$v2xpe_exlsocgr, all_plots$v2x_rule, all_plots$population_affected,
                                    all_plots$dfo_severity, all_plots$duration, all_plots$nevents_sum10,
                                    all_plots$rugged, all_plots$tropical_flood, all_plots$wdi_gdppc, ncol = 3)  # Define the layout with 3 columns
  }else if(GDP) {
    #tag_labels <- c("a)", "b)", "c)", "d)", "e)", "f)", "g)", "h)", "i)", "j)", "k)", "l)", "m)", "n)","o)")
    combined_plot_all <- wrap_plots(all_plots$e_wbgi_vae, all_plots$e_wbgi_gee, all_plots$decay_brds_c,
                                    all_plots$v2xpe_exlsocgr, all_plots$v2x_rule, all_plots$brd_12mb,
                                      all_plots$v2x_polyarchy,
                                    all_plots$population_affected, all_plots$dfo_severity, all_plots$duration,
                                    all_plots$nevents_sum10, all_plots$rugged, all_plots$tropical_flood,
                                    all_plots$wdi_gdppc, all_plots$hdi_l1, ncol = 3)
    
    # Optional: Adjust the positioning or size of the tags
    combined_plot_all <- combined_plot_all & theme(
      plot.tag.position = c(0.01, 0.01),  # Adjusts the position of the tags, move right with the first value
      plot.tag = element_text(size = 14, face = 'bold', hjust = 0, vjust = 0),  # Adjusts alignment
      plot.margin = unit(c(0.3, 0.3, 0.3, 0.5), "cm")  # Add more space at the top (first value is for top margin)
    )
  }else{
    combined_plot_all <- wrap_plots(
      all_plots$e_wbgi_vae, all_plots$e_wbgi_gee, all_plots$decay_brds_c,
      all_plots$v2xpe_exlsocgr, all_plots$v2x_rule, all_plots$brd_12mb,
      ncol = 3)
  }
  print(combined_plot_all)
  
  
  if(GDP){ggsave(filename = paste0(plots_path, "cfx-posterior-epred-80predint-poly-SI.png"), device = png, width = 5.6,  height = 4.6, scale = 2)}else{print("No Poly!")}
  
  if(NOGDP){
    combined_plot_all <- wrap_plots(all_plots$e_wbgi_vae, all_plots$e_wbgi_gee, all_plots$decay_brds_c,
                                    all_plots$v2xpe_exlsocgr, all_plots$v2x_rule, all_plots$brd_12mb,
                                    all_plots$population_affected, all_plots$dfo_severity, all_plots$duration,
                                    all_plots$nevents_sum10, all_plots$rugged, all_plots$tropical_flood, 
                                    ncol = 3) }else if (NOLOC) {
                                      combined_plot_all <- wrap_plots(all_plots$e_wbgi_vae,all_plots$e_wbgi_gee,all_plots$decay_brds_c, 
                                                                      all_plots$v2xpe_exlsocgr,all_plots$v2x_rule, all_plots$population_affected, 
                                                                      all_plots$dfo_severity, all_plots$duration, all_plots$nevents_sum10,
                                                                      all_plots$rugged,all_plots$tropical_flood, all_plots$wdi_gdppc, ncol = 3)  # Define the layout with 3 columns
                                    }else{
                                      # Combine plots with annotations (letters)
                                      
                                      
                                      combined_plot_all <- wrap_plots(all_plots$e_wbgi_vae, all_plots$e_wbgi_gee, all_plots$decay_brds_c,
                                                                      all_plots$v2xpe_exlsocgr, all_plots$v2x_rule, all_plots$brd_12mb,
                                                                      all_plots$population_affected, all_plots$dfo_severity, all_plots$duration,
                                                                      all_plots$nevents_sum10, all_plots$rugged, all_plots$tropical_flood,
                                                                      all_plots$wdi_gdppc, all_plots$hdi_l1, ncol = 3)
                                      
                                      # Optional: Adjust the positioning or size of the tags
                                      combined_plot_all <- combined_plot_all & theme(
                                        plot.tag.position = c(0.01, 0.01),  # Adjusts the position of the tags, move right with the first value
                                        plot.tag = element_text(size = 14, face = 'bold', hjust = 0, vjust = 0),  # Adjusts alignment
                                        plot.margin = unit(c(0.3, 0.3, 0.3, 0.5), "cm")  # Add more space at the top (first value is for top margin)
                                      )
                                      
                                    } 
  
  
  print(combined_plot_all)
  # Save the combined plot with annotations
  ggsave(filename = paste0(plots_path, "Fig3.png"), plot = combined_plot_all, device = png, width = 5.6,  height = 4.6, scale = 2, dpi = 350)
  
  
}




# Calculate model weights in-sample

if(AGG){
  mw_res <- model_weights(
    models$Baseline,
    models$Conflict,
    models$Democracy,
    models$`Institutional quality`,
    models$`Electoral democracy`
  )
} else if (ECONTEST){
  mw_res <- model_weights(
    models$Baseline, 
   models$`Baseline w conflict`
  )} else if(NOLOC) {
    mw_res <- model_weights(
      models$Baseline,
      models$`Conflict history`,
      models$Accountability,
      models$`Gov. effectiveness`,
      models$Inclusion,
      models$`Rule of law`
    )
  } else {
    mw_res <- model_weights(
      models$Baseline,
      models$`Conflict history`,
      models$`Local conflict`,
      models$Accountability,
      models$`Gov. effectiveness`,
      models$Inclusion,
      models$`Rule of law`
    )
  }
names(mw_res) <- gsub("models\\$", "", names(mw_res))  # Remove 'models$'
names(mw_res) <- gsub("`", "", names(mw_res))          # Remove backticks


# Calculate model weights out-of-sample using the test dataset

if(AGG){
  mw_res_oos <- model_weights(
    models$Baseline,
    models$Conflict,
    models$Democracy,
    models$`Institutional quality`,
    models$`Electoral democracy`,
    newdata = test, 
    allow_new_levels = TRUE
  )
} else if (ECONTEST){
  mw_res_oos <- model_weights(
    models$`Baseline`, 
    models$`Baseline w conflict`,
    newdata = test, 
    allow_new_levels = TRUE)
} else if(NOLOC) {
  mw_res_oos <- model_weights(
    models$Baseline,
    models$`Conflict history`,
    models$Accountability,
    models$`Gov. effectiveness`,
    models$Inclusion,
    models$`Rule of law`,
    newdata = test, 
    allow_new_levels = TRUE
  )
} else {
  mw_res_oos <- model_weights(
    models$Baseline,
    models$`Conflict history`,
    models$`Local conflict`,
    models$Accountability,
    models$`Gov. effectiveness`,
    models$Inclusion,
    models$`Rule of law`,
    newdata = test, 
    allow_new_levels = TRUE
  )
  
}


names(mw_res_oos) <- gsub("models\\$", "", names(mw_res_oos))  # Remove 'models$'
names(mw_res_oos) <- gsub("`", "", names(mw_res_oos))          # Remove backticks


# Calculate LOO in-sample
if(AGG) {
  loo_res <- loo(
    models$Baseline,
    models$Conflict,
    models$Democracy,
    models$`Institutional quality`, 
    models$`Electoral democracy`
  )
} else if (ECONTEST){
  loo_res <- loo(
    models$`Baseline`, 
    models$`Baseline w conflict`
  )}else if(NOLOC) {
    loo_res <- loo(
      models$Baseline,
      models$`Conflict history`,
      models$Accountability,
      models$`Gov. effectiveness`,
      models$Inclusion,
      models$`Rule of law`
    )
  } else {
    loo_res <- loo(
      models$Baseline,
      models$`Conflict history`,
      models$`Local conflict`,
      models$Accountability,
      models$`Gov. effectiveness`,
      models$Inclusion,
      models$`Rule of law`
    )
  }


# Match the rownames of the diffs in the LOO object to the model names

# Calculate LOO out-of-sample using the test dataset
if(AGG) {
  loo_res_oos <- loo(
    models$Baseline,
    models$Conflict,
    models$Democracy,
    models$`Institutional quality`,
    models$`Electoral democracy`,
    newdata = test,
    allow_new_levels = TRUE
  ) } else if (ECONTEST){
    loo_res_oos <- loo(
      models$`Baseline`, 
      models$`Baseline w conflict`,
      newdata = test, 
      allow_new_levels = TRUE
    )}else if(NOLOC) {
      loo_res_oos <- loo(
        models$Baseline,
        models$`Conflict history`,
        models$Accountability,
        models$`Gov. effectiveness`,
        models$Inclusion,
        models$`Rule of law`,
        newdata = test, 
        allow_new_levels = TRUE
      )
    }else {
      loo_res_oos <- loo(
        models$Baseline,
        models$`Conflict history`,
        models$`Local conflict`,
        models$Accountability,
        models$`Gov. effectiveness`,
        models$Inclusion,
        models$`Rule of law`,
        newdata = test, 
        allow_new_levels = TRUE
      )
    }


# Match the rownames of the diffs in the LOO object to the model names for out-of-sample

rownames(loo_res$diffs) <- gsub("`", "", sub("models\\$", "", rownames(loo_res$diffs)))

predictive_performance_table <- loo_res$diffs %>% as_tibble(rownames = "mname") %>% 
  mutate(across(!all_of("mname"), ~ as.numeric(.x))) %>% 
  mutate(across(!all_of("mname"), ~ round(.x, digits = 0)))  %>% 
  mutate(elpd_loo = paste0(elpd_loo, " (", se_elpd_loo, ")"),
         elpd_diff = paste0(elpd_diff, " (", se_diff, ")")) %>%
  dplyr::select(mname, elpd_loo, elpd_diff) %>%
  left_join(mw_res %>% round(digits = 2) %>% as_tibble(rownames = "mname")  %>% rename("stacking_weight" = "value"), by = "mname")

rownames(loo_res_oos$diffs) <- gsub("`", "", sub("models\\$", "", rownames(loo_res_oos$diffs)))

predictive_performance_table_oos <- loo_res_oos$diffs %>% as_tibble(rownames = "mname") %>% 
  mutate(across(!all_of("mname"), ~ as.numeric(.x))) %>% 
  mutate(across(!all_of("mname"), ~ round(.x, digits = 0)))  %>% 
  mutate(elpd_loo_oos = paste0(elpd_loo, " (", se_elpd_loo, ")"),
         elpd_diff_oos = paste0(elpd_diff, " (", se_diff, ")")) %>%
  dplyr::select(mname, elpd_loo_oos, elpd_diff_oos) %>%
  left_join(mw_res_oos %>% round(digits = 2) %>% as_tibble(rownames = "mname")  %>% rename("stacking_weight_oos" = "value"), by = "mname")


# Add "c)" as a custom title aligned with the first column (mname)
if(SPLIT){gt_table <- gt(left_join(predictive_performance_table, predictive_performance_table_oos, by = "mname")) %>%
  tab_header(
    title = md("**c)**")  # Adds "c)" at the top and aligns it to the left
  ) %>%
  tab_spanner(label = "In-sample", 
              columns = c("elpd_loo", "elpd_diff", "stacking_weight")) %>%
  tab_spanner(label = "Out-of-sample",
              columns = c("elpd_loo_oos", "elpd_diff_oos", "stacking_weight_oos")) %>%
  cols_label(mname = "Model",
             elpd_loo = "elpd LOO",
             elpd_diff = html("&Delta;elpd"),
             stacking_weight = "stacking weight",
             elpd_loo_oos = "elpd LOO",
             elpd_diff_oos = html("&Delta;elpd"),
             stacking_weight_oos = "stacking weight") %>%
  tab_options(
    table.font.names = 'Arial', 
    table.font.color = "black",
    heading.align = "left",  # Align header (c)) to the left
    table.border.top.width = px(0),  # Remove the top border (grey line)
    table.border.top.color = "white"  # Set the top border color to white (invisible)
  ) %>%
  tab_style(
    style = list(
      cell_borders(sides = "left", color = "lightgray", weight = px(2))
    ),
    locations = cells_body(columns = c(elpd_loo_oos))
  ) %>%
  tab_style(
    style = list(
      cell_borders(sides = "left", color = "lightgray", weight = px(2))
    ),
    locations = cells_column_labels(columns = c(elpd_loo_oos))
  ) %>%
  gtsave(paste0(plots_path, "/Fig4c.html")) 
}else{
  gt_table <- gt(left_join(predictive_performance_table, predictive_performance_table_oos, by = "mname")) %>%
    tab_header(
      title = md("**c)**")  # Adds "c)" at the top and aligns it to the left
    ) %>%
    tab_spanner(label = "In-sample (2000-2014)", 
                columns = c("elpd_loo", "elpd_diff", "stacking_weight")) %>%
    tab_spanner(label = "Out-of-sample (2015-2018)",
                columns = c("elpd_loo_oos", "elpd_diff_oos", "stacking_weight_oos")) %>%
    cols_label(mname = "Model",
               elpd_loo = "elpd LOO",
               elpd_diff = html("&Delta;elpd"),
               stacking_weight = "stacking weight",
               elpd_loo_oos = "elpd LOO",
               elpd_diff_oos = html("&Delta;elpd"),
               stacking_weight_oos = "stacking weight") %>%
    tab_options(
      table.font.names = 'Arial', 
      table.font.color = "black",
      heading.align = "left",  # Align header (c)) to the left
      table.border.top.width = px(0),  # Remove the top border (grey line)
      table.border.top.color = "white"  # Set the top border color to white (invisible)
    ) %>%
    tab_style(
      style = list(
        cell_borders(sides = "left", color = "lightgray", weight = px(2))
      ),
      locations = cells_body(columns = c(elpd_loo_oos))
    ) %>%
    tab_style(
      style = list(
        cell_borders(sides = "left", color = "lightgray", weight = px(2))
      ),
      locations = cells_column_labels(columns = c(elpd_loo_oos))
    ) %>%
    gtsave(paste0(plots_path, "/Fig4c.html"))
}

# Convert the HTML to a PNG with higher DPI (e.g., 400 DPI)
webshot(
  paste0(plots_path, "/Fig4c.html"),
  file = paste0(plots_path, "/Fig4c.png"),
  zoom = 10.33,                # High DPI setting (~400 DPI)
  expand = c(0, 0, 0, 0),      # No extra white space
  vwidth = 1000,               # Set width to match the table (adjust this based on your table size)
  vheight = 600                # Set height to match the table (adjust this based on your table size)
)

###Simplified table:
gt_table <- left_join(predictive_performance_table, predictive_performance_table_oos, by = "mname") %>%
  dplyr::select("mname", "elpd_loo", "elpd_loo_oos") %>%
  gt() %>% 
  fmt_markdown(columns = everything()) %>% 
  cols_label(mname = "Model",
             elpd_loo = "In-sample",
             elpd_loo_oos = "Out-of-sample") %>%
  cols_width(elpd_loo ~ px(120), 
             elpd_loo_oos ~ px(120)) %>%
  tab_style(
    style = list(
      cell_borders(sides = "left", color = "lightgray", weight = px(2))
    ),
    locations = cells_body(columns = c(elpd_loo_oos))
  ) %>%
  tab_style(
    style = list(
      cell_borders(sides = "left", color = "lightgray", weight = px(2))
    ),
    locations = cells_column_labels(columns = c(elpd_loo_oos))
  ) %>% 
  tab_options(table.font.names = 'Arial', table.font.color = "black")  %>% 
  gtsave(paste0(plots_path, "/predictive_performance.png"))



#Plot predictive performance table
# Combine the relevant data into one data frame



weights_table <- left_join(predictive_performance_table, predictive_performance_table_oos, by = "mname") %>% dplyr::select(mname, stacking_weight, stacking_weight_oos)

names(weights_table) <- c("Model", "In-sample", "Out-of-sample")
weights_table$Model <- factor(weights_table$Model)



if(AGG){
  model_colors <- c('Democracy' = "#8eade8",
                    'Institutional quality' =  "#f2aed8",
                    'Electoral democracy' = "#5df0e1",
                    'Conflict' = "#f5b342",
                    'Baseline' = "#CCCCCC") 
} else if (ECONTEST) {
  model_colors <- c('Baseline' = "#CCCCCC",
                    'Baseline w conflict' = "#f0843c") 
}else if(NOLOC) {
  model_colors <- c('Accountability' = "#8eade8",
                    'Inclusion' =  "#3c61a3"  ,
                    'Gov. effectiveness' = "#f2aed8",
                    'Rule of law' = "#c556d1",
                    'Conflict history' =  "#f5b342" ,
                    'Electoral democracy' = "#5df0e1",
                    'Baseline' = "#CCCCCC")
  
} else {
  model_colors <- c('Accountability' = "#8eade8",
                    'Inclusion' =  "#3c61a3"  ,
                    'Gov. effectiveness' = "#f2aed8",
                    'Rule of law' = "#c556d1",
                    'Conflict history' =  "#f5b342" ,
                    'Local conflict' = "#f0843c",
                    'Electoral democracy' = "#5df0e1",
                    'Baseline' = "#CCCCCC")
  
}

if(AGG){
  legend_order <- c("Democracy",
                    "Institutional quality",
                    "Conflict",
                    "Electoral democracy",
                    "Baseline") 
}else if (ECONTEST) { legend_order <- 
  c('Baseline',
    'Baseline w conflict'
  )
}else if(NOLOC){
  legend_order <- c(
    "Accountability",
    "Inclusion",
    "Gov. effectiveness",
    "Rule of law",
    "Conflict history",
    "Baseline")
}else{
  legend_order <- c(
    "Accountability",
    "Inclusion",
    "Gov. effectiveness",
    "Rule of law",
    "Conflict history",
    "Local conflict",
    "Baseline")
}



long_data <- pivot_longer(weights_table, cols = c(`In-sample`, `Out-of-sample`), 
                          names_to = "Sample_Type", 
                          values_to = "Score")

long_data <- long_data %>%
  group_by(Sample_Type) %>%
  mutate(Proportion = Score / sum(Score)) %>%
  ungroup()

if(AGG){legend_order <- c(
  "Democracy",
  "Institutional quality",
  "Conflict",
  "Electoral democracy",
  "Baseline"
)} else if (ECONTEST) { legend_order <- 
  c('Baseline',
    'Baseline w conflict')
}else if(NOLOC) {
  legend_order <- c(
    "Accountability",
    "Inclusion",
    "Gov. effectiveness",
    "Rule of law",
    "Conflict history",
    "Baseline"
  )
}else{
  legend_order <- c(
    "Accountability",
    "Inclusion",
    "Gov. effectiveness",
    "Rule of law",
    "Conflict history",
    "Local conflict",
    "Baseline"
  )
}

library(ggplot2)
library(scales)

# Plot with annotation outside the plot area
weights_plot <- ggplot(long_data, aes(x = Sample_Type, y = Score, fill = Model)) +
  geom_bar(stat = "identity", position = "fill", width = 0.2) +
  scale_y_continuous(labels = percent_format()) +
  scale_fill_manual(values = model_colors, breaks = legend_order) +
  guides(fill = guide_legend(reverse = FALSE)) +
  theme_minimal() +
  labs(x = "", y = "Stacking weights", fill = "Model") +
  theme(
    axis.text.x = element_text(angle = 0, hjust = 0.5, size = 15, family = "Arial", color = "black"),
    axis.text.y = element_text(size = 15, family = "Arial", color = "black"),
    axis.title = element_text(size = 15, family = "Arial", color = "black"),
    axis.title.x = element_text(size = 15, family = "Arial", color = "black"),
    axis.title.y = element_text(size = 15, family = "Arial", color = "black",
                                margin = ggplot2::margin(t = 0, r = 15, b = 0, l = 0)),  # Add margin on the right (r = 1
    legend.title = element_text(size = 15, family = "Arial", color = "black"),
    legend.text = element_text(size = 15, family = "Arial", color = "black"),
    plot.margin = margin(30, 2, 2, 2),   # Minimal margin inside the plot
    plot.background = element_rect(fill = "white", color = "white"),  # Ensure white plot background
    panel.background = element_rect(fill = "white", color = "white")
  )

# Use cowplot to add annotation outside the plot
weight_plot <- ggdraw() +
  draw_plot(weights_plot) +  # The main plot
  draw_label("b)", x = 0.01, y = 0.98, hjust = 0, vjust = 1, size = 15, fontface = "bold", font = 'Arial')

if(ECONTEST){
  ggsave(paste0(plots_path, "Fig4b.png"), plot = weight_plot, width = 7, height = 4, dpi = 350)
}else{
  ggsave(paste0(plots_path, "Fig4b.png"), plot = weight_plot, width = 6.2, height = 4, dpi = 350)
}

#Plot predictive performance table
# Combine the relevant data into one data frame


elpd_table <- left_join(predictive_performance_table, predictive_performance_table_oos, by = "mname") %>% dplyr::select(mname, elpd_diff, elpd_diff_oos)

#best in-sample: acc local conflict, worst: gov eff
#best oos: local conflict rule law gov effectiveness, worst poly, inclusion

names(elpd_table) <- c("Model", 'In-sample', 'Out-of-sample')
elpd_table$Model <- factor(elpd_table$Model)


elpd_table$`In-sample` <- gsub(" \\(.*\\)", "", elpd_table$`In-sample`)
elpd_table$`Out-of-sample` <- gsub(" \\(.*\\)", "", elpd_table$`Out-of-sample`)


# Normalize the scores to a 0-1 range
elpd_table$`In-sample` <- rescale(as.numeric(elpd_table$`In-sample`), to = c(0,1)) #1 is best 0 is worse
elpd_table$`Out-of-sample` <- rescale(as.numeric(elpd_table$`Out-of-sample`, to = c(0,1)))

print(elpd_table)


# Convert to long format
long_data <- pivot_longer(elpd_table, cols = c("In-sample", "Out-of-sample"), names_to = "Sample_Type", values_to = "Score")
long_data$Sample_Type <- factor(long_data$Sample_Type, levels = c("In-sample", "Out-of-sample"),
                                labels = c("In-sample", "Out-of-sample"))


elpd_plot <- ggplot(long_data, aes(x = Sample_Type, y = Score, color = Model)) +
  geom_beeswarm(method = "centre", shape = 18, size = 5, cex = 3.5) +
  theme_minimal() +
  scale_color_manual(values = model_colors, breaks = legend_order) +
  labs(x = "", y = "Normalized elpd difference", fill = "Model") +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5, size = 15, family="Arial", color = "black"),
        legend.position = "none",
        axis.text.y = element_text(size = 15, family="Arial", color = "black", margin = margin(r = 15)), 
        axis.title = element_text(size = 15, family="Arial", color = "black"),
        axis.title.x = element_text(size = 15, family="Arial", color = "black"),
        axis.title.y = element_text(size = 15, family="Arial", color = "black", margin = margin(r = 15)),
        legend.title = element_text(size = 15, family="Arial", color = "black"),
        legend.text = element_text(size = 15, family="Arial", color = "black"), 
        plot.margin = margin(30, 2, 2, 2), # Minimal margin inside the plot
        plot.background = element_rect(fill = "white", color = "white"),  # Ensure white plot background
        panel.background = element_rect(fill = "white", color = "white")
  )
# Use cowplot to add annotation outside the plot

elpd_plot <- ggdraw() +
  draw_plot(elpd_plot) +  # The main plot
  draw_label("a)", x = 0.015, y = 0.99, hjust = 0, vjust = 1, size = 15, fontface = "bold", font = 'Arial')

if(ECONTEST){
 ggsave(paste0(plots_path, "Fig4a.png"), plot = elpd_plot,  width = 5, height = 4, dpi = 350)
}else{
  ggsave(paste0(plots_path, "Fig4a.png"), plot = elpd_plot,  width = 6.2, height = 4, dpi = 350)
}



####COUNTERFACTUAL ANALYSIS####

if(!GDP){print("skip!")}else{
#Predictive fit aggregated at global level ####

#Take value of political indicators for New Zealand in 2018

nz <- readRDS("data/data_newzealand.Rds") 

nz$v2xpe_exlsocgr = 1 - nz$v2xpe_exlsocgr

pol <- c("brd_12mb","decay_brds_c","v2xpe_exlsocgr" ,"e_wbgi_vae", "e_wbgi_gee",         
         "v2x_rule" , "v2x_polyarchy") 

#New Zealand vs test dataset comparison


# Percentiles to calculate
percentiles <- c(0.5, 0.9, 0.95, 0.99)

# Initialize a list to store results
results <- list()

## to retrieve info also on gdp and hdi 

pol2 <- c("v2xpe_exlsocgr" ,"e_wbgi_vae", "e_wbgi_gee",         
          "v2x_rule" , "v2x_polyarchy", "wdi_gdppc", "hdi_l1")

# Loop through each variable and calculate the specified percentiles
for(variable in pol2) {
  # Extract the column data from the dataframe
  column_data <- test[[variable]]
  
  # Calculate the percentiles for the current variable
  percentile_values <- round(quantile(column_data, percentiles, na.rm = TRUE), 2)
  
  # Store the results in the list
  results[[variable]] <- percentile_values
}

# Print or return the results
print(results)


df$continent <- countrycode::countrycode(df$iso3c, "iso3c", "continent")

train <- df %>% dplyr::filter(year >= 2000 & year <= 2014) %>% dplyr::select(all)
test <- df %>% dplyr::filter(year >= 2015) %>% dplyr::select(all)

# Create a copy of df

df_nz <- df
train_nz <- train
test_nz <- test

# Replace values in df_nz with values of the political indicators from nz for the year 2018 in our dataset 
for (var in pol) {
  train_nz[, var] <- nz[, var]
  test_nz[, var] <- nz[, var]
  df_nz[, var] <- nz[, var]
}

###FIGURE

x_limits90 = c(0,200)
y_limits = NULL


q10 <- function(y) quantile(y, 0.1)
q90 <- function(y) quantile(y, 0.9)
q25 <- function(y) quantile(y, 0.025)
q975 <- function(y) quantile(y, 0.975)


##############All political indicators################

#load models
fit_all <- readRDS("results/panel/aggregate/bayes_models/all_model.rds")

data_all_test <- ppc_stat_data(y = test$dead_w, yrep = posterior_predict(fit_all, newdata = test, allow_new_levels = TRUE), stat = "q90")  
data_all_nz_test <- ppc_stat_data(y = test_nz$dead_w, yrep = posterior_predict(fit_all, newdata = test_nz, allow_new_levels = TRUE), stat = "q90") 

binwidth_defined <- 3 

#Start the ggplot call
p <- ggplot() + theme_minimal()


#Function to plot histogram

add_dataset_layers <- function(p, data, dataset_name, binwidth_defined) {
  p + 
    geom_histogram(data = filter(data, variable != "y"), #Assuming 'variable' is the correct column name
                   aes(x = value, fill = dataset_name), 
                   color = "black", linewidth = 0.25, na.rm = TRUE, 
                   binwidth = binwidth_defined) # Apply alpha here
}


p <- add_dataset_layers(p, data_all_test, "All, actual", binwidth_defined)
p <- add_dataset_layers(p, data_all_nz_test, "All, New Zealand", binwidth_defined)

p <- p + 
  scale_fill_manual(values = c("All, actual" = alpha("#F44B3E", 0.7), 
                               "All, New Zealand" = alpha("#2596be", 0.7))) +
  coord_cartesian(xlim = c(NA, 150)) # Limit x-axis range



# Add labels and adjust theme
p <- p + labs( x = "Flood deaths", y = "Predicted frequency") + 
  theme_minimal() +
  theme(legend.title = element_blank(),
        legend.position = "none",
        plot.title = element_text(size = 15, color = "black", family="Arial", hjust = 0.5),
        axis.text = element_text(size = 15, color = "black", family="Arial"),
        legend.text = element_text(size = 15, color = "black", family="Arial"), 
        axis.title =  element_text(size = 15, color = "black", family="Arial"), 
        axis.title.y = element_text(size = 15, family = "Arial", color = "black", margin = margin(r = 15)),
        plot.margin = margin(30, 2, 2, 2),   # Minimal margin inside the plot
        plot.background = element_rect(fill = "white", color = "white"),  # Ensure white plot background
        panel.background = element_rect(fill = "white", color = "white")
  )

# Use cowplot to add annotation outside the plot
p <- ggdraw() +
  draw_plot(p) +  # The main plot
  draw_label("a)", x = 0.01, y = 0.98, hjust = 0, vjust = 1, size = 15, fontface = "bold", font = 'Arial')

# Print the plot
print(p)

ggsave(paste0(plots_path, "Fig5a.png"), height = 5, width = 6, dpi = 350)


###SINGLE MODELS; GLOBAL SAMPLE

# Figure 5b

# Function to load model and generate predictions
generate_predictions <- function(model_path, test_data, test_nz_data, SEEDNUM) {
  set.seed(SEEDNUM)
  fit <- readRDS(model_path)
  predictions <- posterior_predict(fit, newdata = test, allow_new_levels = TRUE)
  predictions_nz <- posterior_predict(fit, newdata = test_nz, allow_new_levels = TRUE)
  list(predictions = predictions, predictions_nz = predictions_nz)
}

compute_summary_stats <- function(predictions, predictions_nz, model_name) {
  # Calculate the means for each set of predictions
  means_test <- colMeans(predictions)
  means_test_nz <- colMeans(predictions_nz)
  
  # Calculate the average of the means and round to 2 decimal places
  avg_means_test <- round(mean(means_test), 2)
  avg_means_test_nz <- round(mean(means_test_nz), 2)
  
  # Calculate the percentage change and round to 2 decimal places
  percentage_change <- round((avg_means_test_nz - avg_means_test) / avg_means_test * 100, 0)
  
  # Calculate the range for the average predictions and round to 2 decimal places
  min_avg_pred <- round(min(means_test) - min(means_test_nz), 2)
  max_avg_pred <- round(max(means_test) - max(means_test_nz), 2)
  
  min_avg_pred<- format(min_avg_pred, big.mark=",",scientific=FALSE)
  max_avg_pred <- format(max_avg_pred, big.mark=",",scientific=FALSE)
  
  # Return a data frame with the summary statistics
  data.frame(
    Model = model_name,
    Avg_Pred_Test = avg_means_test,
    Avg_Pred_Test_NZ = avg_means_test_nz,
    Percentage_Change = percentage_change,
    Range_Difference = paste0(min_avg_pred, " – ", max_avg_pred)
  )
}


  
  # Define model names and paths (make sure to replace these with actual paths)
  model_names <- c("Accountability", "Inclusion", "Gov. effectiveness", "Rule of law", "Conflict history", "Local conflict", "All pol. predictors")
  model_paths <- c(
    paste0(models_path, "accountability_model.rds"),
    paste0(models_path, "inclusion_model.rds"),
    paste0(models_path, "goveff_model.rds"),
    paste0(models_path, "ruleoflaw_model.rds"),
    paste0(models_path, "conflict_country_model.rds"),
    paste0(models_path, "conflict_sub_model.rds"),
    "results/panel/aggregate/bayes_models/all_model.rds")
  
  
  # Generate predictions and compute summary statistics for each model
  
  
  model_summaries <- list()
  
  for (i in 1:length(model_names)) {
    pred <- generate_predictions(model_paths[i], test, test_nz, SEEDNUM)
    model_summaries[[model_names[i]]] <- compute_summary_stats(pred$predictions, pred$predictions_nz, model_names[i])
  }
  
  # Combine all model summaries into a single data frame
  all_model_summaries <- do.call(rbind, model_summaries)
  
  
  # Create a gt table from the combined summary
  # Create a gt table from the combined summary
  gt_table <- gt(all_model_summaries) %>%
    fmt_number(
      columns = c(Avg_Pred_Test, Avg_Pred_Test_NZ),
      decimals = 2,
      use_seps = TRUE  # This will add the comma for thousands separator
    ) %>%
    fmt_number(
      columns = c(Percentage_Change),
      decimals = 0,  # Round to 0 decimal places for the percentage change
      pattern = "{x}%"  # Add the percentage sign
    ) %>%
    cols_label(
      Model = "Model",
      Avg_Pred_Test = html("y&#770;<sub>obs</sub>"),  # Using HTML entities for hat and subscript
      Avg_Pred_Test_NZ = html("y&#770;<sub>cnt</sub>"),
      Percentage_Change = html("&Delta;y&#770;"),
      Range_Difference = html("&Delta;y&#770; (range)")
    ) %>%
    tab_header(
      title = html("<b>b)</b>")  # Add "b)" in bold in the header
    ) %>%
    tab_style(
      style = list(
        cell_text(
          align = "left",  # Align the title to the left
          size = px(15),   # Set the font size
          weight = "bold"
        )
      ),
      locations = cells_title(groups = "title")  # Apply the style to the title
    ) %>%
    tab_options(
      table.font.names = 'Arial',
      table.font.color = "black",
      table.border.top.width = px(0),  # Remove the top border (grey line)
      table.border.top.color = "white",
      heading.title.font.size = 15 # Ensure the table takes up 100% of the width
    )
  
  # Save the gt table to an HTML file
  gtsave(gt_table, filename = paste0(tables_path, "model_summaries.html"))
  
  
  # Convert the HTML to a PNG with higher DPI (e.g., 400 DPI)
  webshot(
    paste0(tables_path, "model_summaries.html"),
    file = paste0(plots_path, "/Fig5b.png"),
    zoom = 10.33,                # High DPI setting (~400 DPI)
    expand = c(0, 0, 0, 0),      # No extra white space
    vwidth = 1000,               # Set width to match the table (adjust this based on your table size)
    vheight = 600                # Set height to match the table (adjust this based on your table size)
  )
  
  
  plots_path
  
  ###COUNTERFACTUAL ANALYSIS FOR GHANA####
  
  gh <- df
  
  
  gh <- subset(gh, (gh$year == 2018 & gh$country_name == "Ghana"))
  
  gh$brd_12mb = 0
  
  #Ghana vs full dataset comparison
  
  # Percentiles to calculate
  percentiles <- c(0.5, 0.9, 0.95, 0.99)
  
  # Initialize a list to store results
  results <- list()
  
  # Loop through each variable and calculate the specified percentiles
  for(variable in pol2) {
    # Extract the column data from the dataframe
    column_data <- test[[variable]]
    
    # Calculate the percentiles for the current variable
    percentile_values <- quantile(column_data, percentiles, na.rm = TRUE)
    
    # Store the results in the list
    results[[variable]] <- percentile_values
  }
  
  # Print or return the results
  print(results)
  
  # Create a copy of df
  
  df_gh <- df
  train_gh <- train
  test_gh <- test
  
  
  # Replace values in df_gh with values from gh for the year 2018 for the specified variables
  for (var in pol) {
    train_gh[, var] <- gh[, var]
    test_gh[, var] <- gh[, var]
    df_gh[, var] <- gh[, var]
  }
  
  ###FIGURE
  
  x_limits90 = c(0,200)
  y_limits = NULL
  
  
  q10 <- function(y) quantile(y, 0.1)
  q90 <- function(y) quantile(y, 0.9)
  q25 <- function(y) quantile(y, 0.025)
  q975 <- function(y) quantile(y, 0.975)
  
  
  ##############All political indicators################
  
  #load models
  fit_all <- readRDS("results/panel/aggregate/bayes_models/all_model.rds")
  
  data_all_test <- ppc_stat_data(y = test$dead_w, yrep = posterior_predict(fit_all, newdata = test, allow_new_levels = TRUE), stat = "q90")  
  data_all_gh_test <- ppc_stat_data(y = test_gh$dead_w, yrep = posterior_predict(fit_all, newdata = test_gh, allow_new_levels = TRUE), stat = "q90") 
  
  binwidth_defined <- 3 
  
  #Start the ggplot call
  p <- ggplot() + theme_minimal()
  
  
  #Function to plot histogram
  
  add_dataset_layers <- function(p, data, dataset_name, binwidth_defined) {
    p + 
      geom_histogram(data = filter(data, variable != "y"), #Assuming 'variable' is the correct column name
                     aes(x = value, fill = dataset_name), 
                     color = "black", linewidth = 0.25, na.rm = TRUE, 
                     binwidth = binwidth_defined) # Apply alpha here
  }
  
  
  p <- add_dataset_layers(p, data_all_test, "All, actual", binwidth_defined)
  p <- add_dataset_layers(p, data_all_gh_test, "All, Ghana", binwidth_defined)
  
  p <- p + 
    scale_fill_manual(values = c("All, actual" = alpha("#F44B3E", 0.7), 
                                 "All, Ghana" = alpha("#2596be", 0.7))) +
    coord_cartesian(xlim = c(NA, 150)) # Limit x-axis range
  
  
  
  # Add labels and adjust theme
  p <- p + labs( x = "Flood deaths", y = "Predicted frequency") + 
    theme(legend.title = element_blank(),
          legend.position = "none",
          plot.title = element_text(size = 14, color = "black", family="Arial", hjust = 0.5),
          axis.text = element_text(size = 12, color = "black", family="Arial"),
          legend.text = element_text(size = 12, color = "black", family="Arial"), 
          axis.title =  element_text(size = 12, color = "black", family="Arial")
          
    )
  # Print the plot
  print(p)
  
  ggsave(paste0(plots_path, "all_counterfactual_outsample_Ghana.png"), height = 5, width = 5)
  
  
  ###SINGLE MODELS; GLOBAL SAMPLE
  
  # Figure 5b
  
  # Function to load model and generate predictions
  generate_predictions <- function(model_path, test_data, test_gh_data, SEEDNUM) {
    # Set the seed for reproducibility
    set.seed(SEEDNUM)
    
    fit <- readRDS(model_path)
    predictions <- posterior_predict(fit, newdata = test, allow_new_levels = TRUE)
    predictions_gh <- posterior_predict(fit, newdata = test_gh, allow_new_levels = TRUE)
    list(predictions = predictions, predictions_gh = predictions_gh)
  }
  
  compute_summary_stats <- function(predictions, predictions_gh, model_name) {
    # Calculate the means for each set of predictions
    means_test <- colMeans(predictions)
    means_test_gh <- colMeans(predictions_gh)
    
    # Calculate the average of the means and round to 2 decimal places
    avg_means_test <- round(mean(means_test), 2)
    avg_means_test_gh <- round(mean(means_test_gh), 2)
    
    # Calculate the percentage change and round to 2 decimal places
    percentage_change <- round((avg_means_test_gh - avg_means_test) / avg_means_test * 100, 0)
    
    # Calculate the range for the average predictions and round to 2 decimal places
    min_avg_pred <- round(min(means_test) - min(means_test_gh), 2)
    max_avg_pred <- round(max(means_test) - max(means_test_gh), 2)
    
    min_avg_pred<- format(min_avg_pred, big.mark=",",scientific=FALSE)
    max_avg_pred <- format(max_avg_pred, big.mark=",",scientific=FALSE)
    
    # Return a data frame with the summary statistics
    data.frame(
      Model = model_name,
      Avg_Pred_Test = avg_means_test,
      Avg_Pred_Test_gh = avg_means_test_gh,
      Percentage_Change = percentage_change,
      Range_Difference = paste0(min_avg_pred, " – ", max_avg_pred)
    )
  }
  
  if(!GDP){print("skip!")}else{
    
    # Define model names and paths (make sure to replace these with actual paths)
    model_names <- c("Accountability", "Inclusion", "Gov. effectiveness", "Rule of law", "Conflict history", "Local conflict", "All pol. predictors")
    model_paths <- c(
      paste0(models_path, "accountability_model.rds"),
      paste0(models_path, "inclusion_model.rds"),
      paste0(models_path, "goveff_model.rds"),
      paste0(models_path, "ruleoflaw_model.rds"),
      paste0(models_path, "conflict_country_model.rds"),
      paste0(models_path, "conflict_sub_model.rds"),
      "results/panel/aggregate/bayes_models/all_model.rds")
    
    
    # Generate predictions and compute summary statistics for each model
    
    
    model_summaries <- list()
    
    for (i in 1:length(model_names)) {
      pred <- generate_predictions(model_paths[i], test, test_gh, SEEDNUM)
      model_summaries[[model_names[i]]] <- compute_summary_stats(pred$predictions, pred$predictions_gh, model_names[i])
    }
    
    # Combine all model summaries into a single data frame
    all_model_summaries <- do.call(rbind, model_summaries)
    
    
    # Assuming all_model_summaries is your data frame
    
    # Create a gt table from the combined summary
    gt_table <- gt(all_model_summaries) %>%
      fmt_number(
        columns = c(Avg_Pred_Test, Avg_Pred_Test_gh),
        decimals = 2,
        use_seps = TRUE  # This will add the comma for thousands separator
      ) %>%
      fmt_number(
        columns = c(Percentage_Change),
        decimals = 0,  # Round to 0 decimal places for the percentage change
        pattern = "{x}%"  # Add the percentage sign
      ) %>%
      cols_label(
        Model = "Model",
        Avg_Pred_Test = html("y&#770;<sub>obs</sub>"),  # Using HTML entities for hat and subscript
        Avg_Pred_Test_gh = html("y&#770;<sub>cnt</sub>"),
        Percentage_Change = html("&Delta;y&#770;"),
        Range_Difference = html("&Delta;y&#770; (range)")
      ) %>% 
      tab_options(table.font.names = 'Arial', table.font.color = "black") 
    # Save the gt table to an HTML file
    gtsave(gt_table, filename = paste0(tables_path, "model_summaries_Ghana.html"))
    gtsave(gt_table, paste0(plots_path, "model_summaries_Ghana.png"))
    
  }
}


###########################################################
##################ALTERNATIVE METRICS######################
###########################################################

if(GDP){
  # Compute WAIC for each model
  waic_in_sample <- map(models, waic)
  waic_out_sample <- map(models, ~ waic(.x, newdata = test, allow_new_levels = TRUE))
  # Format WAIC results (round to 2 digits for the table)
  waic_in_sample_formatted <- map_dbl(waic_in_sample, ~ round(.x$estimates['waic', 'Estimate'], 2))
  waic_out_sample_formatted <- map_dbl(waic_out_sample, ~ round(.x$estimates['waic', 'Estimate'], 2))
  
  
  # Combine WAIC and JS performance metrics into a tibble
  predictive_performance_table_waic <- tibble(
    mname = names(models),
    waic_in_sample = waic_in_sample_formatted,
    waic_out_sample = waic_out_sample_formatted
  )
  
  # Sort by in-sample WAIC (ascending order, smaller is better)
  predictive_performance_table_waic <- predictive_performance_table_waic %>%
    arrange(waic_in_sample) #lower better -- top model should have small WAIC
  
  
  # Create the final gt table with proper headers for WAIC
  gt_table_waic <- gt(predictive_performance_table_waic) %>%
    tab_spanner(label = "In-sample (2000-2014)", 
                columns = c("waic_in_sample")) %>%
    tab_spanner(label = "Out-of-sample (2015-2018)",
                columns = c("waic_out_sample")) %>%
    cols_label(mname = "Model",
               waic_in_sample = "WAIC",
               waic_out_sample = "WAIC") %>%
    tab_options(table.font.names = 'Arial', table.font.color = "black") %>%
    tab_style(
      style = list(
        cell_borders(sides = "left", color = "lightgray", weight = px(2))
      ),
      locations = cells_body(columns = c(waic_out_sample))
    ) %>%
    tab_style(
      style = list(
        cell_borders(sides = "left", color = "lightgray", weight = px(2))
      ),
      locations = cells_column_labels(columns = c(waic_out_sample))
    ) %>%
    gtsave(paste0(tables_path, "/predictive_performance_waic.tex"))
  
}


