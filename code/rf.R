#### This code replicates the analysis in Vesco et al.
#### Produced by P. Vesco, last updated July 3, 2025

#The script runs the Random Forests models for the sensitivity analysis in Supplementary

rm(list = ls(all = TRUE))
library(iml)
library(rstan)
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
TRAIN_MODELS <- TRUE # To fit the models set this to TRUE
GDP <-  TRUE ## As it is set up currently, the script runs the analysis for the MAIN SPECIFICATION presented in the manuscript

###To run the tests presented in the Supplementary Material, you need to set the appropriate test to TRUE 

AFFECTED <- FALSE
AGG <-   FALSE
CUTOFF <- FALSE
DEAD <- FALSE
ECONTEST <- FALSE
FE <- FALSE
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
getwd()
setwd("political_development_flood")

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
  v_baseline <- c("dfo_severity", "duration", "nevents_sum10", "population_affected", "rugged", "hdi_l1", "wdi_gdppc", "tropical_flood", "tropical_flood_dummy")
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

library(ranger)

train$continent <- factor(train$continent)
test$continent <- factor(test$continent, levels = levels(train$continent))

train$continent_nevents <- as.numeric(as.factor(train$continent)) * train$nevents_sum10
test$continent_nevents <- as.numeric(as.factor(test$continent)) * test$nevents_sum10

#####################################

if(TRAIN_MODELS){
baseline <- ranger(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged  + hdi_l1 + wdi_gdppc + tropical_flood_dummy  + continent + continent_nevents + year,
                         data = train, num.trees = 500, mtry = 4, importance = "permutation", seed = SEEDNUM, respect.unordered.factors = "partition")
conflict_country <- ranger(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged  + hdi_l1 + wdi_gdppc + tropical_flood_dummy  + continent + continent_nevents + year + decay_brds_c,
                                 data = train, num.trees = 500, mtry = 4, importance = "permutation", seed = SEEDNUM, respect.unordered.factors = "partition")
conflict_sub <- ranger(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged  + hdi_l1 + wdi_gdppc + tropical_flood_dummy  + continent + continent_nevents + year + brd_12mb,
                             data = train, num.trees = 500, mtry = 4, importance = "permutation", seed = SEEDNUM, respect.unordered.factors = "partition")
accountability <- ranger(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged  + hdi_l1 + wdi_gdppc + tropical_flood_dummy  + continent  + continent_nevents + year + e_wbgi_vae,
                               data = train, num.trees = 500, mtry = 4, importance = "permutation", seed = SEEDNUM, respect.unordered.factors = "partition")
goveff <- ranger(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged  + hdi_l1 + wdi_gdppc + tropical_flood_dummy  + e_wbgi_gee + continent_nevents + year,
                       data = train, num.trees = 500, mtry = 4, importance = "permutation", seed = SEEDNUM, respect.unordered.factors = "partition")
inclusion <- ranger(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged  + hdi_l1 + wdi_gdppc + tropical_flood_dummy  + continent  + continent_nevents + year + v2xpe_exlsocgr,
                          data = train, num.trees = 500, mtry = 4, importance = "permutation", seed = SEEDNUM, respect.unordered.factors = "partition")
ruleoflaw <- ranger(dead_w ~ dfo_severity + duration + nevents_sum10 + population_affected + rugged  + hdi_l1 + wdi_gdppc + tropical_flood_dummy + v2x_rule + continent  + continent_nevents + year,
                          data = train, num.trees = 500, mtry = 4, importance = "permutation", seed = SEEDNUM, respect.unordered.factors = "partition")


# Save the trained models
saveRDS(baseline, paste0(models_path, "baseline_model.rds"))
saveRDS(conflict_country, paste0(models_path, "conflict_country_model.rds"))
saveRDS(conflict_sub, paste0(models_path, "conflict_sub_model.rds"))
saveRDS(accountability, paste0(models_path, "accountability_model.rds"))
saveRDS(goveff, paste0(models_path, "goveff_model.rds"))
saveRDS(inclusion, paste0(models_path, "inclusion_model.rds"))
saveRDS(ruleoflaw, paste0(models_path, "ruleoflaw_model.rds"))
}

fit_baseline <- readRDS(paste0(models_path, "baseline_model.rds"))
fit_conflict_country <- readRDS(paste0(models_path, "conflict_country_model.rds"))
fit_conflict_sub <- readRDS(paste0(models_path, "conflict_sub_model.rds"))
fit_accountability <- readRDS(paste0(models_path, "accountability_model.rds"))
fit_goveff <- readRDS(paste0(models_path, "goveff_model.rds"))
fit_inclusion <- readRDS(paste0(models_path, "inclusion_model.rds"))
fit_ruleoflaw <- readRDS(paste0(models_path, "ruleoflaw_model.rds"))

#combine all models
# Create a list of all models
all_models <- list(
  fit_baseline,
  fit_conflict_country,
  fit_conflict_sub,
  fit_accountability,
  fit_goveff,
  fit_inclusion,
  fit_ruleoflaw)

names(all_models) <- c(
  "Baseline",
  "Conflict history",
  "Local conflict",
  "Accountability",
  "Gov. effectiveness",
  "Inclusion",
  "Rule of law"
)


# Now, create a tibble that contains all  models
fit_df <- tibble(
  fits = all_models,
  mname = names(all_models))

# Define the baseline and political feature sets
baseline_features <- c("dfo_severity", "duration", "nevents_sum10", "population_affected", "rugged", "tropical_flood_dummy",  "hdi_l1", "wdi_gdppc","year","continent", "continent_nevents")

# Map model names to their corresponding political features
political_feature_map <- list(
  "Conflict history" = "decay_brds_c",
  "Local conflict" = "brd_12mb",
  "Accountability" = "e_wbgi_vae",
  "Gov. effectiveness" = "e_wbgi_gee",
  "Inclusion" = "v2xpe_exlsocgr",
  "Rule of law" = "v2x_rule"
)

predict_rf <- function(model, newdata) {
  predict(model, data = newdata)$predictions
}


# Create an empty list to store global SHAP values
global_shap_list <- list()

# Iterate over each model to compute SHAP values
for (model_name in names(political_feature_map)) {
  
  # Get the political feature for the current model
  political_feature <- political_feature_map[[model_name]]
  
  # Get the current model from your list of models
  current_model <- all_models[[model_name]]
  
  # Combine baseline and political features for the current model
  all_features <- c(baseline_features, political_feature)
  
  # Subset the test data to the relevant features
  test_data <- test[, all_features]
  
  # Create the Predictor object for iml
  predictor <- Predictor$new(
    model = current_model,
    data = test_data,
    y = test$dead_w,
    predict.fun = predict_rf
  )
  
  
  # Compute the SHAP values
  shapley <- Shapley$new(predictor, x.interest = test_data)
  
  # Extract SHAP values (ensure they are numeric)
  shap_values <- shapley$results
  shap_values$phi <- as.numeric(shap_values$phi)  # Ensure SHAP values are numeric
  
  # Compute global importance as the mean absolute SHAP values
  global_shap_importance <- aggregate(abs(shap_values$phi), by = list(shap_values$feature), FUN = mean)
  colnames(global_shap_importance) <- c("feature", "importance")
  
  # Store the global SHAP values
  global_shap_list[[model_name]] <- global_shap_importance
}

colors <- c(
  'Accountability' = "#8eade8",
  'Inclusion' = "#3c61a3",
  'Gov. effectiveness' = "#f2aed8",
  'Rule of law' = "#c556d1",
  'Conflict history' = "#f5b342",
  'Local conflict' = "#f0843c", 
  'Baseline' = "#CCCCCC", 
  'dfo_severity' = "#CCCCCC",
  'duration' = "#CCCCCC",
  'nevents_sum10' = "#CCCCCC",
  'rugged' = "#CCCCCC",  # Not in plot, but for completeness
  'tropical_flood' = "#CCCCCC",  # Not in plot, but for completeness
  "wdi_gdppc" = "#CCCCCC",
  "hdi_l1"= "#CCCCCC")

# Combine all the global SHAP importances into one data frame for plotting
global_shap_df <- bind_rows(global_shap_list, .id = "Model")

# Rename features for better understanding
global_shap_df <- global_shap_df %>%
  mutate(feature = case_when(
    feature == "e_wbgi_vae" ~ "Accountability",
    feature == "v2xpe_exlsocgr" ~ "Inclusion",
    feature == "e_wbgi_gee" ~ "Gov. effectiveness",
    feature == "v2x_rule" ~ "Rule of law",
    feature == "decay_brds_c" ~ "Conflict history",
    feature == "brd_12mb" ~ "Local conflict",
    TRUE ~ feature  # Keep original name if no match
  ))


# Compute baseline averages
baseline_shap_df <- global_shap_df %>%
  filter(feature %in% baseline_features) %>%
  group_by(Model) %>%
  summarise(importance = mean(importance), feature = "Baseline")

# Extract political SHAP values
political_shap_df <- global_shap_df %>%
  filter(feature %in% names(political_feature_map))

# Combine baseline and political data
combined_shap_df <- political_shap_df %>%
  bind_rows(baseline_shap_df) %>%
  mutate(bar_type = ifelse(feature == "Baseline", "Baseline", "Political"))

# For ordering, get political importance sorted descending
ordering_df <- political_shap_df %>%
  arrange(desc(importance)) %>%
  mutate(Model = factor(Model, levels = rev(unique(Model)))) %>%
  dplyr::select(Model, importance_order = importance)

combined_shap_df <- combined_shap_df %>%
  left_join(ordering_df, by = "Model") %>%
  arrange(importance_order, bar_type) %>%
  mutate(
    Model = factor(Model, levels = rev(unique(ordering_df$Model))),
    bar_type = factor(bar_type, levels = c("Baseline", "Political"))
  )

# Set the vertical offset for baseline bars
combined_shap_df <- combined_shap_df %>%
  mutate(bar_position = as.numeric(Model) + ifelse(bar_type == "Baseline", -0.2, 0.2))

# Your provided colors
colors <- c(
  'Accountability' = "#8eade8",
  'Inclusion' = "#3c61a3",
  'Gov. effectiveness' = "#f2aed8",
  'Rule of law' = "#c556d1",
  'Conflict history' = "#f5b342",
  'Local conflict' = "#f0843c",
  'Baseline' = "#CCCCCC"
)

# Final corrected plot
shap <- ggplot(combined_shap_df, aes(y = importance, x = bar_position, fill = feature)) +
  geom_bar(stat = "identity", width = 0.35) +
  scale_fill_manual(values = colors) +
  scale_x_continuous(
    breaks = seq_along(levels(combined_shap_df$Model)),
    labels = levels(combined_shap_df$Model)
  ) +
  coord_flip() +
  labs(y = "Mean SHAP value", x = "") +
  theme_bw() +
  theme(
    legend.position = "none",
    axis.text = element_text(size = 16, color = "black"),
    axis.title = element_text(size = 18, color = "black")
  )

shap

# Save your plot
ggsave(filename = paste0(plots_path, "/shapley.png"), plot = shap, width = 10, height = 6)

