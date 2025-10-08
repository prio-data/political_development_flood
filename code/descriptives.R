#### This code produces the analytics and descriptives for the paper Vesco et al.
#### Produced by P. Vesco, last updated July 3, 2025.


rm(list = ls(all = TRUE))
library(gpboost)
library(utils)
#library(assertive) 
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
library(bayesplot)
library(brms)
library(bayestestR)
library(loo)
library(knitr)
library(patchwork)
library(countrycode)
library(tidybayes)
library(mice)
library(gt)
library(scales)
library(sjPlot)
library(ggmap)
library(tidyverse)
library(lubridate)
library(RColorBrewer)
library(rnaturalearth)
library(rnaturalearth)
library(rnaturalearthdata)
library(sf)

setwd("political_development_flood")

#Create folders to save Figures
results_path <- paste0(getwd(),"/","results")
descriptives_path <- paste0(results_path,"/descriptives/")


dir.create(results_path)
dir.create(descriptives_path)


#define variables
v_baseline <- c("dfo_severity", "duration", "nevents_sum10", "population_affected", "rugged", "tropical_flood", "hdi_l1", "wdi_gdppc")
v_conflict <- paste(c(v_baseline, "brd_12mb", "decay_brds_c"))
v_exclusion <- paste(c(v_baseline, "v2xpe_exlsocgr")) #flood_excluded_dummy
v_corruption <- paste(c(v_baseline, "v2xnp_regcorr"))
v_accountability <- paste(c(v_baseline, "e_wbgi_vae"))
v_goveff <- paste(c(v_baseline, "e_wbgi_gee"))
v_ruleoflaw <- paste(c(v_baseline, "v2x_rule")) 
v_regime <- paste(c(v_baseline, "v2x_polyarchy"))


preds <- c("e_wbgi_vae", "v2xpe_exlsocgr", "e_wbgi_gee", "v2x_rule",  "decay_brds_c", "brd_12mb",   "v2x_polyarchy", v_baseline)  ##flood_excluded_dummy


all <- c("dead_w", preds)




##################################
######### DESCRIPTIVES############
##################################


###Map of flood exposure and mortality, Figure S4.


#Mapping floods


df <- readRDS("data/data_final.rds")

##Map flood events
# Load world map with rnaturalearth
world <-  ne_countries(scale = "medium", returnclass = "sf")

# Prepare your data as before
df_country <- df %>% group_by(iso3c) %>% summarise(dead_w = mean(dead_w, na.rm = TRUE))


###Map with total deaths per capita using pop in 2010 as reference
#We need data on population in 2010, from wdi -- stored in imputed version of the data


imputed <- read_rds("data/imputed.rds")


df_pop <- subset(df, df$population_affected > 0)
train <- subset(df_pop, df_pop$year <= 2014)
test <- subset(df_pop, df_pop$year >= 2015)




pop2010 <- imputed %>% dplyr::select(iso3c, year, pop_wdi)
pop2010  <- pop2010 %>% filter(year == 2010)


# Joining population data
df_pc <- df %>% dplyr::select(iso3c, dead_w) %>%
  left_join(pop2010, by = "iso3c")


#view(df_pc)


# Joining population data
train_pc <- train %>% dplyr::select(iso3c, dead_w) %>%
  left_join(pop2010, by = "iso3c")

sum(is.na(train_pc$dead_pc))

test_pc <- test %>% dplyr::select(iso3c, dead_w) %>%
  left_join(pop2010, by = "iso3c")
sum(is.na(test_pc$dead_pc))

# Cleanup and aggregation
df_pc <- df_pc %>%
  dplyr::select(iso3c, year, dead_w, pop_wdi) %>%
  group_by(iso3c) %>%
  summarise(dead_tot = sum(dead_w),   
            dead_pc = (dead_tot/pop_wdi) * 1e6) %>% 
  distinct()




# Cleanup and aggregation
train_pc <- train_pc %>%
  dplyr::select(iso3c, year, dead_w, pop_wdi) %>%
  group_by(iso3c) %>%
  summarise(dead_tot = sum(dead_w),   
            dead_pc = (dead_tot/pop_wdi) * 1e6) %>% 
  distinct()


# Cleanup and aggregation
test_pc <- test_pc %>%
  dplyr::select(iso3c, year, dead_w, pop_wdi) %>%
  group_by(iso3c) %>%
  summarise(dead_tot = sum(dead_w),   
            dead_pc = (dead_tot/pop_wdi) * 1e6) %>% 
  distinct()




# per million people


world_joined <- left_join(world, df_pc, by = c('iso_a3_eh' = 'iso3c'))  # Change 'iso_a2' to the correct column name if different
world_joined_train <- left_join(world, train_pc, by = c('iso_a3_eh' = 'iso3c'))  # Change 'iso_a2' to the correct column name if 
world_joined_test <- left_join(world, test_pc, by = c('iso_a3_eh' = 'iso3c'))  # Change 'iso_a2' to the correct column name if 

summary(world_joined$dead_pc)
summary(world_joined_train$dead_pc)
summary(world_joined_test$dead_pc)


world_joined <- world_joined %>% mutate(dead_pc = case_when(dead_pc <= 5 ~ "0-5",
                                                            dead_pc > 5 & dead_pc <= 10 ~ "6-10",
                                                            dead_pc > 10 & dead_pc <= 100 ~ "11-100",
                                                            dead_pc > 100 & dead_pc <= 1000 ~ "101-1000",
                                                            dead_pc > 1000 & dead_pc <= 10000 ~ "1001-10,000"))


world_joined$dead_pc <- ordered(world_joined$dead_pc, levels = c("0-5", "6-10", "11-100","101-1000","1001-10,000"))

world_joined_train <- world_joined_train %>% mutate(dead_pc = case_when(dead_pc <= 5 ~ "0-5",
                                                                        dead_pc > 5 & dead_pc <= 10 ~ "6-10",
                                                                        dead_pc > 10 & dead_pc <= 100 ~ "11-100",
                                                                        dead_pc > 100 & dead_pc <= 1000 ~ "101-1000",
                                                                        dead_pc > 1000 & dead_pc <= 10000 ~ "1001-10,000"))


world_joined_train$dead_pc <- ordered(world_joined_train$dead_pc, levels = c("0-5", "6-10", "11-100","101-1000","1001-10,000"))


world_joined_test <- world_joined_test %>% mutate(dead_pc = case_when(dead_pc <= 5 ~ "0-5",
                                                                      dead_pc > 5 & dead_pc <= 10 ~ "6-10",
                                                                      dead_pc > 10 & dead_pc <= 100 ~ "11-100",
                                                                      dead_pc > 100 & dead_pc <= 1000 ~ "101-1000",
                                                                      dead_pc > 1000 & dead_pc <= 10000 ~ "1001-10,000"))


world_joined_test$dead_pc <- ordered(world_joined_test$dead_pc, levels = c("0-5", "6-10", "11-100","101-1000","1001-10,000"))




# Set up the color palette
model_colors <- c('#fcfdbf', '#fca50a', '#dd513a', '#932667', '#420a68')




# Read the flood event centroids 


gfd <- readRDS("data/gfd_centroids.rds")


gfd_train <- subset(gfd, as.numeric(year(gfd$began)) <= 2014)
gfd_test <- subset(gfd, as.numeric(year(gfd$began)) > 2014)


g_map <- ggplot(data = world_joined) +
  geom_sf(stat = "identity", color = "black", lwd = 0.1, aes(fill = dead_pc), alpha = 0.6) +
  geom_point(data = gfd, aes(x = lon, y = lat), color = "black", size = 0.5, alpha = 0.5) +
  scale_fill_manual(glue::glue("Flood deaths \n per million people"), values = model_colors, na.value = 'lightgrey') +
  theme_minimal() +
  labs(x = "Longitude", y = "Latitude") +
  theme(legend.position = c(0.08, 0.33),
        legend.box.background = element_rect(color = "black"),
        legend.title = element_text(size = 8),
        legend.text = element_text(size = 8)) +
  coord_sf(expand = FALSE)

g_map <- g_map  + 
  coord_sf(xlim = c(-180, 180), ylim = c(-90, 90), expand = FALSE)

g_map

ggsave(paste0("results/descriptives/Fig2.png"), plot = g_map, device="png", width = 22, height = 10.5, units = "cm")


#train

g_map <- ggplot(data = world_joined_train) +
  geom_sf(stat = "identity", color = "black", lwd = 0.1, aes(fill = dead_pc), alpha = 0.6) +
  geom_point(data = gfd_train, aes(x = lon, y = lat), color = "black", size = 0.5, alpha = 0.5) +
  scale_fill_manual(glue::glue("Flood deaths \n per million people"), values = model_colors, na.value = 'lightgrey') +
  theme_minimal() +
  labs(x = "Longitude", y = "Latitude") +
  theme(legend.position = c(0.08, 0.33),
        legend.box.background = element_rect(color = "black"),
        legend.title = element_text(size = 8),
        legend.text = element_text(size = 8)) +
  coord_sf(expand = FALSE)

g_map <- g_map  + 
  coord_sf(xlim = c(-180, 180), ylim = c(-90, 90), expand = FALSE)

g_map



ggsave(paste0("results/descriptives/map_flood_pc_inferno_train.png"), plot = g_map, device="png", width = 22, height = 10.5, units = "cm")



# Read the flood event centroids 

g_map <- ggplot(data = world_joined_test) +
  geom_sf(stat = "identity", color = "black", lwd = 0.1, aes(fill = dead_pc), alpha = 0.6) +
  geom_point(data = gfd_test, aes(x = lon, y = lat), color = "black", size = 0.5, alpha = 0.5) +
  scale_fill_manual(glue::glue("Flood deaths \n per million people"), values = model_colors, na.value = 'lightgrey') +
  theme_minimal() +
  labs(x = "Longitude", y = "Latitude") +
  theme(legend.position = c(0.08, 0.33),
        legend.box.background = element_rect(color = "black"),
        legend.title = element_text(size = 8),
        legend.text = element_text(size = 8)) +
  coord_sf(expand = FALSE)

g_map <- g_map  + 
  coord_sf(xlim = c(-180, 180), ylim = c(-90, 90), expand = FALSE)

g_map

ggsave(paste0("results/descriptives/map_flood_pc_inferno_test.png"), plot = g_map, device="png", width = 22, height = 10.5, units = "cm")


### Figure S1.
df <- readRDS("data/data_final.rds")
df <- subset(df, df$population_affected > 0)
train <- subset(df, df$year <= 2014)
test <- subset(df, df$year >= 2015)

# Training data plot with individual points and N labels
d_train <- ggplot(train, aes(x = log1p(dead_w))) +
  geom_histogram(aes(y = ..count../sum(..count..)), 
                 fill = "#B0E0E6", color = "steelblue4", size = 0.2) + 
  geom_text(stat = "bin", aes(y = ..count../sum(..count..), 
                              label = ifelse(..count.. > 0, ..count.., "")),
            vjust = -1.2, size = 0.8, family = "Arial", angle = 45) +  # More space above bar
  geom_jitter(aes(y = 0.005), alpha = 0.6, size = 0.3, 
              height = 0.003, color = "steelblue4") +
  scale_y_continuous(limits = c(0, 0.85), labels = scales::percent) +
  scale_x_continuous(breaks = c(0, 2, 4, 6, 8, 10, 12)) +
  theme_minimal() +  # Changed to minimal theme
  labs(title = "", 
       x = "Flood mortality (logged)", 
       y = "Percentage") +
  theme(
    plot.title = element_text(size = 4, color = "black", family = "Arial", hjust = 0.5),
    axis.text = element_text(size = 4, color = "black", family = "Arial"),
    legend.text = element_text(size = 4, color = "black", family = "Arial"), 
    axis.title = element_text(size = 4, color = "black", family = "Arial"),
    panel.grid.major.x = element_blank(),  # Remove vertical major gridlines
    panel.grid.major.y = element_line(color = "white", size = 0.2),  # Keep horizontal gridlines white
    panel.grid.minor = element_blank())  # Remove all minor gridlines

# Test data plot with individual points and N labels
d_test <- ggplot(test, aes(x = log1p(dead_w))) +
  geom_histogram(aes(y = ..count../sum(..count..)), 
                 fill = "#BCE3C5", color = "#02704A", size = 0.2) +
  geom_text(stat = "bin", aes(y = ..count../sum(..count..), 
                              label = ifelse(..count.. > 0, ..count.., "")),
            vjust = -1.2, size = 0.8, family = "Arial", angle = 45) +  # More space above bar
  geom_jitter(aes(y = 0.005), alpha = 0.6, size = 0.3, 
              height = 0.003, color = "#02704A") +
  theme_minimal() +  # Changed to minimal theme
  scale_y_continuous(limits = c(0, 0.85), labels = scales::percent) +
  scale_x_continuous(breaks = c(0, 1, 2, 3, 4, 5, 6)) +
  labs(title = "", 
       x = "Flood mortality (logged)", 
       y = "Percentage") +
  theme(
    plot.title = element_text(size = 4, color = "black", family = "Arial", hjust = 0.5),
    axis.text = element_text(size = 4, color = "black", family = "Arial"),
    legend.text = element_text(size = 4, color = "black", family = "Arial"), 
    axis.title = element_text(size = 4, color = "black", family = "Arial"),
    panel.grid.major.x = element_blank(),  # Remove vertical major gridlines
    panel.grid.major.y = element_line(color = "white", size = 0.2),  # Keep horizontal gridlines white
    panel.grid.minor = element_blank())  # Remove all minor gridlines

# Save with better resolution settings
ggsave("results/descriptives/hist_train.png", plot = d_train, 
       device = "png", width = 4.5, height = 4, dpi = 600, units = "cm")
ggsave("results/descriptives/hist_test.png", plot = d_test, 
       device = "png", width = 4.5, height = 4, dpi = 600, units = "cm")

#Descriptive statistics: Tables S1-S2


library(broom)
library(gt)
library(kableExtra)


##Descriptives for training set and test set


df <- readRDS('data/data_final.rds')


df$gwcode <- factor(df$gwcode)
df$continent <- factor(df$continent)




log_vars <- c("duration", "population_affected" , "wdi_gdppc", "brd_12mb", "decay_brds_c", "nevents_sum10", "hdi_l1")



df$v2xpe_exlsocgr <- 1 - df$v2xpe_exlsocgr

df <-  subset(df, df$population_affected > 0)
df <- df %>%   
  mutate(across(all_of(log_vars), .fns = function(x) log(x+1)))
df$dead_w <- as.integer(df$dead_w)




train <- df %>% dplyr::filter(year >= 2000 & year <= 2014) %>% dplyr::select(all)
test <- df %>% dplyr::filter(year >= 2015) %>% dplyr::select(all)


to_descriptive_stat <- train %>%
  dplyr::select(paste(c(all,'gwcode'))) %>%
  recipe(.) %>%
  prep() %>%
  juice()


descriptives <- to_descriptive_stat %>% skim()




main_table <- descriptives %>% as_tibble() %>% janitor::clean_names() %>% 
  dplyr::select(skim_variable, n_missing,
                numeric_mean, numeric_sd, 
                numeric_p50, numeric_p0, numeric_p100) %>%
  rename(Variable = skim_variable,
         Mean = numeric_mean,
         SD = numeric_sd,
         Median = numeric_p50,
         Min = numeric_p0,
         Max = numeric_p100,
         Missing = n_missing) %>% filter(Variable != "country")




main_table <- main_table[match(all, main_table$Variable),]


main_table[1] <- c('Flood mortality', 'Accountability', 'Inclusion', 'Gov. effectiveness', 'Rule of law',  'Conflict history','Local conflict', 'Electoral Democracy', 'Flood severity', 'Flood duration', 'Past flood events', 'Exposed population', 'Rough terrain', 'Tropical flood dummy', 'Local HDI', 'National GDP pc')
ncountries <- descriptives  %>% as_tibble() %>% filter(skim_variable == "gwcode") %>% pull(factor.n_unique)
nobs <- attributes(descriptives)$data_rows


command <- paste("<tfoot><tr><td colspan='7'>The dataset has", 
                 nobs,"observations and", 
                 ncountries, "countries.</td></tr></tfoot>", sep = " ")


footer <- list("pos" = list(nrow(main_table)), "command" = command)


# Create in print xtable in html
print(xtable::xtable(main_table, 
                     caption="Descriptive statistics for the training set",
                     digits = 2),
      type = "html",
      file = paste0("results/descriptives/descriptive_statistics_train.html"),
      include.rownames=FALSE, caption.placement='top',
      html.table.attributes='align="left"',
      add.to.row = footer)






to_descriptive_stat <- test %>%
  dplyr::select(paste(c(all,'gwcode'))) %>%
  recipe(.) %>%
  prep() %>%
  juice()


descriptives <- to_descriptive_stat %>% skim()




main_table <- descriptives %>% as_tibble() %>% janitor::clean_names() %>% 
  dplyr::select(skim_variable, n_missing,
                numeric_mean, numeric_sd, 
                numeric_p50, numeric_p0, numeric_p100) %>%
  rename(Variable = skim_variable,
         Mean = numeric_mean,
         SD = numeric_sd,
         Median = numeric_p50,
         Min = numeric_p0,
         Max = numeric_p100,
         Missing = n_missing) %>% filter(Variable != "country")




main_table <- main_table[match(all, main_table$Variable),]


main_table[1] <- c('Flood mortality', 'Accountability', 'Inclusion', 'Gov. effectiveness', 'Rule of law',  'Conflict history','Local conflict', 'Electoral Democracy', 'Flood severity', 'Flood duration', 'Past flood events', 'Exposed population', 'Rough terrain', 'Tropical flood dummy', 'Local HDI', 'National GDP pc')
ncountries <- descriptives  %>% as_tibble() %>% filter(skim_variable == "gwcode") %>% pull(factor.n_unique)
nobs <- attributes(descriptives)$data_rows


command <- paste("<tfoot><tr><td colspan='7'>The dataset has", 
                 nobs,"observations and", 
                 ncountries, "countries.</td></tr></tfoot>", sep = " ")


footer <- list("pos" = list(nrow(main_table)), "command" = command)


# Create in print xtable in html
print(xtable::xtable(main_table, 
                     caption="Descriptive statistics for the test set",
                     digits = 2),
      type = "html",
      file = paste0("results/descriptives/descriptive_statistics_test.html"),
      include.rownames=FALSE, caption.placement='top',
      html.table.attributes='align="left"',
      add.to.row = footer)




####Correlation plot Figure S2


df <- readRDS('data/data_final.rds')
df$v2xpe_exlsocgr <- 1 - df$v2xpe_exlsocgr


df$gwcode <- factor(df$gwcode)
df$continent <- factor(df$continent)




log_vars <- c("duration", "population_affected" , "wdi_gdppc", "brd_12mb", "decay_brds_c", "nevents_sum10", "hdi_l1")




df <-  subset(df, df$population_affected > 0)


df <- df %>%   
  mutate(across(all_of(log_vars), .fns = function(x) log(x+1)))


df$dead_w <- as.integer(df$dead_w)


to_descriptive_stat <- df %>%
  dplyr::select(paste(c(all,'gwcode'))) %>%
  recipe(.) %>%
  prep() %>%
  juice()




cormat <- round(cor(to_descriptive_stat %>% dplyr::select(all_of(all)), use = "pairwise.complete.obs"),2)


colnames(cormat) <- c('Flood mortality', 'Accountability', 'Inclusion', 'Gov. effectiveness', 'Rule of law',  'Conflict history','Local conflict', 'Electoral Democracy', 'Flood severity', 'Flood duration', 'Past flood events', 'Exposed population', 'Rough terrain', 'Tropical flood dummy', 'Local HDI', 'National GDP pc')
rownames(cormat) <-  c('Flood mortality', 'Accountability', 'Inclusion', 'Gov. effectiveness', 'Rule of law',  'Conflict history','Local conflict', 'Electoral Democracy', 'Flood severity', 'Flood duration', 'Past flood events', 'Exposed population', 'Rough terrain', 'Tropical flood dummy', 'Local HDI', 'National GDP pc')


cormat[upper.tri(cormat)] <- NA
cormat <- reshape2::melt(cormat, na.rm = TRUE)






g_cor <- ggplot(data = cormat, aes(x=Var1, y=Var2, color=-value, fill = value, label = value)) + 
  geom_tile() + 
  geom_text(size = 1.7) +
  scale_color_viridis_c(guide = 'none', limits = c(-1, 1)) + 
  scale_fill_viridis_c("Pearson's\nCorrelation", limits = c(-1, 1)) +
  coord_fixed() +
  theme_minimal(base_size = 14) +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, color = "black"),
        axis.text.y = element_text(color = "black"),
        axis.title = element_blank())


g_cor


ggsave(paste0(glue::glue("results/descriptives/cormat.png")), plot = g_cor, device="png", width = 20, height = 15, units = "cm")




#Correlation Plots Figure S3.

df <- readRDS('data/data_final.rds')
df$v2xpe_exlsocgr <- 1 - df$v2xpe_exlsocgr

df <-  subset(df, df$population_affected > 0)

df$dead_w_adjusted <- ifelse(df$dead_w == 0, 0.1, df$dead_w)
log_vars <- c("duration", "population_affected" , "wdi_gdppc", "brd_12mb", "decay_brds_c", "nevents_sum10", "hdi_l1")

df <- df %>%   
  mutate(across(all_of(log_vars), .fns = function(x) log(x+1)))



# Modify your plot specifications to use the filtered dataframe
plot_specs <- list(
  list(x = df$e_wbgi_vae, xlab = "Accountability", y = df$dead_w_adjusted, ylab = "Flood mortality"),
  list(x = df$v2xpe_exlsocgr, xlab = "Inclusion", y = df$dead_w_adjusted, ylab = "Flood mortality"),
  list(x = df$e_wbgi_gee, xlab = "Gov. effectiveness", y = df$dead_w_adjusted, ylab = "Flood mortality"),
  list(x = df$v2x_rule, xlab = "Rule of law", y = df$dead_w_adjusted, ylab = "Flood mortality"),
  list(x = df$decay_brds_c, xlab = "Conflict history", y = df$dead_w_adjusted, ylab = "Flood mortality"),
  list(x = df$brd_12mb, xlab = "Local conflict", y = df$dead_w_adjusted, ylab = "Flood mortality")
)




# Loop through each set of specifications and create/save plots
for (spec in plot_specs) {
  # Loop to create and save plots
  for (spec in plot_specs) {
    x_min <- floor(min(spec$x, na.rm = TRUE))
    x_max <- ceiling(max(spec$x, na.rm = TRUE))
    x_breaks <- seq(x_min, x_max, by = 1)
    
    g_cor2 <- ggplot(df, aes(x = spec$x, y = spec$y)) +
      geom_point(alpha = 0.4, color = "steelblue") +
      geom_smooth(method = "lm", color = "orange", se = FALSE) +
      # Apply conditional scaling for x-axis
      {if (identical(spec$x, df$v2xpe_exlsocgr)) 
        scale_x_continuous(limits = c(0, 1)) 
        else if (identical(spec$x, df$e_wbgi_vae))
          scale_x_continuous(limits = c(-2.5, 2.5))
        else 
          scale_x_continuous(breaks = x_breaks)
      } +
      scale_y_log10(breaks = c(0.1, 1, 10, 100, 1000, 10000, 100000),
                    labels = c("0", "1", "10", "100", "1000", "10,000", "100,000")) +
      xlab(spec$xlab) +
      ylab(spec$ylab) +
      theme(legend.title = element_text(size = 16),
            axis.title.x = element_text(size = 20),
            axis.title.y = element_text(size = 20),
            axis.text.x = element_text(color = "black", size = 16),
            axis.text.y = element_text(color = "black", size = 16),
            legend.key.width = unit(0.8, "line"),
            legend.key.height = unit(0.7, "line"),
            legend.position = "none", 
            legend.box.background = element_rect(colour = "black"),
            panel.background = element_rect(fill = "white"),
            panel.grid.major = element_line(color = "lightgrey", linetype = "dashed"),
            #panel.grid.minor = element_line(color = "lightgrey", linetype = "dashed")
      )
    
    # Save the plot
    filename <- paste0('results/descriptives/', spec$xlab, "_", spec$ylab, ".png")
    ggsave(filename, plot = g_cor2, device="png", width = 14, height = 12, units = "cm")
  }
}


#Time Trends 

#Read data
df <- readRDS('data/data_final.rds')

df$v2xpe_exlsocgr <- 1 - df$v2xpe_exlsocgr
df <-  subset(df, df$population_affected > 0)
# Define the labels and colors for the key variables
labels <- c(
  e_wbgi_vae = "Accountability",
  v2xpe_exlsocgr = "Inclusion",
  e_wbgi_gee = "Gov. effectiveness",
  v2x_rule = "Rule of law",
  decay_brds_c = "Conflict history",
  brd_12mb = "Local conflict",
  v2x_polyarchy = "Electoral democracy"
)

colors <- c(
  'e_wbgi_vae' = "#8eade8",
  'v2xpe_exlsocgr' =  "#3c61a3", 
  'e_wbgi_gee' = "#f2aed8",
  'v2x_rule' = "#c556d1",
  'decay_brds_c' = "#f5b342",
  'brd_12mb' =  "#f0843c",
  'v2x_polyarchy' = "#5df0e1"
)


df <-  subset(df, df$population_affected > 0)

log_vars <- c("brd_12mb", "decay_brds_c")

df <- df %>%   
  mutate(across(all_of(log_vars), .fns = function(x) log(x+1)))


# Filter and subset the data for relevant variables and years
df_filtered <- df %>%
  dplyr::select(year, e_wbgi_vae, e_wbgi_gee, v2xpe_exlsocgr, decay_brds_c, v2x_rule, brd_12mb, v2x_polyarchy)

# Aggregate data by year, calculating the mean (or sum) for each variable
df_grouped <- df_filtered %>%
  group_by(year) %>%
  summarise(across(everything(), mean, na.rm = TRUE))

# Define the desired order of variables
ordered_vars <- c("e_wbgi_vae", "e_wbgi_gee", "decay_brds_c", 
                  "v2xpe_exlsocgr", "v2x_rule", "brd_12mb")




# Modify the create_time_trend_plot function to include y-axis limits
create_time_trend_plot <- function(variable, var_label, color) {
  ggplot(df_grouped, aes(x = year, y = .data[[variable]], group = 1)) +
    geom_line(color = color, size = 1) +
    labs(title = "", x = "Year", y = var_label) +
    scale_x_continuous(breaks = c(seq(2000, 2018, by = 3))) +
    # Conditionally set the y-axis limits based on the variable
    scale_y_continuous(limits = if (variable %in% c("v2x_rule", "v2x_polyarchy", "v2xpe_exlsocgr")) {
      c(0, 1)
    } else if (variable %in% c("e_wbgi_vae", "e_wbgi_gee")) {
      c(-2.5, 2.5)
    } else {
      NULL
    }) +
    theme_bw() +
    theme(
      plot.title = element_text(size = 14, hjust = 0.5, color = "black", family = "Arial"),
      axis.text = element_text(size = 10, color = "black", family = "Arial"),
      axis.title = element_text(size = 12, color = "black", family = "Arial")
    )
}

# The rest of the code remains unchanged

# Create individual time trend plots in the desired order
plot_list <- lapply(ordered_vars, function(var) {
  create_time_trend_plot(var, labels[var], colors[var])
})

# Combine all the individual plots into a single layout using patchwork
combined_plot <- wrap_plots(plotlist = plot_list, ncol = 3)


# Save the combined plot to a file
ggsave(paste0("results/descriptives/time_trends.png"), plot = combined_plot, device = "png", width = 10, height = 6, dpi = 350)



#Create within and between variance table Supplementary Fig
#Read data
df <- readRDS('data/data_final.rds')
df$v2xpe_exlsocgr <- 1 - df$v2xpe_exlsocgr

df <-  subset(df, df$population_affected > 0)


vars <- c("e_wbgi_vae", "v2xpe_exlsocgr", "e_wbgi_gee", 
               "v2x_rule", "decay_brds_c", "brd_12mb")


# Compute between and within country standard deviations
summary_sd <- df %>%
  dplyr::select(gwcode, all_of(vars)) %>%
  pivot_longer(-gwcode, names_to = "variable", values_to = "value") %>%
  filter(!is.na(value)) %>%
  group_by(variable) %>%
  summarise(
    between = {
      country_means <- tapply(value, gwcode, mean, na.rm = TRUE)
      sd(country_means, na.rm = TRUE)
    },
    within = {
      country_sds <- tapply(value, gwcode, sd, na.rm = TRUE)
      mean(country_sds, na.rm = TRUE)
    }
  ) %>%
  mutate(ratio = within / between)  # Add the ratio column

summary_sd <- summary_sd %>%
  mutate(label = labels[variable]) %>%
  dplyr::select(label, between, within, ratio)  # Include ratio in selection

names(summary_sd) <- c("Variable", "Between-country SD", "Within-country SD", "Ratio (Within/Between)")

modelsummary::datasummary_df(
  summary_sd,
  output = paste0("results/descriptives/variance_between_within.tex"), 
  fmt = 2
)
