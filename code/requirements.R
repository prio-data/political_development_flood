
# Clear workspace
rm(list = ls(all = TRUE))

# Install required packages
required_packages <- c(
  "randomForest", "posterior", "ggridges", "patchwork", "future", 
  "Metrics", "LongituRF", "splitstackshape", "gpboost", "groupdata2", 
  "rsample", "progress", "caret", "utils", "itsadug", "splitTools", 
  "additive", "cvTools", "workflows", "broom.mixed", "xtable", 
  "glmmTMB", "Amelia", "lme4", "validate", "tidyverse", "HH", 
  "skimr", "recipes", "bayestestR", "knitr", "tidybayes", "mice", 
  "gt", "scales", "sjPlot", "ggmap", "RColorBrewer", "robustlmm", 
  "countrycode", "bayesplot", "brms", "loo", "see", "cmdstanr", 
  "rlang", "ggbeeswarm", "extrafont", "HDInterval", "texreg", 
  "dbarts", "cowplot", "magick", "webshot2"
)

# Install missing packages
installed_packages <- rownames(installed.packages())
to_install <- setdiff(required_packages, installed_packages)
if(length(to_install) > 0) install.packages(to_install)

# Load required libraries
lapply(required_packages, library, character.only = TRUE)

# Install brms from GitHub (make sure your GitHub path is set correctly and you have a Github Token)
tryCatch({
  remotes::install_github("paul-buerkner/brms", dependencies = TRUE)
}, error = function(e) {
  stop("Failed to install 'brms' from GitHub. Check your GitHub PAT and internet connection.")
})

# Additional setup
font_import()
loadfonts(device = "win")
