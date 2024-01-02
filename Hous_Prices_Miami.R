#' ---
#' title: "{shapviz} demo"
#' subtitle: "With Miami House Prices"
#' author: "Michael Mayer"
#' date: "January 11, 2024"
#' output:
#'  html_document:
#'    toc: yes
#'    toc_float: yes
#'    toc_depth: 3
#'    number_sections: no
#'    df_print: paged
#'    theme: paper
#'    code_folding: show
#'    math_method: katex
#' ---

#' ## Introduction
#' 
#' Welcome to this compact {shapviz} demo. We will model logarithmic house prices by
#' typical features using XGBoost and study the model via SHAP.
#' 
#' ## The data
#' 
#' We will use 14k rows of miami house price data prepared by Prof. Steve Bourassa.
#' It is shipped with {shapviz}.

knitr::opts_chunk$set(
  fig.width = 6, fig.height = 5, comment = "", message = FALSE, warning = FALSE
)

#'

library(xgboost)
library(ggplot2)
library(patchwork)
library(shapviz)

# Log transforms and feature selection
df <- miami |> 
  transform(
    log_price = log(SALE_PRC),
    log_living = log(TOT_LVG_AREA),
    log_land = log(LND_SQFOOT),
    center_dist = CNTR_DIST / 1000
  )
xvar <- c(
  "center_dist", "log_living", "log_land", "structure_quality", "age", "month_sold"
)
df <- df[c("log_price", xvar)]

head(df)
summary(df)

#' ## XGBoost model
#' 
#' We fit an XGBoost model by manually tuning the main parameters and choosing the
#' number of trees by early stopping on the validation data.

#' ### Train/valid split

set.seed(1)
ix <- sample(nrow(df), 0.8 * nrow(df))
X_train <- data.matrix(df[ix, xvar])
X_valid <- data.matrix(df[-ix, xvar])

dtrain <- xgb.DMatrix(X_train, label = df$log_price[ix])
dvalid <- xgb.DMatrix(X_valid, label = df$log_price[-ix])

#' ### Fit

params <- list(
  learning_rate = 0.05, 
  objective = "reg:squarederror", 
  max_depth = 6, 
  reg_alpha = 2,
  reg_lambda = 0
)

fit_xgb <- xgb.train(
  params = params, 
  data = dtrain, 
  watchlist = list(valid = dvalid), 
  early_stopping_rounds = 20,
  nrounds = 1000,
  callbacks = list(cb.print.evaluation(period = 100))
)

#' ## SHAP analysis
#' 
#' We use 1000 rows from the training data to describe the model.
#' Note that the training indices have been shuffled during train/test split, so that
#' we are indeed using a random sample here.
#' 
#' ### Make "shapviz" object
#' 
#' For an XGBoost model, `shapviz()` internally calls 
#' `predict(fit_xgb, X_pred, predcontrib = TRUE)` to have access to the native TreeSHAP
#' implementation in XGBoost.
#' Note that the XGBoost method of `shapviz()` uses `X = X_pred` by default. `X` could 
#' be overwritten to contain factor variables.

X_explain <- X_train[1:1000, ]
shap_xgb <- shapviz(fit_xgb, X_pred = X_explain)
shap_xgb
head(shap_xgb$X, 2)

#' ### Importance plots
#' 
#' Note that the plots are "ggplots" and can be modified accordingly.

sv_importance(shap_xgb) +
  ggtitle("SHAP importance plot")

sv_importance(shap_xgb, kind = "bee") +
  ggtitle("SHAP summary plot")

#' ### Dependence plots
#' 
#' Multiple plots are glued together via {patchwork}. This allows different x and color
#' scales. By default, the color scale is automatically chosen by a heuristic.

#+ fig.width=10
sv_dependence(shap_xgb, v = xvar, alpha = 0.2)

#' ### Individual predictions
#' 
#' We can also visualize decompositions of single observations via waterfall plot or 
#' force plot.

sv_waterfall(shap_xgb, row_id = 1)

#+ fig.height=2
sv_force(shap_xgb, row_id = 1)

#' ### Some syntactic sugar
#' 
#' Different operators work on "shapviz" objects, e.g., 
#' 
#' - subset rows and/or columns via `[`,
#' - splitting,
#' - unsplitting via `+` or `rbind()`,
#' - `dim()`, `head()`, `nrow()`, `colnames()` etc.
dim(shap_xgb)

# Split by "structure_quality"
f <- shap_xgb$X$structure_quality
sv_dependence(split(shap_xgb, f), v = "log_living")

#' ## Linear model
#' 
#' We can use {kernelshap} to look at similar plots of any model, e.g., a complex
#' linear model with natural cubic splines and interactions.
#' 
#' ### Fit

library(splines)
library(kernelshap)

fit_lm <- lm(
  log_price ~ ns(center_dist, 5) + ns(log_living, 5) * (center_dist < 65) + log_land + 
              factor(structure_quality) + ns(age, 5) + ns(month_sold, 3),
  data = df, 
  subset = ix
)

#' ### SHAP analysis
#' 
#' The model is not tree-based, and we have only six features. Thus, we use exact
#' permutation SHAP (exact only with respect to the selected 'background' data).
#' 
#' Note that a "shapviz" object can be easily saved as rds or RData file. It contains
#' all information required to build all plots.
#'
#' Further note that we use {patchwork}'s `&` operator to set the same `ylim` to each
#' dependence plot. An equal y-scale immediately gives a good impression on the effect
#' strenght of the features.

df_explain <- as.data.frame(X_explain)

# Set to TRUE to recalculate permutation SHAP values
if (FALSE) {
  system.time(  # 45 seconds
    ks <- permshap(fit_lm, X = df_explain, bg_X = df_explain[1:200, ])
  )
  shap_lm <- shapviz(ks)
  saveRDS(shap_lm, file = "shap_lm.rds")
} else {
  shap_lm <- readRDS("shap_lm.rds")
}

#+ fig.width=10
sv_dependence(shap_lm, v = xvar) &
  ylim(-0.8, 1)

#' ## Bind objects
#' 
#' Multiple "shapviz" objects can be glued together to an "mshapviz" objects via `c()`. 
#' Also multi-output models or `split()` produce "mshapviz" objects.

mshap <- c(xgb = shap_xgb, lm = shap_lm)
mshap

sv_importance(mshap)

#+ fig.height=3
sv_dependence(mshap, v = "log_living")

