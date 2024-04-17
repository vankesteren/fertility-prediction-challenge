library(softImpute)
library(ranger)
train_save_model <- function(cleaned_df, outcome_df) {
  set.seed(45)

  # do imputation
  x_sel   <- as.matrix(cleaned_df)
  imp_mod <- softImpute(x_sel, rank.max = 20)
  x       <- complete(x_sel, imp_mod)
  y       <- outcome_df$new_child
  id      <- !is.na(y)

  # do randomforest
  fit <- ranger(y = factor(y[id]), x = x[id, ], oob.error = TRUE)

  # save
  saveRDS(list(impute_mod = imp_mod, predict_mod = fit), file = "model.rds")

}
