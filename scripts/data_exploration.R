library(data.table)
library(ranger)
library(softImpute)

df_train <-
  data.table::fread(
    "data/training_data/PreFer_train_data.csv",
    keepLeadingZeros = TRUE,
    data.table = FALSE
  )
outcome <-
  data.table::fread(
    "data/training_data/PreFer_train_outcome.csv",
    keepLeadingZeros = TRUE,
    data.table = FALSE
  )


x_sel   <- df_train[,c(42:54, 1440:1477)]
imp_mod <- softImpute(as.matrix(x_sel), rank.max = 20)
x       <- complete(x_sel, si)
y       <- outcome$new_child
id      <- !is.na(y)

fit <- ranger(y = factor(y[id]), x = x[id, ], oob.error = TRUE)
1 - fit$prediction.error
fit$confusion.matrix

saveRDS(list(
  impute_mod = imp_mod,
  predict_mod = fit
), file = "model.rds")
