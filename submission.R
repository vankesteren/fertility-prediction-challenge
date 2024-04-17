library(softImpute)
library(ranger)

clean_df <- function(df, background_df = NULL){
  keepcols <- c(42:54, 1440:1477)

  ## Keeping data with variables selected
  df <- df[ , keepcols ]

  return(df)
}

predict_outcomes <- function(df, background_df = NULL, model_path = "./model.rds"){
  ## This script contains a bare minimum working example
  if( !("nomem_encr" %in% colnames(df)) ) {
    warning("The identifier variable 'nomem_encr' should be in the dataset")
  }

  # extract id column
  id <- df$nomem_encr

  # Load the model
  model <- readRDS(model_path)

  # Preprocess the fake / holdout data
  cleaned_df <- clean_df(df, background_df)

  # impute potential missings
  x_sel <- as.matrix(cleaned_df)
  x_imp <- complete(x_sel, model$impute_mod)

  # Generate predictions from model
  predictions_factor <- predict(model$predict_mod, x_imp)$predictions
  predictions_int <- as.integer(predictions_factor) - 1

  # Output file should be data.frame with two columns, nomem_encr and predictions
  df_predict <- data.frame("nomem_encr" = id, "prediction" = predictions_int)

  # Return only dataset with predictions and identifier
  return(df_predict)
}
