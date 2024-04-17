library(tidyverse)
library(softImpute)
library(ranger)

df <- data.table::fread(
  "data/training_data/PreFer_train_data.csv",
  keepLeadingZeros = TRUE, data.table = FALSE
)
outcome <- data.table::fread(
  "data/training_data/PreFer_train_outcome.csv",
  keepLeadingZeros = TRUE, data.table = FALSE
)

clean_df <- function(df, background_df = NULL) {

  # do some fancy dimension reduction
  df_id <- df$nomem_encr
  df_num <- df |> select(where(is.numeric), where(is.logical), -nomem_encr, -outcome_available)
  df_fac <- df |> select(where(is.character)) |>
    mutate(across(everything(), na_if, y = ""),
           across(everything(), as.factor),
           across(everything(), as.numeric))
  mat_na <-
    cbind(df_num, df_fac) |>
    as.matrix() |>
    scale(scale = FALSE) |>
    as("Incomplete")

  fitsvd <- softImpute(mat_na, rank.max = 200, lambda = 20, trace.it = TRUE, type = "als")


  y <- factor(outcome$new_child[!is.na(outcome$new_child)])
  x <- fitsvd$u[!is.na(outcome$new_child),]
  colnames(x) <- paste0("V", 1:ncol(x))
  fit_rf <- ranger(y = y, x = x, num.trees = 3000)



  # first, select columns
  df_sel <- df |> select(nomem_encr, starts_with("cf"), ends_with("\\d{4}"))

  # then, convert categorical data to appropriate variable types
  codebook <- read_codebook()
  avail_vars <- tibble(var_name = colnames(df))

  # find categorical variables
  cat_vars <-
    codebook |>
    filter(type_var == "categorical") |>
    inner_join(avail_vars, by = join_by(var_name)) |>
    pull(var_name)

  df <- df |> mutate(across(all_of(cat_vars), convert_to_factor))

  return(df)
}

load_and_clean_data <- function() {
  cat("Loading main data")
  df <- data.table::fread("data/training_data/PreFer_train_data.csv", keepLeadingZeros = TRUE, data.table = FALSE)
  cat("Loading background data")
  background_df <- data.table::fread("data/other_data/PreFer_train_background_data.csv", keepLeadingZeros = TRUE, data.table = FALSE)
  cat("Cleaning data...")
  clean_df(df, background_df)
}

read_codebook <- function() {
  read_csv(
    file = "data/codebooks/PreFer_codebook.csv",
    col_types = cols(
      var_name = col_character(),
      var_label = col_character(),
      values_cat = col_character(),
      labels_cat = col_character(),
      unique_values_n = col_double(),
      n_missing = col_double(),
      prop_missing = col_double(),
      type_var = col_factor(),
      note = col_character(),
      year = col_double(),
      survey = col_factor(),
      dataset = col_factor()
    )
  )
}

convert_to_factor <- function(x) {
  colname <- as.character(substitute(x))
  code_entry <- codebook |> filter(var_name == colname)
  lvl <- code_entry |> pull(values_cat) |> str_split(";", simplify = TRUE) |> str_trim()
  lbl <- code_entry |> pull(labels_cat) |> str_split(";", simplify = TRUE) |> str_trim()
  factor(x, levels = lvl, labels = lbl)
}

