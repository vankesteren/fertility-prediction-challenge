library(data.table)
library(tidyverse)
library(Matrix)
library(xgboost)

data <- data.table::fread("data/training_data/PreFer_train_data.csv", 
                          keepLeadingZeros = TRUE, # if FALSE adds zeroes to some dates
                          data.table = FALSE) # returns a data.frame object rather than data.table 

outcome <- data.table::fread("data/training_data/PreFer_train_outcome.csv", 
                          keepLeadingZeros = TRUE, # if FALSE adds zeroes to some dates
                          data.table = FALSE) %>%
  na.omit(.)

bckg <- data.table::fread("data/other_data/PreFer_train_background_data.csv", 
                             keepLeadingZeros = TRUE, # if FALSE adds zeroes to some dates
                             data.table = FALSE)

merged <- left_join(outcome, data, by = "nomem_encr") #%>%
  #left_join(., bckg, by = "nomem_encr")

# IGNORE
# cf_only <- merged %>% 
#   select(starts_with("cf")) %>%
#   select(c(426:656, 1047:1241)) %>%
#   cbind(merged[, 1:2], .)
# 
# years <- unique(substr(colnames(cf_only[3:428]), 3, 4))
# waves <- unique(substr(colnames(cf_only[3:428]), 5, 5))
# 
# long_data <- pivot_longer(cf_only,
#                           names_to = c("year", "wave", ".value"), 
#                           names_pattern = "cf(\\d{2})(L\\d{3})")
# 
# long_data$year <- as.numeric(long_data$year)
# long_data$wave <- as.numeric(substr(long_data$wave, 2, 3))

######

# c refers to Core study
# f “Family and household”, w (Work and schooling), s (Social integration and values),
# h (Health), r (Religion and ethnicity), v (politics and Values), p (Personality),
# a (economic situation: Assets), i (economic situation: Income), d (economic situation: Housing)

merged <- replace(merged, is.na(merged), 999)

family <- merged %>% 
  select(starts_with("cf")) %>%
  cbind(merged[, 1:2], .)

work <- merged %>% 
  select(starts_with("cw")) %>%
  cbind(merged[, 1:2], .) %>%
  keep(~mean(is.na(.)) <= 0.8) # too many columns are nearly all NAs

social <- merged %>% 
  select(starts_with("cs")) %>%
  cbind(merged[, 1:2], .) 

health <- merged %>% 
  select(starts_with("ch")) %>%
  cbind(merged[, 1:2], .) 

religion <- merged %>% 
  select(starts_with("cr")) %>%
  cbind(merged[, 1:2], .) 

values <- merged %>% 
  select(starts_with("cv")) %>%
  cbind(merged[, 1:2], .)

personality <- merged %>% 
  select(starts_with("cp")) %>%
  cbind(merged[, 1:2], .) 

assets <- merged %>% 
  select(starts_with("ca")) %>%
  cbind(merged[, 1:2], .) 

income <- merged %>% 
  select(starts_with("ci")) %>%
  cbind(merged[, 1:2], .) 

housing <- merged %>% 
  select(starts_with("cd")) %>%
  cbind(merged[, 1:2], .)

library(devtools)
source_url("https://github.com/pablo14/shap-values/blob/master/shap.R?raw=TRUE")

test_set <- sample(merged$nomem_encr, nrow(merged) * 0.2)

## FAMILY ##

train_x <- sparse.model.matrix(new_child ~ ., family)[!(family$nomem_encr %in% test_set),-c(1:2)]
train_y <- family[!(family$nomem_encr %in% test_set), "new_child"]
xgboost_train <- xgboost(data = train_x,
                         label = train_y, 
                         max.depth = 10,
                         eta = 1,
                         nthread = 4,
                         nrounds = 4,
                         objective = "binary:logistic",
                         verbose = 2)

test_x <- sparse.model.matrix(new_child ~ ., family)[family$nomem_encr %in% test_set,-c(1:2)]
test_y <- family[family$nomem_encr %in% test_set, "new_child"]

pred <- tibble(new_child = predict(xgboost_train, newdata = test_x)) %>%
  mutate(new_child = factor(ifelse(new_child < 0.5, 0, 1)))

family_cm <- table(observed = test_y, predicted = pred$new_child)
family_cm[2, 2] / sum(family_cm[2, ]) # TPR = correctly predicted 1s 0.6521739
family_cm[1, 1] / sum(family_cm[1, ]) # TNR = correctly predicted 0s 0.9470199
sum(diag(family_cm)) / sum(family_cm) # accuracy 0.8781726

shap_results <- shap.score.rank(xgboost_train,
                                X_train = train_x,
                                shap_approx = T)

family_top10 <- shap_results$mean_shap_score[1:10]
family_top10

## WORK ##

factor_levels <- sapply(work, function(x) length(levels(as.factor(x))))
multi_level <- names(factor_levels[factor_levels > 1])
work <- work[, multi_level]

train_x <- sparse.model.matrix(new_child ~ ., work)[!(work$nomem_encr %in% test_set),-c(1:2)]
train_y <- work[!(work$nomem_encr %in% test_set), "new_child"]
xgboost_train <- xgboost(data = train_x,
                         label = train_y, 
                         max.depth = 10,
                         eta = 1,
                         nthread = 4,
                         nrounds = 4,
                         objective = "binary:logistic",
                         verbose = 2)

test_x <- sparse.model.matrix(new_child ~ ., work)[work$nomem_encr %in% test_set,-c(1:2)]
test_y <- work[work$nomem_encr %in% test_set, "new_child"]

pred <- tibble(new_child = predict(xgboost_train, newdata = test_x)) %>%
  mutate(new_child = factor(ifelse(new_child < 0.5, 0, 1)))

work_cm <- table(observed = test_y, predicted = pred$new_child)
work_cm[2, 2] / sum(work_cm[2, ]) # TPR = correctly predicted 1s 0.3478261
work_cm[1, 1] / sum(work_cm[1, ]) # TNR = correctly predicted 0s 0.8940397
sum(diag(work_cm)) / sum(work_cm) # accuracy 0.7664975

shap_results <- shap.score.rank(xgboost_train,
                                X_train = train_x,
                                shap_approx = T)

work_top10 <- shap_results$mean_shap_score[1:10]
work_top10


## SOCIAL ##

factor_levels <- sapply(social, function(x) length(levels(as.factor(x))))
multi_level <- names(factor_levels[factor_levels > 1])
social <- social[, multi_level]

train_x <- sparse.model.matrix(new_child ~ ., social)[!(social$nomem_encr %in% test_set),-c(1:2)]
train_y <- social[!(social$nomem_encr %in% test_set), "new_child"]
xgboost_train <- xgboost(data = train_x,
                         label = train_y, 
                         max.depth = 10,
                         eta = 1,
                         nthread = 4,
                         nrounds = 4,
                         objective = "binary:logistic",
                         verbose = 2)

test_x <- sparse.model.matrix(new_child ~ ., social)[social$nomem_encr %in% test_set,-c(1:2)]
test_y <- social[social$nomem_encr %in% test_set, "new_child"]

pred <- tibble(new_child = predict(xgboost_train, newdata = test_x)) %>%
  mutate(new_child = factor(ifelse(new_child < 0.5, 0, 1)))

social_cm <- table(observed = test_y, predicted = pred$new_child)
social_cm[2, 2] / sum(social_cm[2, ]) # TPR = correctly predicted 1s 0.2391304
social_cm[1, 1] / sum(social_cm[1, ]) # TNR = correctly predicted 0s 0.9403974
sum(diag(social_cm)) / sum(social_cm) # accuracy 0.7766497

shap_results <- shap.score.rank(xgboost_train,
                                X_train = train_x,
                                shap_approx = T)

social_top10 <- shap_results$mean_shap_score[1:10]
social_top10


## HEALTH ##

factor_levels <- sapply(health, function(x) length(levels(as.factor(x))))
multi_level <- names(factor_levels[factor_levels > 1])
health <- health[, multi_level]

train_x <- sparse.model.matrix(new_child ~ ., health)[!(health$nomem_encr %in% test_set),-c(1:2)]
train_y <- health[!(health$nomem_encr %in% test_set), "new_child"]
xgboost_train <- xgboost(data = train_x,
                         label = train_y, 
                         max.depth = 10,
                         eta = 1,
                         nthread = 4,
                         nrounds = 4,
                         objective = "binary:logistic",
                         verbose = 2)

test_x <- sparse.model.matrix(new_child ~ ., health)[health$nomem_encr %in% test_set,-c(1:2)]
test_y <- health[health$nomem_encr %in% test_set, "new_child"]

pred <- tibble(new_child = predict(xgboost_train, newdata = test_x)) %>%
  mutate(new_child = factor(ifelse(new_child < 0.5, 0, 1)))

health_cm <- table(observed = test_y, predicted = pred$new_child)
health_cm[2, 2] / sum(health_cm[2, ]) # TPR = correctly predicted 1s 0.2826087
health_cm[1, 1] / sum(health_cm[1, ]) # TNR = correctly predicted 0s 0.8940397
sum(diag(health_cm)) / sum(health_cm) # accuracy 0.751269

shap_results <- shap.score.rank(xgboost_train,
                                X_train = train_x,
                                shap_approx = T)

health_top10 <- shap_results$mean_shap_score[1:10]
health_top10


## RELIGION ##

factor_levels <- sapply(religion, function(x) length(levels(as.factor(x))))
multi_level <- names(factor_levels[factor_levels > 1])
religion <- religion[, multi_level]

train_x <- sparse.model.matrix(new_child ~ ., religion)[!(religion$nomem_encr %in% test_set),-c(1:2)]
train_y <- religion[!(religion$nomem_encr %in% test_set), "new_child"]
xgboost_train <- xgboost(data = train_x,
                         label = train_y, 
                         max.depth = 10,
                         eta = 1,
                         nthread = 4,
                         nrounds = 4,
                         objective = "binary:logistic",
                         verbose = 2)

test_x <- sparse.model.matrix(new_child ~ ., religion)[religion$nomem_encr %in% test_set,-c(1:2)]
test_y <- religion[religion$nomem_encr %in% test_set, "new_child"]

pred <- tibble(new_child = predict(xgboost_train, newdata = test_x)) %>%
  mutate(new_child = factor(ifelse(new_child < 0.5, 0, 1)))

religion_cm <- table(observed = test_y, predicted = pred$new_child)
religion_cm[2, 2] / sum(religion_cm[2, ]) # TPR = correctly predicted 1s 0.2826087
religion_cm[1, 1] / sum(religion_cm[1, ]) # TNR = correctly predicted 0s 0.8609272
sum(diag(religion_cm)) / sum(religion_cm) # accuracy 0.7258883

shap_results <- shap.score.rank(xgboost_train,
                                X_train = train_x,
                                shap_approx = T)

religion_top10 <- shap_results$mean_shap_score[1:10]
religion_top10

## VALUES ##

factor_levels <- sapply(values, function(x) length(levels(as.factor(x))))
multi_level <- names(factor_levels[factor_levels > 1])
values <- values[, multi_level]

train_x <- sparse.model.matrix(new_child ~ ., values)[!(values$nomem_encr %in% test_set),-c(1:2)]
train_y <- values[!(values$nomem_encr %in% test_set), "new_child"]
xgboost_train <- xgboost(data = train_x,
                         label = train_y, 
                         max.depth = 10,
                         eta = 1,
                         nthread = 4,
                         nrounds = 4,
                         objective = "binary:logistic",
                         verbose = 2)

test_x <- sparse.model.matrix(new_child ~ ., values)[values$nomem_encr %in% test_set,-c(1:2)]
test_y <- values[values$nomem_encr %in% test_set, "new_child"]

pred <- tibble(new_child = predict(xgboost_train, newdata = test_x)) %>%
  mutate(new_child = factor(ifelse(new_child < 0.5, 0, 1)))

values_cm <- table(observed = test_y, predicted = pred$new_child)
values_cm[2, 2] / sum(values_cm[2, ]) # TPR = correctly predicted 1s 0.2391304
values_cm[1, 1] / sum(values_cm[1, ]) # TNR = correctly predicted 0s 0.9271523
sum(diag(values_cm)) / sum(values_cm) # accuracy 0.7664975

shap_results <- shap.score.rank(xgboost_train,
                                X_train = train_x,
                                shap_approx = T)

values_top10 <- shap_results$mean_shap_score[1:10]
values_top10

## personality ##

factor_levels <- sapply(personality, function(x) length(levels(as.factor(x))))
multi_level <- names(factor_levels[factor_levels > 1])
personality <- personality[, multi_level]

train_x <- sparse.model.matrix(new_child ~ ., personality)[!(personality$nomem_encr %in% test_set),-c(1:2)]
train_y <- personality[!(personality$nomem_encr %in% test_set), "new_child"]
xgboost_train <- xgboost(data = train_x,
                         label = train_y, 
                         max.depth = 10,
                         eta = 1,
                         nthread = 4,
                         nrounds = 4,
                         objective = "binary:logistic",
                         verbose = 2)

test_x <- sparse.model.matrix(new_child ~ ., personality)[personality$nomem_encr %in% test_set,-c(1:2)]
test_y <- personality[personality$nomem_encr %in% test_set, "new_child"]

pred <- tibble(new_child = predict(xgboost_train, newdata = test_x)) %>%
  mutate(new_child = factor(ifelse(new_child < 0.5, 0, 1)))

personality_cm <- table(observed = test_y, predicted = pred$new_child)
personality_cm[2, 2] / sum(personality_cm[2, ]) # TPR = correctly predicted 1s 0.2173913
personality_cm[1, 1] / sum(personality_cm[1, ]) # TNR = correctly predicted 0s 0.8874172
sum(diag(personality_cm)) / sum(personality_cm) # accuracy 0.7309645

shap_results <- shap.score.rank(xgboost_train,
                                X_train = train_x,
                                shap_approx = T)

personality_top10 <- shap_results$mean_shap_score[1:10]
personality_top10

## assets ##

factor_levels <- sapply(assets, function(x) length(levels(as.factor(x))))
multi_level <- names(factor_levels[factor_levels > 1])
assets <- assets[, multi_level]

train_x <- sparse.model.matrix(new_child ~ ., assets)[!(assets$nomem_encr %in% test_set),-c(1:2)]
train_y <- assets[!(assets$nomem_encr %in% test_set), "new_child"]
xgboost_train <- xgboost(data = train_x,
                         label = train_y, 
                         max.depth = 10,
                         eta = 1,
                         nthread = 4,
                         nrounds = 4,
                         objective = "binary:logistic",
                         verbose = 2)

test_x <- sparse.model.matrix(new_child ~ ., assets)[assets$nomem_encr %in% test_set,-c(1:2)]
test_y <- assets[assets$nomem_encr %in% test_set, "new_child"]

pred <- tibble(new_child = predict(xgboost_train, newdata = test_x)) %>%
  mutate(new_child = factor(ifelse(new_child < 0.5, 0, 1)))

assets_cm <- table(observed = test_y, predicted = pred$new_child)
assets_cm[2, 2] / sum(assets_cm[2, ]) # TPR = correctly predicted 1s 0.1086957
assets_cm[1, 1] / sum(assets_cm[1, ]) # TNR = correctly predicted 0s 0.9139073
sum(diag(assets_cm)) / sum(assets_cm) # accuracy 0.7258883

shap_results <- shap.score.rank(xgboost_train,
                                X_train = train_x,
                                shap_approx = T)

assets_top10 <- shap_results$mean_shap_score[1:10]
assets_top10

## income ##

factor_levels <- sapply(income, function(x) length(levels(as.factor(x))))
multi_level <- names(factor_levels[factor_levels > 1])
income <- income[, multi_level]

train_x <- sparse.model.matrix(new_child ~ ., income)[!(income$nomem_encr %in% test_set),-c(1:2)]
train_y <- income[!(income$nomem_encr %in% test_set), "new_child"]
xgboost_train <- xgboost(data = train_x,
                         label = train_y, 
                         max.depth = 10,
                         eta = 1,
                         nthread = 4,
                         nrounds = 4,
                         objective = "binary:logistic",
                         verbose = 2)

test_x <- sparse.model.matrix(new_child ~ ., income)[income$nomem_encr %in% test_set,-c(1:2)]
test_y <- income[income$nomem_encr %in% test_set, "new_child"]

pred <- tibble(new_child = predict(xgboost_train, newdata = test_x)) %>%
  mutate(new_child = factor(ifelse(new_child < 0.5, 0, 1)))

income_cm <- table(observed = test_y, predicted = pred$new_child)
income_cm[2, 2] / sum(income_cm[2, ]) # TPR = correctly predicted 1s 0.3043478
income_cm[1, 1] / sum(income_cm[1, ]) # TNR = correctly predicted 0s 0.9006623
sum(diag(income_cm)) / sum(income_cm) # accuracy 0.7614213

shap_results <- shap.score.rank(xgboost_train,
                                X_train = train_x,
                                shap_approx = T)

income_top10 <- shap_results$mean_shap_score[1:10]
income_top10

## housing ##

factor_levels <- sapply(housing, function(x) length(levels(as.factor(x))))
multi_level <- names(factor_levels[factor_levels > 1])
housing <- housing[, multi_level]

train_x <- sparse.model.matrix(new_child ~ ., housing)[!(housing$nomem_encr %in% test_set),-c(1:2)]
train_y <- housing[!(housing$nomem_encr %in% test_set), "new_child"]
xgboost_train <- xgboost(data = train_x,
                         label = train_y, 
                         max.depth = 10,
                         eta = 1,
                         nthread = 4,
                         nrounds = 4,
                         objective = "binary:logistic",
                         verbose = 2)

test_x <- sparse.model.matrix(new_child ~ ., housing)[housing$nomem_encr %in% test_set,-c(1:2)]
test_y <- housing[housing$nomem_encr %in% test_set, "new_child"]

pred <- tibble(new_child = predict(xgboost_train, newdata = test_x)) %>%
  mutate(new_child = factor(ifelse(new_child < 0.5, 0, 1)))

housing_cm <- table(observed = test_y, predicted = pred$new_child)
housing_cm[2, 2] / sum(housing_cm[2, ]) # TPR = correctly predicted 1s 0.1521739
housing_cm[1, 1] / sum(housing_cm[1, ]) # TNR = correctly predicted 0s 0.9072848
sum(diag(housing_cm)) / sum(housing_cm) # accuracy 0.7309645

shap_results <- shap.score.rank(xgboost_train,
                                X_train = train_x,
                                shap_approx = T)

housing_top10 <- shap_results$mean_shap_score[1:10]
housing_top10

## Top 100 features ##

top_feat <- c(family_top10, work_top10, social_top10, health_top10, religion_top10,
              values_top10, personality_top10, assets_top10, income_top10, housing_top10)
names(top_feat)
