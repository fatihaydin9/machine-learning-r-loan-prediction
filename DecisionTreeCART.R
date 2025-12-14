# ------------------------------------------------------------------------------------
# CART Decision Tree (writeoff) - Stratified 80/20 Holdout + CV Hyperparameter Tuning
# ------------------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
  library(janitor)
})

if (!requireNamespace("rpart", quietly = TRUE)) install.packages("rpart")
library(rpart)

# ------------------------------------------------------------
# Veri okuma + tip dönüşümü
# ------------------------------------------------------------

loans   <- readr::read_csv("Loans.csv", na = c("", "NA", "?", "NULL"), show_col_types = FALSE)
newapps <- readr::read_csv("NewApplicants.csv", na = c("", "NA", "?", "NULL"), show_col_types = FALSE)

loans$writeoff   <- factor(loans$writeoff, levels = c("no", "yes"))
newapps$writeoff <- factor(newapps$writeoff, levels = levels(loans$writeoff))  # NA kalır

cat_cols <- c("gender","marital_status","education","employ_status",
              "spouse_work","residential_status","loan_purpose","collateral")

num_cols <- c("age","nb_depend_child","yrs_current_job","yrs_employed",
              "net_income","spouse_income","yrs_current_address",
              "loan_amount","loan_length")

loans[num_cols]   <- lapply(loans[num_cols], as.numeric)
newapps[num_cols] <- lapply(newapps[num_cols], as.numeric)

loans[cat_cols]   <- lapply(loans[cat_cols], as.factor)
newapps[cat_cols] <- lapply(newapps[cat_cols], as.factor)

# newapps factor seviyeleri loans ile aynı olsun (garanti)
for (col in cat_cols) {
  newapps[[col]] <- factor(newapps[[col]], levels = levels(loans[[col]]))
}

# ------------------------------------------------------------
# Stratified 80/20 Holdout (TEST dokunulmaz)
# ------------------------------------------------------------

set.seed(123)
loans <- loans %>% mutate(row_id = row_number())

train_ids <- loans %>%
  group_by(writeoff) %>%
  slice_sample(prop = 0.8) %>%
  ungroup() %>%
  pull(row_id)

train_df <- loans %>% filter(row_id %in% train_ids) %>% select(-row_id)
test_df  <- loans %>% filter(!row_id %in% train_ids) %>% select(-row_id)

# test factor seviyeleri train ile uyumlu
for (col in cat_cols) {
  test_df[[col]] <- factor(test_df[[col]], levels = levels(train_df[[col]]))
}

# 1) Train/Test yes-no dağılımı (%)  [ÇIKTI]
train_dist_pct <- round(100 * prop.table(table(train_df$writeoff)), 2)
test_dist_pct  <- round(100 * prop.table(table(test_df$writeoff)),  2)

cat("\nTrain dağılımı (%)\n"); print(train_dist_pct)
cat("\nTest dağılımı (%)\n");  print(test_dist_pct)

# ------------------------------------------------------------
# Yardımcı fonksiyonlar
# ------------------------------------------------------------

safe_div <- function(a, b) ifelse(b == 0, NA_real_, a / b)

metrics_from_cm <- function(cm) {
  acc  <- sum(diag(cm)) / sum(cm)
  prec <- safe_div(cm["yes","yes"], sum(cm[,"yes"]))
  rec  <- safe_div(cm["yes","yes"], sum(cm["yes",]))
  f1   <- ifelse(is.na(prec) | is.na(rec) | (prec + rec) == 0, NA_real_,
                 2 * prec * rec / (prec + rec))
  tibble(accuracy = acc, precision = prec, recall = rec, f1 = f1)
}

best_threshold_by_f1 <- function(truth, prob_yes, thresholds = seq(0.1, 0.99, by = 0.01)) {
  f1_vals <- sapply(thresholds, function(t) {
    pred <- factor(ifelse(prob_yes >= t, "yes", "no"), levels = c("no","yes"))
    cm <- table(truth = truth, pred = pred)
    metrics_from_cm(cm)$f1
  })
  thresholds[which.max(f1_vals)]
}

make_stratified_folds <- function(y, k = 5, seed = 123) {
  set.seed(seed)
  folds <- rep(NA_integer_, length(y))
  for (cls in levels(y)) {
    idx <- which(y == cls)
    folds[idx] <- sample(rep(1:k, length.out = length(idx)))
  }
  folds
}

# ------------------------------------------------------------
# CV ile hyperparameter tuning (SADECE train_df üzerinde)
# ------------------------------------------------------------

set.seed(123)
k <- 5
fold_id <- make_stratified_folds(train_df$writeoff, k = k, seed = 123)

# Küçük ama etkili grid (istersen genişletebilirsin)
grid <- expand.grid(
  cp       = c(0.001, 0.005, 0.01, 0.02),
  minsplit = c(50, 100, 200),
  minbucket= c(20, 50, 80),
  maxdepth = c(3, 5, 7),
  stringsAsFactors = FALSE
) %>%
  dplyr::filter(minbucket < minsplit)

eval_one_setting <- function(cp, minsplit, minbucket, maxdepth) {
  
  oof_prob <- rep(NA_real_, nrow(train_df))
  
  for (i in 1:k) {
    tr <- train_df[fold_id != i, , drop = FALSE]
    va <- train_df[fold_id == i, , drop = FALSE]
    
    fit <- rpart(
      writeoff ~ .,
      data = tr,
      method = "class",
      control = rpart.control(
        cp = cp,
        minsplit = minsplit,
        minbucket = minbucket,
        maxdepth = maxdepth,
        xval = 0
      )
    )
    
    # "yes" olasılığı
    pr <- predict(fit, newdata = va, type = "prob")[, "yes"]
    oof_prob[fold_id == i] <- pr
  }
  
  # OOF üzerinden threshold seç (F1 maks.)
  th <- best_threshold_by_f1(train_df$writeoff, oof_prob)
  
  pred_oof <- factor(ifelse(oof_prob >= th, "yes", "no"), levels = c("no","yes"))
  cm_oof <- table(truth = train_df$writeoff, pred = pred_oof)
  m <- metrics_from_cm(cm_oof)
  
  tibble(
    cp = cp, minsplit = minsplit, minbucket = minbucket, maxdepth = maxdepth,
    chosen_threshold = th,
    oof_f1 = m$f1,
    oof_accuracy = m$accuracy
  )
}

tuning_list <- vector("list", nrow(grid))
for (g in 1:nrow(grid)) {
  tuning_list[[g]] <- eval_one_setting(
    cp = grid$cp[g],
    minsplit = grid$minsplit[g],
    minbucket = grid$minbucket[g],
    maxdepth = grid$maxdepth[g]
  )
}

tuning_results <- bind_rows(tuning_list) %>%
  arrange(desc(oof_f1), desc(oof_accuracy))

best_setting <- tuning_results %>% slice(1)

best_cp       <- best_setting$cp
best_minsplit <- best_setting$minsplit
best_minbucket<- best_setting$minbucket
best_maxdepth <- best_setting$maxdepth
chosen_threshold <- best_setting$chosen_threshold

# ------------------------------------------------------------
# Best hyperparam ile TRAIN'de eğit, TEST'te değerlendir
# ------------------------------------------------------------

dt_fit <- rpart(
  writeoff ~ .,
  data = train_df,
  method = "class",
  control = rpart.control(
    cp = best_cp,
    minsplit = best_minsplit,
    minbucket = best_minbucket,
    maxdepth = best_maxdepth,
    xval = 0
  )
)

# Train acc (gap için)
prob_train <- predict(dt_fit, newdata = train_df, type = "prob")[, "yes"]
pred_train <- factor(ifelse(prob_train >= chosen_threshold, "yes", "no"), levels = c("no","yes"))
cm_train <- table(truth = train_df$writeoff, pred = pred_train)
train_acc <- metrics_from_cm(cm_train)$accuracy

# Test performansı
prob_test <- predict(dt_fit, newdata = test_df, type = "prob")[, "yes"]
pred_test <- factor(ifelse(prob_test >= chosen_threshold, "yes", "no"), levels = c("no","yes"))
cm_test <- table(truth = test_df$writeoff, pred = pred_test)
test_m <- metrics_from_cm(cm_test)
test_acc <- test_m$accuracy

# 2) Confusion Matrix (TEST)  [ÇIKTI]
cat("\nConfusion Matrix\n"); print(cm_test)

# 3) Metrikler (TEST) + chosen_threshold + best hyperparams  [ÇIKTI]
cat("\nMetrikler\n")
print(test_m %>%
        mutate(
          chosen_threshold = chosen_threshold,
          best_cp = best_cp,
          best_minsplit = best_minsplit,
          best_minbucket = best_minbucket,
          best_maxdepth = best_maxdepth
        ) %>%
        select(best_cp, best_minsplit, best_minbucket, best_maxdepth,
               chosen_threshold, everything()))

# 4) Train_acc, Test_acc, gap  [ÇIKTI]
cat("\nTrain/Test Accuracy & Gap\n")
print(tibble(
  train_acc = train_acc,
  test_acc  = test_acc,
  gap       = abs(train_acc - test_acc)
))

# ------------------------------------------------------------
# Final model (tüm Loans) + NewApplicants tahmini + CSV
# ------------------------------------------------------------

final_dt <- rpart(
  writeoff ~ .,
  data = loans %>% select(-row_id),
  method = "class",
  control = rpart.control(
    cp = best_cp,
    minsplit = best_minsplit,
    minbucket = best_minbucket,
    maxdepth = best_maxdepth,
    xval = 0
  )
)

new_prob <- predict(final_dt, newdata = newapps, type = "prob")[, "yes"]
new_pred <- ifelse(new_prob >= chosen_threshold, "yes", "no")

submission <- newapps %>%
  mutate(writeoff_prob = new_prob,
         writeoff_pred = new_pred)

write_csv(submission, "NewApplicants_Predictions_CART_Holdout_Tuned.csv")