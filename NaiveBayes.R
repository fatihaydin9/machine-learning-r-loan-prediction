# ------------------------------------------------------------
# Naive Bayes (writeoff) - Stratified 80/20 Holdout
# ------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
  library(janitor)
})

if (!requireNamespace("e1071", quietly = TRUE)) install.packages("e1071")
library(e1071)

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

# newapps level uyumu (garanti)
for (col in cat_cols) {
  newapps[[col]] <- factor(newapps[[col]], levels = levels(loans[[col]]))
}

# ------------------------------------------------------------
# Stratified 80/20 split
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

# 1) Train/Test yes-no dağılımı (%)
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

best_threshold_by_f1 <- function(truth, prob_yes, thresholds = seq(0.1, 0.90, by = 0.01)) {
  f1_vals <- sapply(thresholds, function(t) {
    pred <- factor(ifelse(prob_yes >= t, "yes", "no"), levels = c("no","yes"))
    cm <- table(truth = truth, pred = pred)
    metrics_from_cm(cm)$f1
  })
  thresholds[which.max(f1_vals)]
}

# ------------------------------------------------------------
# Basit imputasyon (Train/Test için)  -- NB NA sevmez, temizleyelim
# ------------------------------------------------------------

# Numeric: median (train üzerinden)
num_medians_train <- sapply(num_cols, function(cn) median(train_df[[cn]], na.rm = TRUE))
for (cn in num_cols) {
  train_df[[cn]][is.na(train_df[[cn]])] <- num_medians_train[[cn]]
  test_df[[cn]][is.na(test_df[[cn]])]   <- num_medians_train[[cn]]
}

# Categorical: NA -> "missing" (train seviyelerine göre)
for (cn in cat_cols) {
  train_df[[cn]] <- as.character(train_df[[cn]])
  test_df[[cn]]  <- as.character(test_df[[cn]])
  
  train_df[[cn]][is.na(train_df[[cn]])] <- "missing"
  test_df[[cn]][is.na(test_df[[cn]])]   <- "missing"
  
  train_df[[cn]] <- factor(train_df[[cn]])
  test_df[[cn]]  <- factor(test_df[[cn]], levels = levels(train_df[[cn]]))
}

# ------------------------------------------------------------
# Model: Naive Bayes
# ------------------------------------------------------------

fit_nb <- naiveBayes(writeoff ~ ., data = train_df, laplace = 1)

# Threshold seçimi (SADECE train üzerinde F1 maks.)
prob_train <- predict(fit_nb, newdata = train_df, type = "raw")[, "yes"]
chosen_threshold <- best_threshold_by_f1(train_df$writeoff, prob_train)

# Train acc (gap için)
pred_train <- factor(ifelse(prob_train >= chosen_threshold, "yes", "no"), levels = c("no","yes"))
cm_train <- table(truth = train_df$writeoff, pred = pred_train)
train_acc <- metrics_from_cm(cm_train)$accuracy

# Test performansı
prob_test <- predict(fit_nb, newdata = test_df, type = "raw")[, "yes"]
pred_test <- factor(ifelse(prob_test >= chosen_threshold, "yes", "no"), levels = c("no","yes"))
cm_test <- table(truth = test_df$writeoff, pred = pred_test)
test_m <- metrics_from_cm(cm_test)
test_acc <- test_m$accuracy

# 2) Confusion Matrix (TEST)
cat("\nConfusion Matrix\n"); print(cm_test)

# 3) Metrikler (TEST) + chosen_threshold
cat("\nMetrikler\n")
test_metrics_out <- test_m %>%
  mutate(chosen_threshold = chosen_threshold) %>%
  select(chosen_threshold, everything())
print(test_metrics_out)

# 4) Train_acc, Test_acc, gap
cat("\nTrain/Test Accuracy & Gap\n")
print(tibble(
  train_acc = train_acc,
  test_acc  = test_acc,
  gap       = abs(train_acc - test_acc)
))

# ------------------------------------------------------------
# Final model (tüm Loans) + NewApplicants tahmini + CSV
# ------------------------------------------------------------

loans_full <- loans %>% select(-row_id)

# Naive Bayes 
final_nb <- naiveBayes(writeoff ~ ., data = loans_full, laplace = 1)

# Tahmin (newapps içindeki writeoff kolonu varsa çıkar)
new_prob <- predict(final_nb, newdata = newapps %>% select(-writeoff), type = "raw")[, "yes"]
new_pred <- ifelse(new_prob >= chosen_threshold, "yes", "no")

submission <- newapps %>%
  mutate(writeoff_prob = new_prob,
         writeoff_pred = new_pred)

write_csv(submission, "NewApplicants_Predictions_NaiveBayes_Holdout.csv")