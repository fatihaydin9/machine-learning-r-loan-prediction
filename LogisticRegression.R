# ------------------------------------------------------------
# Logistic Regression (writeoff) - Stratified 80/20 Holdout
# ------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
  library(janitor)
})

# Veri okuma işleminin gerçekleştirilmesi
loans   <- readr::read_csv("Loans.csv", na = c("", "NA", "?", "NULL"))
newapps <- readr::read_csv("NewApplicants.csv", na = c("", "NA", "?", "NULL"))

# Target(hedef) değişken için seviyelerin ayarlanması
loans$writeoff   <- factor(loans$writeoff, levels = c("no", "yes"))
newapps$writeoff <- factor(newapps$writeoff, levels = levels(loans$writeoff))

cat_cols <- c("gender","marital_status","education","employ_status",
              "spouse_work","residential_status","loan_purpose","collateral")

num_cols <- c("age","nb_depend_child","yrs_current_job","yrs_employed",
              "net_income","spouse_income","yrs_current_address",
              "loan_amount","loan_length")

# Tip dönüşümlerinin gerçekleştirilmesi
loans[num_cols]   <- lapply(loans[num_cols], as.numeric)
newapps[num_cols] <- lapply(newapps[num_cols], as.numeric)

loans[cat_cols]   <- lapply(loans[cat_cols], as.factor)
newapps[cat_cols] <- lapply(newapps[cat_cols], as.factor)

# kolonların uyumunun sağlanması
for (col in cat_cols) {
  newapps[[col]] <- factor(newapps[[col]], levels = levels(loans[[col]]))
}

# Stratified 80/20 split (naive baseline: %54.7 no , %45.3 yes)
set.seed(123)
loans <- loans %>% mutate(row_id = row_number())

train_ids <- loans %>%
  group_by(writeoff) %>%
  slice_sample(prop = 0.8) %>%
  ungroup() %>%
  pull(row_id)

train_df <- loans %>% filter(row_id %in% train_ids) %>% select(-row_id)
test_df  <- loans %>% filter(!row_id %in% train_ids) %>% select(-row_id)

for (col in cat_cols) {
  test_df[[col]] <- factor(test_df[[col]], levels = levels(train_df[[col]]))
}

# Yardımcı fonksiyonlar
safe_div <- function(a, b) ifelse(b == 0, NA_real_, a / b)
metrics_from_cm <- function(cm) {
  acc  <- sum(diag(cm)) / sum(cm)
  prec <- safe_div(cm["yes","yes"], sum(cm[,"yes"]))
  rec  <- safe_div(cm["yes","yes"], sum(cm["yes",]))
  f1   <- ifelse(is.na(prec) | is.na(rec) | (prec + rec) == 0, NA_real_,
                 2 * prec * rec / (prec + rec))
  tibble(accuracy = acc, precision = prec, recall = rec, f1 = f1)
}

# F1'i maksimize eden threshold'u bul
best_threshold_by_f1 <- function(truth, prob_yes, thresholds = seq(0.1, 0.9, by = 0.01)) {
  truth <- factor(truth, levels = c("no","yes"))  # güvenli olsun
  
  f1_vals <- sapply(thresholds, function(t) {
    pred <- factor(ifelse(prob_yes >= t, "yes", "no"), levels = c("no","yes"))
    cm <- table(truth = truth, pred = pred)
    as.numeric(metrics_from_cm(cm)$f1)
  })
  
  if (all(is.na(f1_vals))) return(0.5)
  thresholds[which.max(f1_vals)]
}

# Model eğitiminin gerçekleştirilmesi
fit_log <- glm(writeoff ~ ., data = train_df, family = binomial)

# Threshold değeri (Precision/Recall Tradeoff için F1 'e bakarak optimum değer bulunur)
prob_train_tmp <- predict(fit_log, newdata = train_df, type = "response")
chosen_threshold <- best_threshold_by_f1(
  truth    = train_df$writeoff,
  prob_yes = prob_train_tmp,
  thresholds = seq(0.1, 0.9, by = 0.01)
)

# 1) Train/Test yes-no dağılımı (%)
train_dist_pct <- round(100 * prop.table(table(train_df$writeoff)), 2)
test_dist_pct  <- round(100 * prop.table(table(test_df$writeoff)),  2)

cat("\nTrain dağılımı (%)\n"); print(train_dist_pct)
cat("\nTest dağılımı (%)\n");  print(test_dist_pct)

# Eğitim veri setinin metrikleri
prob_train <- predict(fit_log, newdata = train_df, type = "response")
pred_train <- factor(ifelse(prob_train >= chosen_threshold, "yes", "no"), levels = c("no","yes"))
cm_train   <- table(truth = train_df$writeoff, pred = pred_train)
train_acc  <- metrics_from_cm(cm_train)$accuracy

# Test veri setinin metrikleri
prob_test <- predict(fit_log, newdata = test_df, type = "response")
pred_test <- factor(ifelse(prob_test >= chosen_threshold, "yes", "no"), levels = c("no","yes"))
cm_test   <- table(truth = test_df$writeoff, pred = pred_test)
test_m    <- metrics_from_cm(cm_test)
test_acc  <- test_m$accuracy

# 2) Confusion Matrix (karışıklık matrisi)
cat("\nConfusion Matrix\n"); print(cm_test)

# 3) Metriklerin konsolda gösterilmesi
cat("\nMetrikler\n")
test_metrics_out <- test_m %>% mutate(chosen_threshold = chosen_threshold) %>%
  select(chosen_threshold, everything())
print(test_metrics_out)

# 4) Train_acc, Test_acc, gap değerleri
cat("\nTrain/Test Accuracy & Gap\n")
print(tibble(
  train_acc = train_acc,
  test_acc  = test_acc,
  gap       = abs(train_acc - test_acc)
))

# ------------------------------------------------------------
# NewApplicants tahmini + CSV output
# ------------------------------------------------------------

final_fit <- glm(writeoff ~ ., data = loans %>% select(-row_id), family = binomial)

new_prob <- predict(final_fit, newdata = newapps, type = "response")
new_pred <- ifelse(new_prob >= chosen_threshold, "yes", "no")

submission <- newapps %>%
  mutate(writeoff_prob = new_prob,
         writeoff_pred = new_pred)

write_csv(submission, "NewApplicants_Predictions_Logistic_Holdout.csv")