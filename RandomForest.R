# ------------------------------------------------------------
# Random Forest (writeoff) - Stratified 80/20 Holdout + CV Hyperparameter Tuning
# ------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
  library(janitor)
})

if (!requireNamespace("ranger", quietly = TRUE)) install.packages("ranger")
library(ranger)

# ------------------------------------------------------------
# Veri okuma + tip dönüşümü
# ------------------------------------------------------------

loans   <- readr::read_csv("Loans.csv", na = c("", "NA", "?", "NULL"), show_col_types = FALSE)
newapps <- readr::read_csv("NewApplicants.csv", na = c("", "NA", "?", "NULL"), show_col_types = FALSE)

loans$writeoff   <- factor(loans$writeoff, levels = c("no", "yes"))
newapps$writeoff <- factor(newapps$writeoff, levels = levels(loans$writeoff))

cat_cols <- c("gender","marital_status","education","employ_status",
              "spouse_work","residential_status","loan_purpose","collateral")

num_cols <- c("age","nb_depend_child","yrs_current_job","yrs_employed",
              "net_income","spouse_income","yrs_current_address",
              "loan_amount","loan_length")

loans[num_cols]   <- lapply(loans[num_cols], as.numeric)
newapps[num_cols] <- lapply(newapps[num_cols], as.numeric)

loans[cat_cols]   <- lapply(loans[cat_cols], as.factor)
newapps[cat_cols] <- lapply(newapps[cat_cols], as.factor)

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

for (col in cat_cols) {
  test_df[[col]] <- factor(test_df[[col]], levels = levels(train_df[[col]]))
}

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

best_threshold_by_f1 <- function(truth, prob_yes, thresholds = seq(0.1, 0.9, by = 0.01)) {
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

p <- ncol(train_df %>% select(-writeoff))

grid <- expand.grid(
  # klasik varsayılan başlangıç sqrt(p) 'dir; tune etmek için farklı değerler.
  mtry = unique(pmax(1, c(floor(sqrt(p)), floor(p/4), floor(p/3), floor(p/2), p))),
  # min.node.size aralığı; 1 “esnek”, 10 “daha kontrollü” seçenek verir.
  min.node.size = c(1, 5, 10),  
  # 0.632 standart değerdir; 0.8 de sıkça kullanılmaktadır.
  sample.fraction = c(0.632, 0.8), 
  # stringleri factor olarak okuma FALSE, gerekli çevrimler zaten yapılmıştı.
  stringsAsFactors = FALSE
)

# HIZ AYARLARI:
num_trees_tune  <- 400   # tuning sırasında hızlı
num_trees_final <- 1000   # final eğitim daha güçlü

# İstersen daha fazla thread kullan:
n_threads <- min(8, max(1, parallel::detectCores() - 1))

eval_one_setting <- function(mtry, min.node.size, sample.fraction) {
  
  oof_prob <- rep(NA_real_, nrow(train_df))
  
  for (i in 1:k) {
    tr <- train_df[fold_id != i, , drop = FALSE]
    va <- train_df[fold_id == i, , drop = FALSE]
    
    fit <- ranger(
      writeoff ~ .,
      data = tr,
      probability = TRUE,
      num.trees = num_trees_tune,
      mtry = mtry,
      min.node.size = min.node.size,
      sample.fraction = sample.fraction,
      num.threads = n_threads,
      seed = 123
    )
    
    oof_prob[fold_id == i] <- predict(fit, data = va)$predictions[, "yes"]
  }
  
  th <- best_threshold_by_f1(train_df$writeoff, oof_prob)
  
  pred_oof <- factor(ifelse(oof_prob >= th, "yes", "no"), levels = c("no","yes"))
  cm_oof <- table(truth = train_df$writeoff, pred = pred_oof)
  m <- metrics_from_cm(cm_oof)
  
  tibble(
    mtry = mtry,
    min.node.size = min.node.size,
    sample.fraction = sample.fraction,
    chosen_threshold = th,
    oof_f1 = m$f1
  )
}

# >>> pmap yerine progress görebileceğin loop <<<
tuning_list <- vector("list", nrow(grid))

for (g in 1:nrow(grid)) {
  cat(sprintf("\nTuning %d/%d  (mtry=%d, min.node.size=%d, sample.fraction=%.3f)\n",
              g, nrow(grid), grid$mtry[g], grid$min.node.size[g], grid$sample.fraction[g]))
  flush.console()
  
  tuning_list[[g]] <- eval_one_setting(
    mtry = grid$mtry[g],
    min.node.size = grid$min.node.size[g],
    sample.fraction = grid$sample.fraction[g]
  )
}

tuning_results <- bind_rows(tuning_list) %>% arrange(desc(oof_f1))
best_setting <- tuning_results %>% slice(1)

# ------------------------------------------------------------
# Best hyperparam ile TRAIN'de eğit, TEST'te değerlendir
# ------------------------------------------------------------

best_mtry   <- best_setting$mtry
best_min_ns <- best_setting$min.node.size
best_sfrac  <- best_setting$sample.fraction
chosen_threshold <- best_setting$chosen_threshold

rf_fit <- ranger(
  writeoff ~ .,
  data = train_df,
  probability = TRUE,
  num.trees = num_trees_final,
  mtry = best_mtry,
  min.node.size = best_min_ns,
  sample.fraction = best_sfrac,
  num.threads = n_threads,
  seed = 123
)

# İstenen çıktılar (lojistik şablon gibi)
cat("\nTrain dağılımı (%)\n")
print(round(100 * prop.table(table(train_df$writeoff)), 2))

cat("\nTest dağılımı (%)\n")
print(round(100 * prop.table(table(test_df$writeoff)), 2))

# Train/Test acc + gap (CM basmadan)
prob_train <- predict(rf_fit, data = train_df)$predictions[, "yes"]
pred_train <- factor(ifelse(prob_train >= chosen_threshold, "yes", "no"), levels = c("no","yes"))
cm_train <- table(truth = train_df$writeoff, pred = pred_train)
train_acc <- metrics_from_cm(cm_train)$accuracy

prob_test <- predict(rf_fit, data = test_df)$predictions[, "yes"]
pred_test <- factor(ifelse(prob_test >= chosen_threshold, "yes", "no"), levels = c("no","yes"))
cm_test <- table(truth = test_df$writeoff, pred = pred_test)
test_m <- metrics_from_cm(cm_test)
test_acc <- test_m$accuracy

cat("\nConfusion Matrix\n"); print(cm_test)

cat("\nMetrikler\n")
print(test_m %>%
        mutate(chosen_threshold = chosen_threshold,
               best_mtry = best_mtry,
               best_min_node = best_min_ns,
               best_sample_fraction = best_sfrac) %>%
        select(best_mtry, best_min_node, best_sample_fraction, chosen_threshold, everything()))

cat("\nTrain/Test Accuracy & Gap\n")
print(tibble(train_acc = train_acc, test_acc = test_acc, gap = abs(train_acc - test_acc)))

# ------------------------------------------------------------
# Final model (tüm Loans) + NewApplicants tahmini + CSV
# ------------------------------------------------------------

final_rf <- ranger(
  writeoff ~ .,
  data = loans %>% select(-row_id),
  probability = TRUE,
  num.trees = num_trees_final,
  mtry = best_mtry,
  min.node.size = best_min_ns,
  sample.fraction = best_sfrac,
  num.threads = n_threads,
  seed = 123
)

new_prob <- predict(final_rf, data = newapps)$predictions[, "yes"]
new_pred <- ifelse(new_prob >= chosen_threshold, "yes", "no")

submission <- newapps %>%
  mutate(writeoff_prob = new_prob,
         writeoff_pred = new_pred)

write_csv(submission, "NewApplicants_Predictions_RF_Holdout_Tuned.csv")
