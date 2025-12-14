# --------------------------------------------------------------------------------
# kNN (writeoff) - Stratified 80/20 Holdout + Hızlı CV (k tuning) + Threshold(F1)
# --------------------------------------------------------------------------------

# Paket yüklemeleri (konsol çıktısını azaltmak için)
suppressPackageStartupMessages({
  library(tidyverse)
  library(janitor)
})

# kNN için hızlı komşu arama (FNN)
if (!requireNamespace("FNN", quietly = TRUE)) install.packages("FNN")
library(FNN)

# ------------------------------------------------------------
# Veri okuma + tip dönüşümü
# ------------------------------------------------------------

# Loans ve NewApplicants dosyalarını oku (NA mapleme dahil)
loans   <- readr::read_csv("Loans.csv", na = c("", "NA", "?", "NULL"), show_col_types = FALSE)
newapps <- readr::read_csv("NewApplicants.csv", na = c("", "NA", "?", "NULL"), show_col_types = FALSE)

# Target seviyelerini sabitle (no referans, yes olay)
loans$writeoff   <- factor(loans$writeoff, levels = c("no", "yes"))
newapps$writeoff <- factor(newapps$writeoff, levels = levels(loans$writeoff))

# Kategorik ve sayısal kolonların listesi
cat_cols <- c("gender","marital_status","education","employ_status",
              "spouse_work","residential_status","loan_purpose","collateral")
num_cols <- c("age","nb_depend_child","yrs_current_job","yrs_employed",
              "net_income","spouse_income","yrs_current_address",
              "loan_amount","loan_length")

# Sayısal kolonları numeric'e çevir
loans[num_cols]   <- lapply(loans[num_cols], as.numeric)
newapps[num_cols] <- lapply(newapps[num_cols], as.numeric)

# Kategorik kolonları factor'a çevir
loans[cat_cols]   <- lapply(loans[cat_cols], as.factor)
newapps[cat_cols] <- lapply(newapps[cat_cols], as.factor)

# NewApplicants factor seviyelerini Loans ile hizala (level mismatch hatasını önler)
for (col in cat_cols) {
  newapps[[col]] <- factor(newapps[[col]], levels = levels(loans[[col]]))
}

# ------------------------------------------------------------
# Stratified 80/20 Holdout (TEST dokunulmaz)
# ------------------------------------------------------------

# Tekrarlanabilir sonuç için seed
set.seed(123)

# Split için satır id ekle
loans <- loans %>% mutate(row_id = row_number())

# Stratified split: writeoff sınıfına göre %80 train örnekle
train_ids <- loans %>%
  group_by(writeoff) %>%
  slice_sample(prop = 0.8) %>%
  ungroup() %>%
  pull(row_id)

# Train/Test setlerini oluştur
train_df <- loans %>% filter(row_id %in% train_ids) %>% select(-row_id)
test_df  <- loans %>% filter(!row_id %in% train_ids) %>% select(-row_id)

# Test factor seviyelerini train ile hizala (garanti)
for (col in cat_cols) {
  test_df[[col]] <- factor(test_df[[col]], levels = levels(train_df[[col]]))
}

# ------------------------------------------------------------
# HIZ: Imputation yok -> NA satırları model girişinden çıkar
# ------------------------------------------------------------

# Modele girecek zorunlu kolonları tanımla (train/test için target dahil)
need_cols_train <- c("writeoff", cat_cols, num_cols)

# Tahmin yapılacak kolonlar (newapps için target yok)
need_cols_pred  <- c(cat_cols, num_cols)

# Train set: sadece gerekli kolonları al ve NA içeren satırları düşür
train_use <- train_df %>%
  select(all_of(need_cols_train)) %>%
  drop_na()

# Test set: sadece gerekli kolonları al ve NA içeren satırları düşür
test_use <- test_df %>%
  select(all_of(need_cols_train)) %>%
  drop_na()

# NewApplicants: NA varsa satırı düşürmek yerine ayrı gruba ayır (sonra NA tahmin basılacak)
newapps <- newapps %>% mutate(row_id = row_number())

# NewApplicants içinde eksiksiz satırlar (tahmin yapılabilir)
new_ok  <- newapps %>%
  select(row_id, all_of(need_cols_pred)) %>%
  drop_na()

# NewApplicants içinde eksik satırlar (tahmin NA bırakılacak)
new_bad <- newapps %>%
  select(row_id, all_of(need_cols_pred)) %>%
  filter(if_any(all_of(need_cols_pred), is.na))

# ------------------------------------------------------------
# Yardımcı fonksiyonlar (metrik + threshold + fold + ölçekleme)
# ------------------------------------------------------------

# 0'a bölme hatasını önlemek için güvenli bölme
safe_div <- function(a, b) ifelse(b == 0, NA_real_, a / b)

# Confusion matrix'ten accuracy/precision/recall/F1 üret
metrics_from_cm <- function(cm) {
  acc  <- sum(diag(cm)) / sum(cm)
  prec <- safe_div(cm["yes","yes"], sum(cm[,"yes"]))
  rec  <- safe_div(cm["yes","yes"], sum(cm["yes",]))
  f1   <- ifelse(is.na(prec) | is.na(rec) | (prec + rec) == 0, NA_real_,
                 2 * prec * rec / (prec + rec))
  tibble(accuracy = acc, precision = prec, recall = rec, f1 = f1)
}

# F1'i maksimize eden threshold'u bul (eldeki truth + prob_yes üzerinden)
best_threshold_by_f1 <- function(truth, prob_yes, thresholds = seq(0.1, 0.9, by = 0.01)) {
  f1_vals <- sapply(thresholds, function(t) {
    pred <- factor(ifelse(prob_yes >= t, "yes", "no"), levels = c("no","yes"))
    cm <- table(truth = truth, pred = pred)
    metrics_from_cm(cm)$f1
  })
  if (all(is.na(f1_vals))) return(0.5)
  thresholds[which.max(f1_vals)]
}

# Stratified K-fold üret (train içinde CV için)
make_stratified_folds <- function(y, k = 5, seed = 123) {
  set.seed(seed)
  folds <- rep(NA_integer_, length(y))
  for (cls in levels(y)) {
    idx <- which(y == cls)
    folds[idx] <- sample(rep(1:k, length.out = length(idx)))
  }
  folds
}

# Factor + numeric veriyi dummy değişkenlere çevirip matrise dönüştür (writeoff hariç)
to_x_matrix <- function(df) model.matrix(~ . - 1 - writeoff, data = df)

# Dışarıdan verilen center/scale ile standardize et (train istatistikleriyle)
scale_with <- function(X, center, scale) {
  sweep(sweep(X, 2, center, "-"), 2, scale, "/")
}

# ------------------------------------------------------------
# Model matrix üretimi + kolon hizalama
# ------------------------------------------------------------

# Train için dummy matrix üret (writeoff hedefi hariç)
X_train_raw <- to_x_matrix(train_use)

# Test için dummy matrix üret (writeoff hedefi hariç)
X_test_raw  <- model.matrix(~ . - 1 - writeoff, data = test_use)

# Train hedef vektörü
y_train <- train_use$writeoff

# Test dummy kolonlarını train kolonlarına göre hizala (aynı feature uzayı)
X_test_raw <- X_test_raw[, colnames(X_train_raw), drop = FALSE]

# NewApplicants dummy matrix (sadece NA'sız satırlar için) ve train kolonlarına hizalama
if (nrow(new_ok) > 0) {
  
  # NewApplicants (ok) için dummy matrix üret
  X_new_raw <- model.matrix(~ . - 1, data = new_ok %>% select(-row_id))
  
  # Train'de olup newapps'te olmayan kolonları 0 ile ekle
  miss_cols <- setdiff(colnames(X_train_raw), colnames(X_new_raw))
  if (length(miss_cols) > 0) {
    X_new_raw <- cbind(
      X_new_raw,
      matrix(0, nrow = nrow(X_new_raw), ncol = length(miss_cols),
             dimnames = list(NULL, miss_cols))
    )
  }
  
  # Kolon sırasını train ile aynı yap
  X_new_raw <- X_new_raw[, colnames(X_train_raw), drop = FALSE]
  
} else {
  X_new_raw <- NULL
}

# ------------------------------------------------------------
# CV ile best_k seçimi (kfold=5 + küçük k_grid) + OOF threshold
# ------------------------------------------------------------

# CV fold sayısı ve fold ataması (train içinde)
set.seed(123)
kfold <- 5
fold_id <- make_stratified_folds(y_train, k = kfold, seed = 123)

# Denenecek k değerleri (küçük grid = hızlı tuning)
k_grid <- c(9, 15, 30)

# Threshold aralığı (F1 maks. için aranacak)
thresholds <- seq(0.1, 0.9, by = 0.01)

# Tek bir k değeri için OOF olasılık üretip OOF F1 ölç
eval_one_k <- function(k_val) {
  
  # OOF olasılık vektörü (her satır kendi valid fold'undan gelir)
  oof_prob <- rep(NA_real_, nrow(X_train_raw))
  
  # K-fold CV döngüsü
  for (i in 1:kfold) {
    
    # Train/valid indekslerini ayır
    tr_idx <- which(fold_id != i)
    va_idx <- which(fold_id == i)
    
    # Fold train/valid matrislerini hazırla
    Xtr <- X_train_raw[tr_idx, , drop = FALSE]
    Xva <- X_train_raw[va_idx, , drop = FALSE]
    ytr <- y_train[tr_idx]
    
    # Ölçekleme istatistiklerini sadece fold train'den çıkar (leakage yok)
    ctr <- colMeans(Xtr)
    sdr <- apply(Xtr, 2, sd)
    sdr[sdr == 0] <- 1
    
    # Fold train/valid standardize et
    Xtr_s <- scale_with(Xtr, ctr, sdr)
    Xva_s <- scale_with(Xva, ctr, sdr)
    
    # Valid için komşu indekslerini bul (FNN hızlı)
    nn <- FNN::get.knnx(Xtr_s, Xva_s, k = k_val)$nn.index
    
    # Valid için "yes" olasılığını komşu oranı olarak hesapla
    oof_prob[va_idx] <- apply(nn, 1, function(ix) mean(ytr[ix] == "yes"))
  }
  
  # OOF olasılıklar üzerinden F1 maksimize eden threshold seç
  th <- best_threshold_by_f1(y_train, oof_prob, thresholds = thresholds)
  
  # Seçilen threshold ile OOF confusion matrix + F1 hesapla
  pred_oof <- factor(ifelse(oof_prob >= th, "yes", "no"), levels = c("no","yes"))
  cm_oof <- table(truth = y_train, pred = pred_oof)
  m <- metrics_from_cm(cm_oof)
  
  # k için özet sonuç
  tibble(best_k = k_val, chosen_threshold = th, oof_f1 = m$f1)
}

# k_grid üzerinde tuning sonuçlarını üret ve en iyi OOF F1'e göre sırala
tuning_results <- bind_rows(lapply(k_grid, eval_one_k)) %>% arrange(desc(oof_f1))

# En iyi k ve onun seçtiği threshold
best_setting <- tuning_results %>% slice(1)
best_k <- best_setting$best_k
chosen_threshold <- best_setting$chosen_threshold

# ------------------------------------------------------------
# Final: Train üzerinde kur -> Test üzerinde değerlendir
# ------------------------------------------------------------

# Tüm train üzerinde ölçekleme istatistiklerini hesapla
center_all <- colMeans(X_train_raw)
scale_all  <- apply(X_train_raw, 2, sd)
scale_all[scale_all == 0] <- 1

# Train ve test matrislerini aynı ölçekle standardize et
X_train <- scale_with(X_train_raw, center_all, scale_all)
X_test  <- scale_with(X_test_raw,  center_all, scale_all)

# Train accuracy (gap için): self-neighbor bias'ı azaltmak adına k+1 alıp kendini çıkar
nn_train <- FNN::get.knnx(X_train, X_train, k = best_k + 1)$nn.index
prob_train <- sapply(1:nrow(nn_train), function(i) {
  ix <- nn_train[i, ]
  ix <- ix[ix != i]
  ix <- ix[1:best_k]
  mean(y_train[ix] == "yes")
})

# Train sınıf tahmini + train accuracy
pred_train <- factor(ifelse(prob_train >= chosen_threshold, "yes", "no"), levels = c("no","yes"))
cm_train <- table(truth = y_train, pred = pred_train)
train_acc <- metrics_from_cm(cm_train)$accuracy

# Test için komşuları train'den bul, olasılık ve sınıf üret
nn_test <- FNN::get.knnx(X_train, X_test, k = best_k)$nn.index
prob_test <- apply(nn_test, 1, function(ix) mean(y_train[ix] == "yes"))
pred_test <- factor(ifelse(prob_test >= chosen_threshold, "yes", "no"), levels = c("no","yes"))

# Test confusion matrix + metrikler
cm_test <- table(truth = test_use$writeoff, pred = pred_test)
test_m <- metrics_from_cm(cm_test)
test_acc <- test_m$accuracy

# ------------------------------------------------------------
# İstenen çıktılar (sadece)
# ------------------------------------------------------------

# Train sınıf dağılımını yüzde olarak bas (stratified split kanıtı)
cat("\nTrain dağılımı (%)\n")
print(round(100 * prop.table(table(train_use$writeoff)), 2))

# Test sınıf dağılımını yüzde olarak bas (stratified split kanıtı)
cat("\nTest dağılımı (%)\n")
print(round(100 * prop.table(table(test_use$writeoff)), 2))

# Test confusion matrix
cat("\nConfusion Matrix\n")
print(cm_test)

# Test metrikleri (best_k + chosen_threshold dahil)
cat("\nMetrikler\n")
print(test_m %>%
        mutate(best_k = best_k, chosen_threshold = chosen_threshold) %>%
        select(best_k, chosen_threshold, everything()))

# Train/Test accuracy ve gap (overfitting kontrolü)
cat("\nTrain/Test Accuracy & Gap\n")
print(tibble(train_acc = train_acc, test_acc = test_acc, gap = abs(train_acc - test_acc)))

# ------------------------------------------------------------
# NewApplicants tahmini + CSV output
# ------------------------------------------------------------

# NA'sız NewApplicants satırları için standardize et + kNN olasılık üret
if (!is.null(X_new_raw) && nrow(new_ok) > 0) {
  
  # NewApplicants (ok) matrisini train ölçeğiyle standardize et
  X_new <- scale_with(X_new_raw, center_all, scale_all)
  
  # NewApplicants komşularını train'den bul
  nn_new <- FNN::get.knnx(X_train, X_new, k = best_k)$nn.index
  
  # NewApplicants için "yes" olasılığı (komşu oranı)
  new_prob_ok <- apply(nn_new, 1, function(ix) mean(y_train[ix] == "yes"))
  
  # NewApplicants sınıf tahmini (threshold ile)
  new_pred_ok <- ifelse(new_prob_ok >= chosen_threshold, "yes", "no")
  
  # NA'sız kısım çıktı tablosu
  submission_ok <- new_ok %>%
    mutate(writeoff_prob = new_prob_ok,
           writeoff_pred = new_pred_ok)
  
} else {
  
  # Tahmin edilebilir satır yoksa boş tibble üret
  submission_ok <- tibble(row_id = integer(),
                          writeoff_prob = numeric(),
                          writeoff_pred = character())
}

# NA'lı NewApplicants satırları için tahminleri NA bırak
submission_bad <- new_bad %>%
  mutate(writeoff_prob = NA_real_,
         writeoff_pred = NA_character_)

# Tahmin edilebilen + edilemeyen satırları birleştir ve orijinal sıraya koy
submission <- bind_rows(submission_ok, submission_bad) %>%
  arrange(row_id)

# row_id olmadan CSV output üret (ödev formatı için)
write_csv(submission %>% select(-row_id), "NewApplicants_Predictions_kNN_Holdout_Tuned.csv")
