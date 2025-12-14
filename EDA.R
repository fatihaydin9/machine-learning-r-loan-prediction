# Paketler (yüklü değilse bir kere install.packages(...) yapılmalı)
library(tidyverse)
library(janitor)
library(dplyr)
library(tibble)

# Verileri .csv dosyasından oku
# not: na= içine ?, boş string vb koyuyoruz (testte writeoff '?' olabiliyor)
loans <- readr::read_csv("Loans.csv", na = c("", "NA", "?", "NULL"))
newapps <- readr::read_csv("NewApplicants.csv", na = c("", "NA", "?", "NULL"))

# Verinin doğru okunup okunmadığından emin olmak için glimpse kullanıyoruz.
dim(loans); dim(newapps)
glimpse(loans)
glimpse(newapps)

# Loans içerisinde eksik writeoff var mı?
sum(is.na(loans$writeoff))
table(loans$writeoff, useNA = "ifany")

# Veri seti içinde NA değerler var mı?
na_count <- sapply(loans, function(x) sum(is.na(x)))
sort(na_count, decreasing = TRUE)

# Nicel veri değerlerinin özeti
num_cols <- names(loans)[sapply(loans, is.numeric)]
summary(loans[, num_cols])

# Net Gelir 0 olanların hepsi işsiz (unemployed) mi?
loans %>%
  filter(net_income == 0) %>%
  count(employ_status, spouse_work, sort = TRUE) %>%
  head(10)

#boxplot(loan_amount ~ writeoff, data=loans, main="loan_amount by writeoff")
#boxplot(net_income ~ writeoff, data=loans, main="net_income by writeoff")
#boxplot(loan_length ~ writeoff, data=loans, main="loan_length by writeoff")

# Kontrol edeceğimiz kategorik değişkenler (target hariç)
cat_cols <- c("gender","marital_status","education","employ_status",
              "spouse_work","residential_status","loan_purpose","collateral")

# Kategorik değerlerin tutarlılığına bakılması
check_levels <- function(train, test, col) {
  train_vals <- unique(na.omit(as.character(train[[col]])))
  test_vals  <- unique(na.omit(as.character(test[[col]])))
  
  novel_in_test  <- setdiff(test_vals, train_vals)   # testte var, trainde yok
  missing_in_test <- setdiff(train_vals, test_vals)  # trainde var, testte yok
  
  tibble(
    variable = col,
    n_train_levels_seen = length(train_vals),
    n_test_levels_seen  = length(test_vals),
    n_novel_in_test     = length(novel_in_test),
    novel_in_test       = paste(novel_in_test, collapse = ", "),
    n_missing_in_test   = length(missing_in_test),
    missing_in_test     = paste(missing_in_test, collapse = ", ")
  )
}

level_report <- bind_rows(lapply(cat_cols, \(v) check_levels(loans, newapps, v)))
level_report %>% arrange(desc(n_novel_in_test), desc(n_missing_in_test))