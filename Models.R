# Paketler (yüklü değilse bir kere install.packages(...) yapılmalı)
library(tidyverse)
library(janitor)

# Verileri .csv dosyasından oku
# not: na= içine ?, boş string vb koyuyoruz (testte writeoff '?' olabiliyor)
loans <- readr::read_csv("Loans.csv", na = c("", "NA", "?", "NULL"))
newapps <- readr::read_csv("NewApplicants.csv", na = c("", "NA", "?", "NULL"))

# -------------------------- TİP DÖNÜŞÜMLERİ ---------------------------------- 

# Target
target <- "writeoff"

# Kategorik sütunlar (target hariç)
cat_cols <- c("gender","marital_status","education","employ_status",
              "spouse_work","residential_status","loan_purpose","collateral")

# Sayısal sütunlar
num_cols <- c("age","nb_depend_child","yrs_current_job","yrs_employed",
              "net_income","spouse_income","yrs_current_address",
              "loan_amount","loan_length")

# 1) Sayısalları numeric'e çevir
loans[num_cols]   <- lapply(loans[num_cols], as.numeric)
newapps[num_cols] <- lapply(newapps[num_cols], as.numeric)

# 2) Kategorikleri factor'a çevir
loans[cat_cols]   <- lapply(loans[cat_cols], as.factor)
newapps[cat_cols] <- lapply(newapps[cat_cols], as.factor)

# 3) Target (train) factor olsun; testte NA kalacak ama factor tipinde dursun
loans[[target]] <- as.factor(loans[[target]])
newapps[[target]] <- factor(newapps[[target]], levels = levels(loans[[target]]))

# 4) Kontrol
dim(loans); dim(newapps)
dim(loans); dim(newapps)
glimpse(loans)
glimpse(newapps)