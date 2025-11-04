############################################################
# BUS 445 – Pathway 2: Service Factors & Heterogeneity
# uses:
#   cell2celltrain.csv
#   cell2cellholdout.csv
#
# Outputs:
#   clean_service_vars.csv
#   model_coefficients.csv
#   holdout_predictions.csv
############################################################

## 1. LOAD DATA -------------------------------------------------
train <- read.csv("cell2celltrain.csv", stringsAsFactors = FALSE)
holdout <- read.csv("cell2cellholdout.csv", stringsAsFactors = FALSE)

## 2. DATA PREP & CLEANING --------------------------------------
# keep only service-related variables we care about
clean_df <- train[, c(
  "CustomerID",
  "Churn",
  "MonthlyRevenue",
  "MonthlyMinutes",
  "OverageMinutes",
  "PercChangeMinutes",
  "PercChangeRevenues",
  "DroppedCalls",
  "BlockedCalls",
  "DroppedBlockedCalls",
  "UnansweredCalls",
  "CustomerCareCalls",
  "MonthsInService"
)]

# make churn numeric 0/1 for modeling
clean_df$ChurnFlag <- ifelse(clean_df$Churn == "Yes", 1, 0)

# make network reliability
# if DroppedBlockedCalls exists, use it; else sum Dropped + Blocked
clean_df$NetworkReliability <- ifelse(
  !is.na(clean_df$DroppedBlockedCalls),
  clean_df$DroppedBlockedCalls,
  clean_df$DroppedCalls + clean_df$BlockedCalls
)

# save the clean modeling file
write.csv(clean_df, "clean_service_vars.csv", row.names = FALSE)

## 3. EDA (very simple) -----------------------------------------
# churn rate
print(table(clean_df$Churn))
print(prop.table(table(clean_df$Churn)))

# if you want quick histograms in RStudio:
hist(clean_df$CustomerCareCalls,
     main = "CustomerCareCalls",
     xlab = "Customer care calls")
hist(clean_df$NetworkReliability,
     main = "Network Reliability (dropped+blocked)",
     xlab = "Dropped/Blocked calls")
hist(clean_df$MonthsInService,
     main = "Months in Service",
     xlab = "MonthsInService")

## 4. MODELING ---------------------------------------------------
# pick a small, businessy set of predictors
model_data <- clean_df[, c(
  "ChurnFlag",
  "CustomerCareCalls",
  "NetworkReliability",
  "OverageMinutes",
  "MonthsInService",
  "MonthlyRevenue"
)]

# drop rows with missing in those cols
model_data <- na.omit(model_data)

# fit logistic regression
m <- glm(
  ChurnFlag ~ CustomerCareCalls +
    NetworkReliability +
    OverageMinutes +
    MonthsInService +
    MonthlyRevenue,
  data = model_data,
  family = binomial(link = "logit")
)

summary(m)

# coefficient table with odds ratios & 95% CI
coefs <- summary(m)$coefficients
odds  <- exp(coef(m))
ci    <- exp(confint(m))  # may print a message, that's fine

coef_table <- data.frame(
  Term = rownames(coefs),
  Estimate = coefs[, "Estimate"],
  StdError = coefs[, "Std. Error"],
  zvalue   = coefs[, "z value"],
  pvalue   = coefs[, "Pr(>|z|)"],
  OddsRatio = odds,
  CI_lower = ci[, 1],
  CI_upper = ci[, 2],
  row.names = NULL
)

write.csv(coef_table, "model_coefficients.csv", row.names = FALSE)

## 5. SCORE HOLDOUT ----------------------------------------------
# build the same fields on holdout
holdout$NetworkReliability <- ifelse(
  !is.na(holdout$DroppedBlockedCalls),
  holdout$DroppedBlockedCalls,
  holdout$DroppedCalls + holdout$BlockedCalls
)

holdout_model <- holdout[, c(
  "CustomerID",
  "CustomerCareCalls",
  "NetworkReliability",
  "OverageMinutes",
  "MonthsInService",
  "MonthlyRevenue"
)]

# some rows may have NAs; replace with 0 to allow prediction
holdout_model[is.na(holdout_model)] <- 0

# predict churn probability
holdout_model$PredictedChurnProb <- predict(
  m,
  newdata = holdout_model,
  type = "response"
)

# save
write.csv(
  holdout_model[, c("CustomerID", "PredictedChurnProb")],
  "holdout_predictions.csv",
  row.names = FALSE
)

cat("Done. Files written:\n",
    "- clean_service_vars.csv\n",
    "- model_coefficients.csv\n",
    "- holdout_predictions.csv\n")

############################################################
# EXTRA analysis on top of your working script
# assumes clean_service_vars.csv already exists
############################################################

df <- read.csv("clean_service_vars.csv", stringsAsFactors = FALSE)

# 1) churn-by-bins (this never breaks) ----------------------
df$CCC_bin <- cut(
  df$CustomerCareCalls,
  breaks = c(-Inf,0,1,2,3,5,Inf),
  labels = c("0","1","2","3","4-5","6+")
)
churn_by_ccc <- aggregate(ChurnFlag ~ CCC_bin, data = df, FUN = mean, na.rm = TRUE)
churn_by_ccc$ChurnPercent <- round(churn_by_ccc$ChurnFlag * 100, 1)
print(churn_by_ccc)

df$Net_bin <- cut(
  df$NetworkReliability,
  breaks = c(-Inf,0,1,3,5,10,Inf),
  labels = c("0","1","2-3","4-5","6-10","10+")
)
churn_by_net <- aggregate(ChurnFlag ~ Net_bin, data = df, FUN = mean, na.rm = TRUE)
churn_by_net$ChurnPercent <- round(churn_by_net$ChurnFlag * 100, 1)
print(churn_by_net)

df$TenureSeg <- ifelse(df$MonthsInService <= 12, "Short (<=12)", "Long (>12)")
churn_by_tenure <- aggregate(ChurnFlag ~ TenureSeg, data = df, FUN = mean, na.rm = TRUE)
churn_by_tenure$ChurnPercent <- round(churn_by_tenure$ChurnFlag * 100, 1)
print(churn_by_tenure)

# 2) build a bigger model ONLY with columns that really exist ----
base_vars <- c(
  "CustomerCareCalls",
  "NetworkReliability",
  "OverageMinutes",
  "MonthsInService",
  "MonthlyRevenue"
)

optional_vars <- c("PercChangeMinutes", "PercChangeRevenues", "CreditRating_num")

# keep only optional vars that are actually in the file AND not all NA
good_optional <- c()
for (v in optional_vars) {
  if (v %in% names(df)) {
    non_na <- sum(!is.na(df[[v]]))
    if (non_na > 100) {   # at least 100 non-missing rows = usable
      good_optional <- c(good_optional, v)
    }
  }
}

vars_to_use <- c("ChurnFlag", base_vars, good_optional)
vars_to_use <- intersect(vars_to_use, names(df))  # just in case

model_df <- df[, vars_to_use, drop = FALSE]
model_df <- na.omit(model_df)

if (nrow(model_df) < 100) {
  cat("\nNot enough complete rows to fit the bigger model with optional variables.\n")
  cat("I used the base 5 variables earlier — keep those.\n")
} else {
  # build model formula: churn ~ all other columns
  rhs <- paste(setdiff(names(model_df), "ChurnFlag"), collapse = " + ")
  form <- as.formula(paste("ChurnFlag ~", rhs))
  
  m_big <- glm(form, data = model_df, family = binomial())
  print(summary(m_big))
  
  # odds ratios
  coefs <- summary(m_big)$coefficients
  odds  <- exp(coef(m_big))
  ci    <- exp(confint(m_big))
  
  coef_table2 <- data.frame(
    Term = rownames(coefs),
    Estimate = coefs[, "Estimate"],
    StdError = coefs[, "Std. Error"],
    zvalue   = coefs[, "z value"],
    pvalue   = coefs[, "Pr(>|z|)"],
    OddsRatio = odds,
    CI_lower = ci[, 1],
    CI_upper = ci[, 2],
    row.names = NULL
  )
  write.csv(coef_table2, "model2_coefficients.csv", row.names = FALSE)
  cat("\nWrote model2_coefficients.csv with the extra variables that actually existed.\n")
}

cat("\nExtra summaries you can talk about:\n",
    "- churn_by_ccc (service dissatisfaction)\n",
    "- churn_by_net (network issues)\n",
    "- churn_by_tenure (early customers churn more)\n")


