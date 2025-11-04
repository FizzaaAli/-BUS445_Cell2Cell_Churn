############################################################
# Pathway 2 – focus on specific service/experience vars
# Vars we care about:
# - DroppedBlockedCalls
# - CustomerCareCalls
# - MonthlyRevenue
# - MonthlyMinutes
# - OverageMinutes
# - PercChangeMinutes
# - PercChangeRevenues
# - MonthsInService
# - CreditRating_num
# - Churn (target)
############################################################

## 1. LOAD DATA -------------------------------------------------
train <- read.csv("cell2celltrain.csv", stringsAsFactors = FALSE)
holdout <- read.csv("cell2cellholdout.csv", stringsAsFactors = FALSE)

## 2. PICK ONLY THE VARIABLES WE WANT ---------------------------
wanted <- c(
  "CustomerID",
  "Churn",
  "DroppedBlockedCalls",
  "CustomerCareCalls",
  "MonthlyRevenue",
  "MonthlyMinutes",
  "OverageMinutes",
  "PercChangeMinutes",
  "PercChangeRevenues",
  "MonthsInService",
  "CreditRating_num"
)

# keep only the ones that actually exist in your file
have <- intersect(wanted, names(train))
clean_df <- train[, have, drop = FALSE]

# make target numeric
clean_df$ChurnFlag <- ifelse(clean_df$Churn == "Yes", 1, 0)

# save the cleaned subset
write.csv(clean_df, "clean_service_vars_v2.csv", row.names = FALSE)

## 3. QUICK EDA --------------------------------------------------
# churn balance
cat("\n--- Churn balance ---\n")
print(table(clean_df$Churn))
print(prop.table(table(clean_df$Churn)))

# simple histograms (view in RStudio Plots pane)
if ("CustomerCareCalls" %in% names(clean_df)) {
  hist(clean_df$CustomerCareCalls,
       main = "CustomerCareCalls",
       xlab = "CustomerCareCalls")
}
if ("DroppedBlockedCalls" %in% names(clean_df)) {
  hist(clean_df$DroppedBlockedCalls,
       main = "DroppedBlockedCalls",
       xlab = "DroppedBlockedCalls")
}
if ("OverageMinutes" %in% names(clean_df)) {
  hist(clean_df$OverageMinutes,
       main = "OverageMinutes",
       xlab = "OverageMinutes")
}

## 4. MODELING ---------------------------------------------------
# predictors we WANT to use (in this order)
model_vars <- c(
  "DroppedBlockedCalls",
  "CustomerCareCalls",
  "MonthlyRevenue",
  "MonthlyMinutes",
  "OverageMinutes",
  "PercChangeMinutes",
  "PercChangeRevenues",
  "MonthsInService",
  "CreditRating_num"
)

# keep only those that really exist
model_vars <- intersect(model_vars, names(clean_df))

# build modeling data: target + predictors
model_data <- clean_df[, c("ChurnFlag", model_vars), drop = FALSE]

# drop rows with missing values in these columns
model_data <- na.omit(model_data)

# if we have at least 1 predictor, fit the model
if (ncol(model_data) > 1 && nrow(model_data) > 100) {
  
  # churn ~ all selected vars
  formula_txt <- paste("ChurnFlag ~", paste(model_vars, collapse = " + "))
  m <- glm(as.formula(formula_txt),
           data = model_data,
           family = binomial())
  
  cat("\n--- Logistic regression summary ---\n")
  print(summary(m))
  
  # coefficient table with odds ratios + CI
  coefs <- summary(m)$coefficients
  odds  <- exp(coef(m))
  ci    <- exp(confint(m))
  
  coef_table <- data.frame(
    Term      = rownames(coefs),
    Estimate  = coefs[, "Estimate"],
    StdError  = coefs[, "Std. Error"],
    zvalue    = coefs[, "z value"],
    pvalue    = coefs[, "Pr(>|z|)"],
    OddsRatio = odds,
    CI_lower  = ci[, 1],
    CI_upper  = ci[, 2],
    row.names = NULL
  )
  
  write.csv(coef_table, "model_v2_coefficients.csv", row.names = FALSE)
  
} else {
  stop("Not enough predictors or rows to fit the model. Check which of the chosen variables are actually in your CSV.")
}

## 5. SCORE HOLDOUT WITH THE SAME VARIABLES ----------------------
# make sure holdout has the new vars too
# (if not, create them as 0 so predict() won't break)
for (v in model_vars) {
  if (!v %in% names(holdout)) {
    holdout[[v]] <- 0
  }
}

# predict churn probability on holdout
holdout$PredictedChurnProb_v2 <- predict(
  m,
  newdata = holdout[, model_vars, drop = FALSE],
  type = "response"
)

# save only ID + prediction
if ("CustomerID" %in% names(holdout)) {
  out_holdout <- holdout[, c("CustomerID", "PredictedChurnProb_v2")]
} else {
  out_holdout <- data.frame(PredictedChurnProb_v2 = holdout$PredictedChurnProb_v2)
}

write.csv(out_holdout, "holdout_predictions_v2.csv", row.names = FALSE)

cat("\nDone. Files written:\n",
    "- clean_service_vars_v2.csv\n",
    "- model_v2_coefficients.csv\n",
    "- holdout_predictions_v2.csv\n")
