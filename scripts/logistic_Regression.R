###############################################################
# BUS 445 – Storytelling with Data
# Case: Cell2Cell Churn Challenge
# Pathway 2: "The What" – Diagnosing Service Factors & Heterogeneity
# STEP 3: Logistic Regression Modeling (Week 9)
# Author: Fizza Ali
###############################################################


# --- 0. Setup ------------------------------------------------
library(tidyverse)
library(car)
library(pROC)
library(broom)
library(caret)


# --- 1. Load cleaned training data ---------------------------
train <- read.csv("data/clean_service_vars.csv")



# --- 2. Load and preprocess holdout --------------------------
holdout <- read.csv("data/cell2cellholdout.csv", stringsAsFactors = FALSE)

holdout <- holdout %>%
  mutate(Churn_flag = ifelse(tolower(trimws(Churn)) %in% c("yes","y","1","true"), 1, 0),
         CreditRating_num = as.numeric(str_extract(CreditRating, "\\d+"))) %>%
  select(all_of(names(train))) %>%
  mutate(across(-Churn_flag, as.numeric)) %>%
  drop_na()

# Confirm structure
identical(names(train), names(holdout))



# --- 3) Build logistic model on the train --------------------

logit_base <- glm(
  Churn_flag ~ CustomerCareCalls +
    DroppedBlockedCalls +
    MonthlyRevenue +
    MonthlyMinutes +
    OverageMinutes +
    PercChangeMinutes +
    PercChangeRevenues +
    MonthsInService,
  data = train,
  family = binomial
)


summary(logit_base)



# --- 4) # Multicollinearity check ---------------
vif_values <- vif(logit_base)
vif_values



# --- 5) Predict probabilities --------------------------

# Predict probabilities on training set
train$pred_prob <- predict(logit_base, newdata = train, type = "response")

# ROC curve and AUC
roc_train <- roc(train$Churn_flag, train$pred_prob)
auc_train <- auc(roc_train)

cat("AUC on training data:", round(auc_train, 3), "\n")

plot(roc_train, col = "blue", main = paste("ROC Curve (AUC =", round(auc_train, 3), ")"))

# Confusion matrix (using 0.5 cutoff)
train$pred_class <- ifelse(train$pred_prob > 0.5, 1, 0)
confusionMatrix(
  as.factor(train$pred_class),
  as.factor(train$Churn_flag),
  positive = "1"
)



# --- 6. Interpretability – Odds Ratios ------------------------
coef_tab <- tidy(logit_base, conf.int = TRUE, exponentiate = TRUE) %>%
  rename(OR = estimate, CI_low = conf.low, CI_high = conf.high, p_value = p.value)

coef_tab

coef_tab %>%
  arrange(desc(abs(log(OR)))) %>%
  print(n = Inf)


# --- 7. Predict on holdout (no true churn labels) -------------
holdout$pred_prob <- predict(logit_base, newdata = holdout, type = "response")

# Optionally create binary prediction (using same cutoff)
holdout$pred_class <- ifelse(holdout$pred_prob > 0.5, 1, 0)

# Save predictions for future evaluation
write.csv(holdout[, c("pred_prob", "pred_class")],
          "data/holdout_predictions.csv", row.names = FALSE)










