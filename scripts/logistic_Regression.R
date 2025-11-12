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




# --- SEGMENT MODEL: Heterogeneity by Credit Rating -------------

# Make sure interaction variable exists
train <- train %>%
  mutate(Care_Credit_Interaction = CustomerCareCalls * CreditRating_num)

# Fit logistic regression with interaction
logit_segment <- glm(
  Churn_flag ~ CustomerCareCalls + DroppedBlockedCalls + OverageMinutes +
    PercChangeMinutes + PercChangeRevenues + MonthsInService +
    CreditRating_num + Care_Credit_Interaction,
  data = train,
  family = binomial
)

summary(logit_segment)

# Get odds ratios and 95% CIs
library(broom)
coef_seg <- tidy(logit_segment, conf.int = TRUE, exponentiate = TRUE) %>%
  rename(OR = estimate, CI_low = conf.low, CI_high = conf.high, p_value = p.value) %>%
  arrange(desc(abs(log(OR))))

coef_seg

# --- EXTENDED INTERACTION MODEL ------------------------------
library(broom)

train <- train %>%
  mutate(
    Care_Credit_Interaction = CustomerCareCalls * CreditRating_num,
    Quality_Credit_Interaction = DroppedBlockedCalls * CreditRating_num,
    Care_Tenure_Interaction = CustomerCareCalls * MonthsInService
  )

logit_multi <- glm(
  Churn_flag ~ CustomerCareCalls + DroppedBlockedCalls + OverageMinutes +
    PercChangeMinutes + PercChangeRevenues + MonthsInService + CreditRating_num +
    Care_Credit_Interaction + Quality_Credit_Interaction + Care_Tenure_Interaction,
  data = train,
  family = binomial
)

summary(logit_multi)

coef_multi <- tidy(logit_multi, conf.int = TRUE, exponentiate = TRUE) %>%
  rename(OR = estimate, CI_low = conf.low, CI_high = conf.high, p_value = p.value) %>%
  arrange(desc(abs(log(OR))))

coef_multi


library(sjPlot)
plot_model(logit_multi, type = "int", terms = c("CustomerCareCalls", "CreditRating_num"))
plot_model(logit_multi, type = "int", terms = c("CustomerCareCalls", "MonthsInService"))




# --- PREDICTED CHURN PROBABILITIES ---------------------------

# Make sure model and data are loaded
# Model: logit_multi  (from earlier)
# Data: train

# Create scenario dataset (like Python version)
scenario_df <- expand.grid(
  CustomerCareCalls = c(0, 2, 5),             # none, moderate, high engagement
  DroppedBlockedCalls = median(train$DroppedBlockedCalls, na.rm = TRUE),
  OverageMinutes = median(train$OverageMinutes, na.rm = TRUE),
  PercChangeMinutes = median(train$PercChangeMinutes, na.rm = TRUE),
  PercChangeRevenues = median(train$PercChangeRevenues, na.rm = TRUE),
  MonthsInService = c(3, 12, 36),             # new, mid, long-term
  CreditRating_num = c(2, 5, 7)               # good, medium, poor credit
)

# Add interaction terms
scenario_df <- scenario_df %>%
  mutate(
    Care_Credit_Interaction = CustomerCareCalls * CreditRating_num,
    Quality_Credit_Interaction = DroppedBlockedCalls * CreditRating_num,
    Care_Tenure_Interaction = CustomerCareCalls * MonthsInService
  )

# Predict churn probabilities
scenario_df$Predicted_Churn_Prob <- predict(logit_multi, newdata = scenario_df, type = "response")

# Round and print results in pivot style
library(tidyr)
pivot_result <- scenario_df %>%
  select(CreditRating_num, MonthsInService, CustomerCareCalls, Predicted_Churn_Prob) %>%
  pivot_wider(names_from = CustomerCareCalls, values_from = Predicted_Churn_Prob)

print(pivot_result, n = 20)


ggplot(scenario_df, aes(x = CustomerCareCalls, y = Predicted_Churn_Prob,
                        color = factor(CreditRating_num), group = CreditRating_num)) +
  geom_line(size = 1.1) +
  geom_point(size = 2) +
  facet_wrap(~MonthsInService) +
  scale_color_manual(values = c("darkgreen", "orange", "red"),
                     name = "Credit Rating\n(1=Best, 7=Worst)") +
  theme_minimal() +
  labs(title = "Predicted Churn Probability by Customer Care and Credit Rating",
       subtitle = "Faceted by Tenure",
       x = "Number of Customer Care Calls",
       y = "Predicted Probability of Churn")




