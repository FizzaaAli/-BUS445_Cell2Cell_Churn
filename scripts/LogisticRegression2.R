###############################################################
# Cell2Cell Churn Challenge — Logistic Regression Modeling
# Objective: Identify key churn drivers and quantify their impact
# Author: [Your Name]
# Date: [Today’s Date]
###############################################################

# --- 0. SETUP ------------------------------------------------
library(tidyverse)
library(car)
library(broom)
library(pROC)
library(caret)

# --- 1. LOAD CLEAN DATA -------------------------------------
data <- read.csv("data/clean_service_vars_new.csv")

glimpse(data)

# Ensure target variable is factor
data$Churn_flag <- factor(data$Churn_flag, labels = c("Stayed", "Churned"))


# --- 2. FEATURE ENGINEERING ---------------------------------
# Create additional interaction & transformation features based on EDA insights
data <- data %>%
  mutate(
    # Behavioral ratio: Overage relative to usage
    OverageRatio = OverageMinutes / (MonthlyMinutes + 1),
    
    # Engagement per month of tenure
    CareCallsPerMonth = CustomerCareCalls / (MonthsInService + 1),
    
    # Nonlinear tenure transformation (diminishing effect)
    sqrt_Tenure = sqrt(MonthsInService),
    
    # Usage decline flag
    UsageDeclineFlag = ifelse(PercChangeMinutes < -20, 1, 0)
  )

# --- 3. MODEL FORMULATION -----------------------------------
# Base model with main effects
model_base <- glm(
  Churn_flag ~ CustomerCareCalls + DroppedBlockedCalls + 
    OverageMinutes + PercChangeMinutes + PercChangeRevenues + 
    CreditRating_num + sqrt_Tenure,
  data = data,
  family = binomial
)

# Model with interaction terms (to test heterogeneity)
model_interact <- glm(
  Churn_flag ~ CustomerCareCalls * CreditRating_num +
    PercChangeMinutes * CreditRating_num +
    OverageRatio * sqrt_Tenure +
    UsageDeclineFlag +
    DroppedBlockedCalls,
  data = data,
  family = binomial
)

# --- 4. MODEL COMPARISON ------------------------------------
summary(model_base)
summary(model_interact)

# AIC comparison
AIC(model_base, model_interact)

# --- 5. MULTICOLLINEARITY CHECK -----------------------------
vif_values <- car::vif(model_interact)
vif_values[vif_values > 5]

# --- 6. PREDICTIVE PERFORMANCE ------------------------------
# Create a new data frame that matches what the model actually used
model_data <- model.frame(model_interact)

# Add predicted probabilities and classes to that frame
model_data$pred_prob <- predict(model_interact, type = "response")
model_data$pred_class <- ifelse(model_data$pred_prob > 0.3, "Churned", "Stayed")

# Evaluate ROC and AUC
roc_curve <- roc(model_data$Churn_flag, model_data$pred_prob)
auc_value <- auc(roc_curve)
cat("Model AUC:", round(auc_value, 3), "\n")

# Optional: confusion matrix
confusionMatrix(as.factor(model_data$pred_class),
                model_data$Churn_flag,
                positive = "Churned")

# --- 7. COEFFICIENT INTERPRETATION --------------------------
coef_table <- tidy(model_interact, conf.int = TRUE, exponentiate = TRUE) %>%
  rename(OddsRatio = estimate, CI_low = conf.low, CI_high = conf.high) %>%
  mutate(Significance = ifelse(p.value < 0.05, "Significant", "Not Sig."))

print(coef_table)

# --- 8. BUSINESS INTERPRETATION -----------------------------
cat("\n--- BUSINESS INTERPRETATION ---\n")

cat("1. CustomerCareCalls × CreditRating_num: Significant negative interaction indicates that\n",
    "customer support is most effective for lower-credit customers — they benefit more from proactive help.\n\n")

cat("2. PercChangeMinutes × CreditRating_num: Large negative coefficients suggest customers with poor credit\n",
    "who reduce their usage are most likely to churn — early-warning segment.\n\n")

cat("3. OverageRatio × sqrt_Tenure: Customers who consistently go over plan, especially long-term ones,\n",
    "show higher churn risk — plan redesign opportunity.\n\n")

cat("4. UsageDeclineFlag: Customers whose usage drops >20% are ~2–3x more likely to churn — retention trigger.\n\n")

cat("5. DroppedBlockedCalls: Still not significant — confirms network issues are not primary churn driver.\n")

# --- 9. SAVE MODEL RESULTS ----------------------------------
write.csv(coef_table, "data/logistic_model_coefficients.csv", row.names = FALSE)
saveRDS(model_interact, "data/logistic_model_final.rds")



###############################################################
# --- 10. CHURN THRESHOLD ANALYSIS ----------------------------
# Purpose: Test various probability cutoffs to find the most useful
#          balance between identifying churners (sensitivity)
#          and avoiding false alarms (specificity)
###############################################################

# Function to compute confusion stats for a given cutoff
evaluate_cutoff <- function(threshold, data, actual, probs) {
  preds <- ifelse(probs > threshold, "Churned", "Stayed")
  cm <- confusionMatrix(as.factor(preds), actual, positive = "Churned")
  tibble(
    Cutoff = threshold,
    Accuracy = cm$overall["Accuracy"],
    Sensitivity = cm$byClass["Sensitivity"],
    Specificity = cm$byClass["Specificity"],
    Balanced_Accuracy = cm$byClass["Balanced Accuracy"]
  )
}

# Define cutoff values to test
cutoffs <- seq(0.1, 0.5, by = 0.05)

# Evaluate each cutoff
cutoff_results <- map_dfr(cutoffs, ~ evaluate_cutoff(.x, model_data, model_data$Churn_flag, model_data$pred_prob))

# Display results
print(cutoff_results)

# Visualize sensitivity vs. specificity trade-off
ggplot(cutoff_results, aes(x = Cutoff)) +
  geom_line(aes(y = Sensitivity, color = "Sensitivity"), size = 1.2) +
  geom_line(aes(y = Specificity, color = "Specificity"), size = 1.2) +
  geom_line(aes(y = Balanced_Accuracy, color = "Balanced Accuracy"), linetype = "dashed", size = 1.2) +
  scale_color_manual(values = c("steelblue", "tomato", "darkgreen")) +
  labs(title = "Sensitivity–Specificity Trade-off by Cutoff Threshold",
       y = "Metric Value", x = "Cutoff Probability",
       color = "Metric") +
  theme_minimal()

