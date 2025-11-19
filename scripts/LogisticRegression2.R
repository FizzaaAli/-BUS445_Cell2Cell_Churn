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
library(ggplot2)
library(dplyr)
library(scales)


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


###############################################################
# --- 11. SEGMENTATION ----------------------------------------
###############################################################

# Add predictions to original data
data$pred_prob <- predict(model_interact, newdata = data, type = "response")
data$pred_class <- ifelse(data$pred_prob >= 0.30, "Churned", "Stayed")

# Create churn-risk segments
data$RiskSegment <- case_when(
  data$pred_prob >= 0.30 ~ "High",
  data$pred_prob >= 0.20 ~ "Medium",
  TRUE ~ "Low"
)

data$RiskSegment <- factor(data$RiskSegment, levels = c("Low","Medium","High"))

table(data$RiskSegment)
prop.table(table(data$RiskSegment))


###############################################################
# --- 12. Segment-Level Summary Stats -------------------------
###############################################################

segment_summary <- data %>%
  mutate(ChurnBinary = ifelse(Churn_flag == "Churned", 1, 0)) %>%
  group_by(RiskSegment) %>%
  summarise(
    Count = n(),
    Avg_Revenue = mean(MonthlyRevenue, na.rm = TRUE),
    Avg_Tenure = mean(MonthsInService, na.rm = TRUE),
    Avg_ChurnProb = mean(pred_prob, na.rm = TRUE),
    Actual_ChurnRate = mean(ChurnBinary, na.rm = TRUE),
    .groups = "drop"
  )

print(segment_summary)


###############################################################
# --- 13. Plot: Predicted Churn Probability by Segment --------
###############################################################

ggplot(segment_summary, aes(x = RiskSegment, y = Avg_ChurnProb, fill = RiskSegment)) +
  geom_col() +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  scale_fill_manual(values = c("Low" = "#6CC24A",
                               "Medium" = "#1E90FF",
                               "High" = "#FF6B6B")) +
  labs(
    title = "Predicted Churn Probability by Segment",
    x = "Risk Segment",
    y = "Predicted Churn Probability"
  ) +
  theme_minimal(base_size = 16)

###############################################################
# --- 14. Plot: Average Revenue by Segment --------------------
###############################################################

ggplot(segment_summary, aes(x = RiskSegment, y = Avg_Revenue, fill = RiskSegment)) +
  geom_col() +
  scale_fill_manual(values = c("Low" = "#6CC24A",
                               "Medium" = "#1E90FF",
                               "High" = "#FF6B6B")) +
  labs(
    title = "Average Monthly Revenue by Segment",
    x = "Risk Segment",
    y = "Revenue ($)"
  ) +
  theme_minimal(base_size = 16)






###############################################################
# --- MORE SEGMENTATION GRAPHS ----------------------------------------
###############################################################

# Add predicted churn probability to the ORIGINAL data
data$pred_prob <- predict(model_interact, newdata = data, type = "response")

# Classify each customer by predicted churn (for diagnostics, not final rule)
data$pred_class <- ifelse(data$pred_prob >= 0.30, "Churned", "Stayed")

# Create churn-risk segments based on predicted probability
data$RiskSegment <- case_when(
  data$pred_prob >= 0.30 ~ "High",
  data$pred_prob >= 0.20 ~ "Medium",
  TRUE ~ "Low"
)

# Order segments nicely
data$RiskSegment <- factor(data$RiskSegment, levels = c("Low", "Medium", "High"))

# Quick counts (for your notes)
table(data$RiskSegment)
prop.table(table(data$RiskSegment))


###############################################################
# --- SEGMENT-LEVEL SUMMARY STATS -------------------------
###############################################################

segment_summary <- data %>%
  mutate(ChurnBinary = ifelse(Churn_flag == "Churned", 1, 0)) %>%
  group_by(RiskSegment) %>%
  summarise(
    Count          = n(),
    Avg_Revenue    = mean(MonthlyRevenue, na.rm = TRUE),
    Avg_Tenure     = mean(MonthsInService,   na.rm = TRUE),
    Avg_ChurnProb  = mean(pred_prob,         na.rm = TRUE),
    Actual_ChurnRate = mean(ChurnBinary,     na.rm = TRUE),
    .groups = "drop"
  ) %>%
  # Expected revenue loss per segment
  mutate(
    Expected_Loss = Count * Avg_Revenue * Avg_Tenure * Avg_ChurnProb
  )

print(segment_summary)
# This should roughly match:
# Low:   ~1.1k customers, rev ~81, churn ~27%
# Medium: ~33k customers, rev ~54, churn ~26%
# High:  ~16.6k customers, rev ~67, churn ~33%


###############################################################
# --- PROFIT-AT-RISK MATRIX (BUBBLE PLOT) -----------------
# X = predicted churn risk, Y = avg revenue, size = expected loss
###############################################################

# Overall averages for reference lines
overall_risk <- weighted.mean(segment_summary$Avg_ChurnProb, segment_summary$Count)
overall_rev  <- weighted.mean(segment_summary$Avg_Revenue,   segment_summary$Count)

p_profit <- ggplot(segment_summary,
                   aes(x = Avg_ChurnProb,
                       y = Avg_Revenue,
                       size = Expected_Loss,
                       fill = RiskSegment,
                       label = RiskSegment)) +
  geom_hline(yintercept = overall_rev,  linetype = "dashed", color = "grey60") +
  geom_vline(xintercept = overall_risk, linetype = "dashed", color = "grey60") +
  geom_point(shape = 21, color = "white", alpha = 0.9) +
  geom_text(vjust = -1, fontface = "bold") +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1),
                     name   = "Average Predicted Churn Probability") +
  scale_y_continuous(name = "Average Monthly Revenue ($)") +
  scale_size_continuous(name = "Expected Revenue Loss",
                        labels = scales::dollar_format(accuracy = 1)) +
  scale_fill_manual(values = c("Low" = "#6CC24A",
                               "Medium" = "#1E90FF",
                               "High" = "#FF6B6B")) +
  labs(
    title = "Risk vs Value: Profit-at-Risk by Segment",
    subtitle = "High-Risk, High-Revenue customers sit in the top-right ‘danger’ zone"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "right")

p_profit


###############################################################
# --- DRIVER IMPORTANCE PLOT (ODDS RATIOS) ----------------
# Focus on the main actionable drivers from model_interact
###############################################################

# coef_table already exists from earlier:
# coef_table <- tidy(model_interact, conf.int = TRUE, exponentiate = TRUE) %>%
#   rename(OddsRatio = estimate, CI_low = conf.low, CI_high = conf.high)

driver_terms <- c(
  "OverageRatio",
  "UsageDeclineFlag",
  "CustomerCareCalls:CreditRating_num",
  "CreditRating_num:PercChangeMinutes",
  "OverageRatio:sqrt_Tenure"
)

driver_labels <- c(
  "OverageRatio"                    = "Overage ratio",
  "UsageDeclineFlag"                = "Usage decline > 20%",
  "CustomerCareCalls:CreditRating_num" = "Care calls × credit rating",
  "CreditRating_num:PercChangeMinutes" = "Usage change × credit rating",
  "OverageRatio:sqrt_Tenure"        = "Overage × tenure"
)

driver_df <- coef_table %>%
  filter(term %in% driver_terms) %>%
  mutate(
    Driver = driver_labels[term],
    Driver = factor(Driver, levels = rev(driver_labels))  # nice order top–down
  )

p_drivers <- ggplot(driver_df,
                    aes(x = Driver, y = OddsRatio)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey60") +
  geom_point(size = 3, color = "#1E90FF") +
  geom_errorbar(aes(ymin = CI_low, ymax = CI_high),
                width = 0.2, color = "#1E90FF") +
  coord_flip() +
  labs(
    title = "Key Drivers of Churn (Odds Ratios)",
    x     = "",
    y     = "Odds Ratio (log scale)"
  ) +
  scale_y_log10() +
  theme_minimal(base_size = 14)

p_drivers






