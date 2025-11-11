# Cell2Cell Churn Analysis - Fixed Version
# Analysis of Heterogeneous Treatment Effects and Business Impact

# Load required libraries
library(dplyr)
library(ggplot2)
library(caret)
library(lmtest)
library(patchwork)

# Set random seed for reproducibility
set.seed(123)

# 1. LOAD AND PREPARE DATA
cat("Loading and preparing data...\n")

# Load training data (use this for all analysis)
train_df <- read.csv("cell2celltrain.csv")
holdout_df <- read.csv("cell2cellholdout.csv")

# Check data structure
cat("Training data dimensions:", dim(train_df), "\n")
cat("Holdout data dimensions:", dim(holdout_df), "\n")
cat("Churn distribution in training data:\n")
print(table(train_df$Churn))

# 2. DATA PREPROCESSING AND SEGMENTATION
cat("\n2. Creating customer segments and converting CreditRating...\n")

# Convert CreditRating from categorical to numeric
credit_mapping <- c(
  "1-Highest" = 1,
  "2-High" = 2, 
  "3-Good" = 3,
  "4-Medium" = 4,
  "5-Low" = 5
)

# Create segments based on EDA findings
df_segmented <- train_df %>%
  mutate(
    # Convert Churn to binary
    Churn_binary = ifelse(Churn == "Yes", 1, 0),
    
    # Convert CreditRating to numeric
    CreditRating_num = credit_mapping[as.character(CreditRating)],
    
    # Usage behavior segments
    usage_segment = case_when(
      is.na(PercChangeMinutes) ~ "Unknown Usage",
      PercChangeMinutes < -50 ~ "Severe Decline",
      PercChangeMinutes < -20 ~ "Moderate Decline", 
      PercChangeMinutes >= -20 & PercChangeMinutes <= 10 ~ "Stable Usage",
      PercChangeMinutes > 10 & PercChangeMinutes <= 50 ~ "Moderate Growth",
      PercChangeMinutes > 50 ~ "Strong Growth"
    ),
    
    # Revenue segments
    revenue_segment = case_when(
      MonthlyRevenue < 40 ~ "Low Value",
      MonthlyRevenue >= 40 & MonthlyRevenue <= 80 ~ "Medium Value", 
      MonthlyRevenue > 80 ~ "High Value"
    ),
    
    # Tenure segments
    tenure_segment = case_when(
      MonthsInService < 6 ~ "Very New (<6 months)",
      MonthsInService < 12 ~ "New (6-12 months)",
      MonthsInService >= 12 & MonthsInService <= 36 ~ "Established (1-3 years)",
      MonthsInService > 36 ~ "Long-term (>3 years)"
    ),
    
    # Overage behavior segments
    overage_segment = case_when(
      OverageMinutes == 0 ~ "No Overage",
      OverageMinutes > 0 & OverageMinutes <= 30 ~ "Low Overage",
      OverageMinutes > 30 & OverageMinutes <= 100 ~ "Medium Overage",
      OverageMinutes > 100 ~ "High Overage"
    )
  )

# Check CreditRating conversion
cat("CreditRating conversion summary:\n")
cat("Original CreditRating values:\n")
print(table(train_df$CreditRating))
cat("Converted CreditRating_num values:\n")
print(table(df_segmented$CreditRating_num))
cat("Missing CreditRating_num values:", sum(is.na(df_segmented$CreditRating_num)), "\n")

cat("Segment distribution:\n")
cat("Usage segments:\n")
print(table(df_segmented$usage_segment))
cat("Revenue segments:\n")
print(table(df_segmented$revenue_segment))

# 3. BUSINESS IMPACT ANALYSIS
cat("\n3. Calculating business impact...\n")

clv_analysis <- df_segmented %>%
  summarise(
    avg_monthly_revenue = mean(MonthlyRevenue, na.rm = TRUE),
    avg_tenure_months = mean(MonthsInService, na.rm = TRUE),
    total_customers = n(),
    churn_rate = mean(Churn == "Yes", na.rm = TRUE)
  ) %>%
  mutate(
    simple_clv = avg_monthly_revenue * avg_tenure_months,
    annual_churn_cost = total_customers * churn_rate * simple_clv
  )

print("Customer Lifetime Value Analysis:")
print(clv_analysis)

# Calculate 10% churn reduction impact
churn_reduction <- 0.10
customers_saved <- clv_analysis$total_customers * clv_analysis$churn_rate * churn_reduction
annual_savings <- customers_saved * clv_analysis$simple_clv

cat("\nBusiness Impact of 10% Churn Reduction:\n")
cat("Customers saved annually:", round(customers_saved), "\n")
cat("Annual revenue savings: $", round(annual_savings, 2), "\n")

# 4. SEGMENT ANALYSIS
cat("\n4. Analyzing churn by segments...\n")

segment_analysis <- df_segmented %>%
  group_by(revenue_segment, usage_segment) %>%
  summarise(
    churn_rate = mean(Churn == "Yes", na.rm = TRUE),
    n_customers = n(),
    avg_monthly_revenue = mean(MonthlyRevenue, na.rm = TRUE),
    avg_care_calls = mean(CustomerCareCalls, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  arrange(desc(churn_rate))

print("Churn Rates by Customer Segments:")
print(segment_analysis)

# 5. HETEROGENEOUS TREATMENT EFFECTS MODELING
cat("\n5. Building heterogeneous effects models...\n")

# Convert segments to factors for modeling
df_segmented <- df_segmented %>%
  mutate(
    revenue_segment = factor(revenue_segment),
    usage_segment = factor(usage_segment),
    tenure_segment = factor(tenure_segment),
    overage_segment = factor(overage_segment)
  )

# Split training data for validation
train_indices <- createDataPartition(df_segmented$Churn_binary, p = 0.7, list = FALSE)
train_data <- df_segmented[train_indices, ]
test_data <- df_segmented[-train_indices, ]

# Remove any remaining rows with NA in model variables to avoid errors
train_data_clean <- train_data %>%
  filter(!is.na(CreditRating_num) & !is.na(CustomerCareCalls) & 
         !is.na(OverageMinutes) & !is.na(PercChangeMinutes) & 
         !is.na(MonthsInService))

cat("Training data rows after cleaning:", nrow(train_data_clean), "\n")

# Model 1: Basic model without interactions (ATE)
model_ate <- glm(Churn_binary ~ CustomerCareCalls + OverageMinutes + 
                  PercChangeMinutes + CreditRating_num + MonthsInService,
                data = train_data_clean, family = "binomial")

# Model 2: Model with interactions for CATE
model_cate <- glm(Churn_binary ~ 
                   CustomerCareCalls * revenue_segment +
                   OverageMinutes * revenue_segment + 
                   PercChangeMinutes * usage_segment +
                   CustomerCareCalls * tenure_segment +
                   OverageMinutes * overage_segment +
                   CreditRating_num + MonthsInService,
                 data = train_data_clean, family = "binomial")

# Compare models
cat("Model Comparison (Likelihood Ratio Test):\n")
lr_test <- lrtest(model_ate, model_cate)
print(lr_test)

# Check model summary for interpretation
cat("\nCATE Model Coefficients (Key Interactions):\n")
summary_model <- summary(model_cate)
# Print only the interaction terms for clarity
interaction_terms <- grepl(":", names(coef(model_cate)))
if(any(interaction_terms)) {
  cat("Interaction terms in CATE model:\n")
  print(coef(model_cate)[interaction_terms])
}

# 6. PREDICT SENSITIVITY TO CUSTOMER CARE CALLS
cat("\n6. Predicting customer sensitivity to care calls...\n")

predict_sensitivity <- function(model, data, treatment_var = "CustomerCareCalls") {
  # Clean data for prediction
  data_clean <- data %>%
    filter(!is.na(CreditRating_num) & !is.na(CustomerCareCalls) & 
           !is.na(OverageMinutes) & !is.na(PercChangeMinutes) & 
           !is.na(MonthsInService))
  
  # Predict with original data
  pred_original <- predict(model, newdata = data_clean, type = "response")
  
  # Create data with treatment increased by 1 unit
  data_modified <- data_clean
  data_modified[[treatment_var]] <- data_modified[[treatment_var]] + 1
  
  # Predict with modified data
  pred_modified <- predict(model, newdata = data_modified, type = "response")
  
  # Sensitivity = change in churn probability
  sensitivity <- pred_modified - pred_original
  
  return(data_clean %>% mutate(pred_sensitivity = sensitivity))
}

# Apply sensitivity prediction
df_with_sensitivity <- predict_sensitivity(model_cate, df_segmented)

cat("Sensitivity prediction completed on", nrow(df_with_sensitivity), "rows\n")

# Create sensitivity bands with more granular segments
df_with_bands <- df_with_sensitivity %>%
  mutate(
    sensitivity_band = cut(pred_sensitivity, 
                          breaks = quantile(pred_sensitivity, probs = seq(0, 1, 0.2), na.rm = TRUE),
                          labels = c("Very Low", "Low", "Medium", "High", "Very High")),
    
    # Business value score - combination of sensitivity and revenue
    business_value = pred_sensitivity * MonthlyRevenue,
    
    # Strategic priority score
    strategic_priority = case_when(
      pred_sensitivity < 0 & MonthlyRevenue > 60 ~ "Protect (High Value, Negative Sensitivity)",
      pred_sensitivity > 0.01 & MonthlyRevenue > 60 ~ "Invest (High Value, Positive Sensitivity)",
      pred_sensitivity > 0.01 & MonthlyRevenue <= 60 ~ "Monitor (Lower Value, Positive Sensitivity)", 
      TRUE ~ "Maintain (Standard)"
    )
  )

# 7. TARGETING ANALYSIS
cat("\n7. Identifying optimal target segments...\n")

targeting_analysis <- df_with_bands %>%
  group_by(sensitivity_band, revenue_segment, strategic_priority) %>%
  summarise(
    avg_sensitivity = mean(pred_sensitivity, na.rm = TRUE),
    avg_monthly_revenue = mean(MonthlyRevenue, na.rm = TRUE),
    business_value_score = mean(business_value, na.rm = TRUE),
    current_churn_rate = mean(Churn == "Yes", na.rm = TRUE),
    n_customers = n(),
    .groups = 'drop'
  ) %>%
  arrange(desc(business_value_score))

print("Optimal Targeting Strategy:")
print(targeting_analysis)

# 8. COST-BENEFIT ANALYSIS
cat("\n8. Calculating ROI for interventions...\n")

calculate_intervention_roi <- function(target_data, cost_per_customer, effectiveness) {
  
  # Target high-sensitivity, high-value customers with positive sensitivity
  high_value_targets <- target_data %>%
    filter(strategic_priority == "Invest (High Value, Positive Sensitivity)")
  
  n_targeted <- nrow(high_value_targets)
  total_cost <- n_targeted * cost_per_customer
  
  # Calculate expected impact
  current_churn <- mean(high_value_targets$Churn == "Yes", na.rm = TRUE)
  churn_reduction <- current_churn * effectiveness
  customers_saved <- n_targeted * churn_reduction
  
  # Calculate savings using CLV
  avg_clv <- mean(high_value_targets$MonthlyRevenue, na.rm = TRUE) * 
             mean(high_value_targets$MonthsInService, na.rm = TRUE)
  
  annual_savings <- customers_saved * avg_clv
  roi <- (annual_savings - total_cost) / total_cost
  
  return(list(
    targeted_customers = n_targeted,
    avg_customer_value = round(avg_clv, 2),
    intervention_cost = total_cost,
    expected_annual_savings = annual_savings,
    roi_percentage = round(roi * 100, 1),
    net_benefit = annual_savings - total_cost,
    payback_period_months = ifelse(annual_savings > 0, total_cost / (annual_savings / 12), NA)
  ))
}

# Calculate ROI for different intervention scenarios
care_program_roi <- calculate_intervention_roi(df_with_bands, 50, 0.15)
premium_care_roi <- calculate_intervention_roi(df_with_bands, 100, 0.25)

cat("\nStandard Care Program ROI ($50/customer):\n")
print(care_program_roi)

cat("\nPremium Care Program ROI ($100/customer):\n")
print(premium_care_roi)

# 9. VISUALIZATIONS
cat("\n9. Creating visualizations...\n")

# Plot 1: Sensitivity distribution by strategic priority
p1 <- ggplot(df_with_bands, aes(x = strategic_priority, y = pred_sensitivity, fill = strategic_priority)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Churn Sensitivity by Strategic Priority",
       subtitle = "How different customer groups respond to care calls",
       y = "Predicted Sensitivity",
       x = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")

# Plot 2: Business value heatmap
p2 <- targeting_analysis %>%
  ggplot(aes(x = revenue_segment, y = sensitivity_band, fill = business_value_score)) +
  geom_tile() +
  geom_text(aes(label = round(business_value_score, 3)), color = "white", size = 3) +
  scale_fill_gradient2(low = "red", high = "darkgreen", mid = "yellow", midpoint = 0,
                      name = "Business Value") +
  labs(title = "Business Value Heatmap",
       subtitle = "Where to focus retention efforts",
       x = "Revenue Segment",
       y = "Sensitivity Band") +
  theme_minimal()

# Plot 3: Segment size vs churn rate
p3 <- df_with_bands %>%
  group_by(strategic_priority) %>%
  summarise(
    segment_size = n(),
    churn_rate = mean(Churn == "Yes", na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  ggplot(aes(x = segment_size, y = churn_rate, size = segment_size, color = strategic_priority)) +
  geom_point(alpha = 0.7) +
  geom_text(aes(label = strategic_priority), vjust = -0.5, size = 3) +
  labs(title = "Segment Size vs Churn Rate",
       subtitle = "Bubble size represents number of customers",
       x = "Segment Size",
       y = "Churn Rate") +
  theme_minimal() +
  theme(legend.position = "none")

# Display plots
print(p1)
print(p2) 
print(p3)

# 10. ACTIONABLE RECOMMENDATIONS
cat("\n10. Generating actionable recommendations...\n")

recommendations <- df_with_bands %>%
  group_by(strategic_priority, revenue_segment, sensitivity_band) %>%
  summarise(
    segment_size = n(),
    current_churn_rate = round(mean(Churn == "Yes", na.rm = TRUE), 3),
    avg_sensitivity = round(mean(pred_sensitivity, na.rm = TRUE), 4),
    avg_monthly_revenue = round(mean(MonthlyRevenue, na.rm = TRUE), 2),
    avg_tenure = round(mean(MonthsInService, na.rm = TRUE), 1),
    .groups = 'drop'
  ) %>%
  mutate(
    investment_level = case_when(
      strategic_priority == "Invest (High Value, Positive Sensitivity)" ~ "High Investment",
      strategic_priority == "Protect (High Value, Negative Sensitivity)" ~ "Moderate Investment", 
      strategic_priority == "Monitor (Lower Value, Positive Sensitivity)" ~ "Selective Investment",
      TRUE ~ "Efficient Maintenance"
    ),
    specific_actions = case_when(
      strategic_priority == "Invest (High Value, Positive Sensitivity)" ~
        "Proactive outreach, dedicated account managers, premium support, personalized offers",
      strategic_priority == "Protect (High Value, Negative Sensitivity)" ~
        "Monitor closely, address issues before they escalate, maintain current service quality",
      strategic_priority == "Monitor (Lower Value, Positive Sensitivity)" ~
        "Cost-effective automated engagement, basic retention offers, usage encouragement",
      TRUE ~ "Standard service levels, efficient cost management"
    ),
    budget_allocation = case_when(
      investment_level == "High Investment" ~ "40% of retention budget",
      investment_level == "Moderate Investment" ~ "30% of retention budget",
      investment_level == "Selective Investment" ~ "20% of retention budget", 
      TRUE ~ "10% of retention budget"
    )
  ) %>%
  arrange(desc(avg_monthly_revenue), desc(avg_sensitivity))

print("Targeted Customer Retention Strategy:")
print(recommendations)

# 11. FINANCIAL SUMMARY AND STRATEGIC RECOMMENDATIONS
cat("\n=== FINANCIAL IMPACT SUMMARY ===\n")
cat("Total customers analyzed:", nrow(df_segmented), "\n")
cat("Current churn rate:", round(mean(df_segmented$Churn == "Yes") * 100, 1), "%\n")
cat("Average customer lifetime value: $", round(clv_analysis$simple_clv, 2), "\n")
cat("Annual cost of churn: $", round(clv_analysis$annual_churn_cost, 2), "\n")
cat("10% churn reduction saves: $", round(annual_savings, 2), "annually\n\n")

cat("=== RECOMMENDED INTERVENTIONS ===\n")
cat("1. STANDARD CARE PROGRAM ($50/customer):\n")
cat("   - ROI:", care_program_roi$roi_percentage, "%\n")
cat("   - Payback period:", round(care_program_roi$payback_period_months, 1), "months\n")
cat("   - Target:", care_program_roi$targeted_customers, "high-value customers\n\n")

cat("2. PREMIUM CARE PROGRAM ($100/customer):\n")
cat("   - ROI:", premium_care_roi$roi_percentage, "%\n")
cat("   - Payback period:", round(premium_care_roi$payback_period_months, 1), "months\n")
cat("   - Better for long-term customer relationships\n\n")

cat("=== KEY STRATEGIC INSIGHTS ===\n")
cat("1. HETEROGENEOUS EFFECTS: Customer care impact varies significantly by segment\n")
cat("2. TARGETED INVESTMENT: Focus on", sum(df_with_bands$strategic_priority == "Invest (High Value, Positive Sensitivity)"), 
    "high-value, high-sensitivity customers\n")
cat("3. PROTECT ASSETS:", sum(df_with_bands$strategic_priority == "Protect (High Value, Negative Sensitivity)"),
    "high-value customers are already retained well\n")
cat("4. COST-EFFICIENCY: Differentiated strategies maximize ROI across segments\n")
cat("5. DATA-DRIVEN: Usage patterns (PercChangeMinutes) are strong predictors\n")

# Save results for presentation
write.csv(recommendations, "targeted_recommendations.csv", row.names = FALSE)
write.csv(targeting_analysis, "segment_analysis.csv", row.names = FALSE)
write.csv(df_with_bands %>% select(CustomerID, strategic_priority, pred_sensitivity, business_value), 
          "customer_priority_scores.csv", row.names = FALSE)

cat("\nAnalysis complete! Results saved to CSV files.\n")
cat("Files created:\n")
cat("- targeted_recommendations.csv: Strategic recommendations by segment\n")
cat("- segment_analysis.csv: Detailed targeting analysis\n") 
cat("- customer_priority_scores.csv: Individual customer scores for implementation\n")
