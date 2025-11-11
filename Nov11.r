# Cell2Cell Churn Analysis - Corrected Approach
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
    )
  )

# Remove rows with missing CreditRating
df_segmented <- df_segmented %>% filter(!is.na(CreditRating_num))

cat("Data after cleaning:", nrow(df_segmented), "rows\n")

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

# 5. HETEROGENEOUS TREATMENT EFFECTS MODELING - CORRECTED APPROACH
cat("\n5. Building heterogeneous effects models...\n")

# Convert segments to factors for modeling
df_segmented <- df_segmented %>%
  mutate(
    revenue_segment = factor(revenue_segment),
    usage_segment = factor(usage_segment),
    tenure_segment = factor(tenure_segment)
  )

# Split training data for validation
train_indices <- createDataPartition(df_segmented$Churn_binary, p = 0.7, list = FALSE)
train_data <- df_segmented[train_indices, ]
test_data <- df_segmented[-train_indices, ]

# Remove any remaining rows with NA in model variables
train_data_clean <- train_data %>%
  filter(!is.na(CreditRating_num) & !is.na(CustomerCareCalls) & 
         !is.na(OverageMinutes) & !is.na(PercChangeMinutes) & 
         !is.na(MonthsInService))

cat("Training data rows after cleaning:", nrow(train_data_clean), "\n")

# FOCUS ON DIFFERENT TREATMENTS - Let's model sensitivity to different factors
# Instead of just customer care calls, let's model sensitivity to usage changes and overage

# Model 1: Sensitivity to usage changes
model_usage <- glm(Churn_binary ~ 
                   PercChangeMinutes * revenue_segment +
                   PercChangeMinutes * usage_segment +
                   CustomerCareCalls + OverageMinutes + 
                   CreditRating_num + MonthsInService,
                 data = train_data_clean, family = "binomial")

# Model 2: Sensitivity to overage charges
model_overage <- glm(Churn_binary ~ 
                     OverageMinutes * revenue_segment +
                     OverageMinutes * tenure_segment +
                     CustomerCareCalls + PercChangeMinutes + 
                     CreditRating_num + MonthsInService,
                   data = train_data_clean, family = "binomial")

# 6. PREDICT SENSITIVITY TO USAGE CHANGES (More meaningful treatment)
cat("\n6. Predicting customer sensitivity to usage changes...\n")

predict_usage_sensitivity <- function(model, data, treatment_var = "PercChangeMinutes") {
  # Clean data for prediction
  data_clean <- data %>%
    filter(!is.na(CreditRating_num) & !is.na(CustomerCareCalls) & 
           !is.na(OverageMinutes) & !is.na(PercChangeMinutes) & 
           !is.na(MonthsInService))
  
  # Predict with original data
  pred_original <- predict(model, newdata = data_clean, type = "response")
  
  # Create data with treatment increased by 10 units (10% increase in minutes)
  data_modified <- data_clean
  data_modified[[treatment_var]] <- data_modified[[treatment_var]] + 10
  
  # Predict with modified data
  pred_modified <- predict(model, newdata = data_modified, type = "response")
  
  # Sensitivity = change in churn probability for 10% usage increase
  sensitivity <- pred_modified - pred_original
  
  return(data_clean %>% mutate(pred_sensitivity = sensitivity))
}

# Apply sensitivity prediction for usage changes
df_with_sensitivity <- predict_usage_sensitivity(model_usage, df_segmented)

cat("Usage sensitivity prediction completed on", nrow(df_with_sensitivity), "rows\n")
cat("Sensitivity distribution:\n")
print(summary(df_with_sensitivity$pred_sensitivity))

# Create strategic segments based on usage sensitivity
df_with_bands <- df_with_sensitivity %>%
  mutate(
    # Use absolute value for sensitivity bands since we care about magnitude of response
    sensitivity_magnitude = abs(pred_sensitivity),
    sensitivity_band = cut(sensitivity_magnitude, 
                          breaks = quantile(sensitivity_magnitude, probs = seq(0, 1, 0.2), na.rm = TRUE),
                          labels = c("Very Low", "Low", "Medium", "High", "Very High")),
    
    # Business value score - combination of sensitivity magnitude and revenue
    business_value = sensitivity_magnitude * MonthlyRevenue,
    
    # Strategic priority based on EDA findings - CORRECTED LOGIC
    strategic_priority = case_when(
      # High value customers who are sensitive to interventions
      MonthlyRevenue > 60 & sensitivity_magnitude > quantile(sensitivity_magnitude, 0.6, na.rm = TRUE) ~ "High Priority - Invest",
      # High churn risk customers based on EDA
      Churn == "Yes" & revenue_segment == "High Value" ~ "High Priority - Protect", 
      Churn == "Yes" & revenue_segment == "Medium Value" ~ "Medium Priority - Retain",
      PercChangeMinutes < -20 & revenue_segment == "High Value" ~ "Medium Priority - Re-engage",
      TRUE ~ "Standard - Maintain"
    )
  )

# 7. TARGETING ANALYSIS
cat("\n7. Identifying optimal target segments...\n")

targeting_analysis <- df_with_bands %>%
  group_by(strategic_priority, revenue_segment) %>%
  summarise(
    avg_sensitivity_magnitude = mean(sensitivity_magnitude, na.rm = TRUE),
    avg_monthly_revenue = mean(MonthlyRevenue, na.rm = TRUE),
    business_value_score = mean(business_value, na.rm = TRUE),
    current_churn_rate = mean(Churn == "Yes", na.rm = TRUE),
    avg_usage_change = mean(PercChangeMinutes, na.rm = TRUE),
    n_customers = n(),
    .groups = 'drop'
  ) %>%
  arrange(desc(business_value_score))

print("Optimal Targeting Strategy:")
print(targeting_analysis)

# 8. COST-BENEFIT ANALYSIS - BASED ON EDA FINDINGS
cat("\n8. Calculating ROI for interventions based on EDA insights...\n")

calculate_intervention_roi <- function(segment_data, priority_level, cost_per_customer, effectiveness) {
  
  # Target customers based on strategic priority
  targets <- segment_data %>%
    filter(strategic_priority == priority_level)
  
  if(nrow(targets) == 0) {
    return(list(
      targeted_customers = 0,
      intervention_cost = 0,
      expected_annual_savings = 0,
      roi_percentage = 0,
      net_benefit = 0
    ))
  }
  
  n_targeted <- nrow(targets)
  total_cost <- n_targeted * cost_per_customer
  
  # Calculate expected impact based on current churn rate
  current_churn <- mean(targets$Churn == "Yes", na.rm = TRUE)
  churn_reduction <- current_churn * effectiveness
  customers_saved <- n_targeted * churn_reduction
  
  # Calculate savings using CLV
  avg_clv <- mean(targets$MonthlyRevenue, na.rm = TRUE) * 
             mean(targets$MonthsInService, na.rm = TRUE)
  
  annual_savings <- customers_saved * avg_clv
  roi <- ifelse(total_cost > 0, (annual_savings - total_cost) / total_cost, 0)
  
  return(list(
    targeted_customers = n_targeted,
    avg_customer_value = round(avg_clv, 2),
    intervention_cost = total_cost,
    expected_annual_savings = annual_savings,
    roi_percentage = round(roi * 100, 1),
    net_benefit = annual_savings - total_cost
  ))
}

# Calculate ROI for different intervention strategies based on EDA
high_priority_roi <- calculate_intervention_roi(df_with_bands, "High Priority - Invest", 100, 0.20)
protect_roi <- calculate_intervention_roi(df_with_bands, "High Priority - Protect", 50, 0.15)
reengage_roi <- calculate_intervention_roi(df_with_bands, "Medium Priority - Re-engage", 30, 0.25)

cat("\nHigh Priority Investment ROI ($100/customer):\n")
print(high_priority_roi)

cat("\nProtect High Value Customers ROI ($50/customer):\n")
print(protect_roi)

cat("\nRe-engagement Program ROI ($30/customer):\n")
print(reengage_roi)

# 9. VISUALIZATIONS
cat("\n9. Creating visualizations...\n")

# Plot 1: Churn rate by usage segment and revenue
p1 <- df_segmented %>%
  group_by(usage_segment, revenue_segment) %>%
  summarise(churn_rate = mean(Churn == "Yes", na.rm = TRUE), .groups = 'drop') %>%
  ggplot(aes(x = usage_segment, y = churn_rate, fill = revenue_segment)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Churn Rates by Usage Pattern and Revenue Segment",
       subtitle = "Based on EDA findings - declining usage correlates with higher churn",
       y = "Churn Rate", x = "Usage Pattern") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Plot 2: Strategic priority distribution
p2 <- df_with_bands %>%
  group_by(strategic_priority) %>%
  summarise(n_customers = n(), .groups = 'drop') %>%
  ggplot(aes(x = strategic_priority, y = n_customers, fill = strategic_priority)) +
  geom_bar(stat = "identity") +
  labs(title = "Customer Distribution by Strategic Priority",
       y = "Number of Customers", x = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")

# Plot 3: Business impact by segment
p3 <- targeting_analysis %>%
  ggplot(aes(x = revenue_segment, y = strategic_priority, size = business_value_score, color = current_churn_rate)) +
  geom_point(alpha = 0.7) +
  scale_color_gradient2(low = "green", high = "red", mid = "yellow", midpoint = 0.3) +
  labs(title = "Business Value and Churn Risk by Segment",
       subtitle = "Size = Business Value, Color = Churn Rate",
       x = "Revenue Segment", y = "Strategic Priority") +
  theme_minimal()

# Display plots
print(p1)
print(p2)
print(p3)

# 10. ACTIONABLE RECOMMENDATIONS BASED ON EDA
cat("\n10. Generating actionable recommendations based on EDA insights...\n")

recommendations <- df_with_bands %>%
  group_by(strategic_priority, revenue_segment) %>%
  summarise(
    segment_size = n(),
    current_churn_rate = round(mean(Churn == "Yes", na.rm = TRUE), 3),
    avg_monthly_revenue = round(mean(MonthlyRevenue, na.rm = TRUE), 2),
    avg_usage_change = round(mean(PercChangeMinutes, na.rm = TRUE), 1),
    avg_tenure = round(mean(MonthsInService, na.rm = TRUE), 1),
    .groups = 'drop'
  ) %>%
  mutate(
    # Recommendations based on EDA patterns
    recommendation = case_when(
      strategic_priority == "High Priority - Invest" ~ "Proactive premium care: Address before issues escalate",
      strategic_priority == "High Priority - Protect" ~ "Retention offers: High-value customers at risk",
      strategic_priority == "Medium Priority - Re-engage" ~ "Usage encouragement: Combat declining usage patterns",
      strategic_priority == "Medium Priority - Retain" ~ "Targeted retention: Medium-value churn risks",
      TRUE ~ "Efficient maintenance: Standard service levels"
    ),
    key_metrics = case_when(
      strategic_priority == "High Priority - Invest" ~ paste("High sensitivity, Revenue: $", avg_monthly_revenue),
      strategic_priority == "High Priority - Protect" ~ paste("Current churn: ", current_churn_rate*100, "%, Revenue: $", avg_monthly_revenue),
      strategic_priority == "Medium Priority - Re-engage" ~ paste("Usage decline: ", avg_usage_change, "%, Revenue: $", avg_monthly_revenue),
      TRUE ~ paste("Revenue: $", avg_monthly_revenue, ", Churn: ", current_churn_rate*100, "%")
    ),
    budget_priority = case_when(
      strategic_priority == "High Priority - Invest" ~ "Highest",
      strategic_priority == "High Priority - Protect" ~ "High", 
      strategic_priority %in% c("Medium Priority - Re-engage", "Medium Priority - Retain") ~ "Medium",
      TRUE ~ "Standard"
    )
  ) %>%
  arrange(desc(budget_priority), desc(avg_monthly_revenue))

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
cat("1. HIGH PRIORITY INVESTMENT:\n")
cat("   - Target:", high_priority_roi$targeted_customers, "high-sensitivity customers\n")
cat("   - ROI:", high_priority_roi$roi_percentage, "%\n")
cat("   - Focus: Proactive premium care\n\n")

cat("2. PROTECT HIGH-VALUE CUSTOMERS:\n")
cat("   - Target:", protect_roi$targeted_customers, "high-value at-risk customers\n")
cat("   - ROI:", protect_roi$roi_percentage, "%\n")
cat("   - Focus: Retention offers and dedicated support\n\n")

cat("3. RE-ENGAGEMENT PROGRAM:\n")
cat("   - Target:", reengage_roi$targeted_customers, "customers with declining usage\n")
cat("   - ROI:", reengage_roi$roi_percentage, "%\n")
cat("   - Focus: Usage encouragement and win-back campaigns\n\n")

cat("=== KEY STRATEGIC INSIGHTS FROM EDA ===\n")
cat("1. USAGE DECLINE PREDICTS CHURN: Customers with <-20% usage change have 30%+ churn rates\n")
cat("2. HIGH VALUE VULNERABILITY: High-value customers still churn at 20-30% rates\n")
cat("3. SERVICE ENGAGEMENT: Retained customers make slightly more care calls\n")
cat("4. OVERAGE FRUSTRATION: High overage minutes correlate with churn risk\n")
cat("5. NETWORK QUALITY: Dropped calls show weak relationship with churn\n")

# Save results for presentation
write.csv(recommendations, "targeted_recommendations_corrected.csv", row.names = FALSE)
write.csv(targeting_analysis, "segment_analysis_corrected.csv", row.names = FALSE)

cat("\nAnalysis complete! Results saved to CSV files.\n")
cat("Key insight: Customer care calls show complex relationships - focus on usage patterns and value segments instead.\n")
