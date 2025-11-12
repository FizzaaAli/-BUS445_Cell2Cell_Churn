###############################################################
# --- 12. RISK SEGMENTATION ----------------------------------
# Goal: Translate churn probabilities into actionable customer risk groups
###############################################################

# Add risk segments based on predicted probabilities
model_data <- model_data %>%
  mutate(
    RiskSegment = case_when(
      pred_prob >= 0.3 ~ "High Risk",
      pred_prob >= 0.2 ~ "Medium Risk",
      TRUE ~ "Low Risk"
    )
  )

# View distribution of customers by risk group
table(model_data$RiskSegment)
prop.table(table(model_data$RiskSegment))


# Join back to main data for access to service variables
model_summary <- model_data %>%
  left_join(data, by = c("Churn_flag")) %>%
  group_by(RiskSegment) %>%
  summarise(
    Avg_Prob = mean(pred_prob),
    Avg_Revenue = mean(MonthlyRevenue, na.rm = TRUE),
    Avg_Tenure = mean(MonthsInService, na.rm = TRUE),
    Avg_CareCalls = mean(CustomerCareCalls, na.rm = TRUE),
    Churn_Rate = mean(as.numeric(Churn_flag == "Churned"))
  ) %>%
  arrange(desc(Avg_Prob))

print(model_summary)


# Plot churn probability distribution by risk segment
ggplot(model_data, aes(x = pred_prob, fill = RiskSegment)) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
  scale_fill_manual(values = c("Low Risk" = "steelblue",
                               "Medium Risk" = "goldenrod",
                               "High Risk" = "tomato")) +
  theme_minimal() +
  labs(
    title = "Distribution of Predicted Churn Probability by Risk Segment",
    x = "Predicted Probability of Churn",
    y = "Number of Customers",
    fill = "Risk Segment"
  )
