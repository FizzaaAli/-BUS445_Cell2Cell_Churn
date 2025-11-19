###############################################################
# Cell2Cell Churn Challenge — Segmentation & Executive Plots
# Generates all plots for the presentation
###############################################################

# --- 0. SETUP ------------------------------------------------
library(tidyverse)
library(ggthemes)
library(scales)
library(broom)

# --- 1. LOAD CLEAN DATA -------------------------------------
df <- read.csv("data/clean_service_vars_new.csv")

# Ensure Churn_flag is factor
df$Churn_flag <- factor(df$Churn_flag, labels = c("Stayed", "Churned"))

# --- 2. FEATURE ENGINEERING (same as model script) ----------
df <- df %>%
  mutate(
    OverageRatio = OverageMinutes / (MonthlyMinutes + 1),
    sqrt_Tenure = sqrt(MonthsInService),
    UsageDeclineFlag = ifelse(PercChangeMinutes < -20, 1, 0)
  )

# --- 3. RISK SEGMENTATION -----------------------------------
# Define risk groups based on churn probability threshold (0.3 cutoff)
# You must load the model_interact from LogisticRegression.R
model_interact <- readRDS("data/logistic_model_final.rds")

df$pred_prob <- predict(model_interact, newdata = df, type = "response")

df <- df %>%
  mutate(
    RiskSegment = case_when(
      pred_prob > 0.40 ~ "High",
      pred_prob > 0.25 ~ "Medium",
      TRUE ~ "Low"
    )
  )

df$RiskSegment <- factor(df$RiskSegment, levels = c("Low", "Medium", "High"))

# --- 4. PLOTS ------------------------------------------------
# All saved under /plots directory
dir.create("plots", showWarnings = FALSE)

## 4.1 Odds Ratio Plot ---------------------------------------
coef_df <- tidy(model_interact, conf.int = TRUE, exponentiate = TRUE)

p1 <- coef_df %>%
  filter(term != "(Intercept)") %>%
  ggplot(aes(x = reorder(term, estimate), y = estimate)) +
  geom_point(color = "steelblue", size = 3) +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.2) +
  coord_flip() +
  theme_minimal() +
  labs(title = "Odds Ratios of Key Predictors",
       x = "Predictor",
       y = "Odds Ratio")

ggsave("plots/odds_ratios.png", p1, width = 8, height = 6)

## 4.2 Churn Rate by Segment ---------------------------------
p2 <- df %>%
  group_by(RiskSegment) %>%
  summarise(ChurnRate = mean(Churn_flag == "Churned")) %>%
  ggplot(aes(x = RiskSegment, y = ChurnRate, fill = RiskSegment)) +
  geom_col() +
  scale_y_continuous(labels = percent) +
  theme_minimal() +
  labs(title = "Churn Rate by Risk Segment",
       y = "Churn Rate", x = "")

ggsave("plots/churn_rate_by_segment.png", p2, width = 7, height = 5)

## 4.3 Revenue by Segment -------------------------------------
p3 <- df %>%
  group_by(RiskSegment) %>%
  summarise(AverageRevenue = mean(MonthlyRevenue)) %>%
  ggplot(aes(x = RiskSegment, y = AverageRevenue, fill = RiskSegment)) +
  geom_col() +
  theme_minimal() +
  labs(title = "Average Monthly Revenue by Segment",
       y = "Revenue ($)", x = "")

ggsave("plots/revenue_by_segment.png", p3, width = 7, height = 5)

## 4.4 Dual Axis: Revenue + Churn -----------------------------
segment_summary <- df %>%
  group_by(RiskSegment) %>%
  summarise(
    ChurnRate = mean(Churn_flag == "Churned"),
    AvgRevenue = mean(MonthlyRevenue)
  )

p4 <- ggplot(segment_summary, aes(x = RiskSegment)) +
  geom_col(aes(y = AvgRevenue), fill = "steelblue", alpha = 0.7) +
  geom_line(aes(y = ChurnRate * 200), group = 1, color = "tomato", size = 1.5) +
  geom_point(aes(y = ChurnRate * 200), color = "tomato", size = 3) +
  scale_y_continuous(
    name = "Average Revenue ($)",
    sec.axis = sec_axis(~./200, name = "Churn Rate")
  ) +
  theme_minimal() +
  labs(title = "Revenue & Churn Risk by Segment")

ggsave("plots/dualaxis_revenue_churn.png", p4, width = 8, height = 5)

## 4.5 Tenure vs Care Calls -----------------------------------
p5 <- df %>%
  ggplot(aes(x = MonthsInService, y = CustomerCareCalls, color = RiskSegment)) +
  geom_point(alpha = 0.3) +
  theme_minimal() +
  labs(title = "Tenure vs Customer Care Calls by Risk Segment",
       x = "Tenure (Months)",
       y = "Customer Care Calls")

ggsave("plots/tenure_vs_carecalls.png", p5, width = 8, height = 5)


## 4.5 Tenure vs Customer Care Calls — BAR VERSION --------------
# Bucket tenure into meaningful groups
df <- df %>%
  mutate(TenureGroup = case_when(
    MonthsInService <= 12 ~ "0–12",
    MonthsInService <= 24 ~ "13–24",
    MonthsInService <= 36 ~ "25–36",
    MonthsInService <= 48 ~ "37–48",
    MonthsInService <= 60 ~ "49–60",
    TRUE ~ "61+"
  ))

df$TenureGroup <- factor(df$TenureGroup,
                         levels = c("0–12","13–24","25–36","37–48","49–60","61+"))

# Create summary table
tenure_summary <- df %>%
  group_by(TenureGroup, RiskSegment) %>%
  summarise(AvgCareCalls = mean(CustomerCareCalls, na.rm = TRUE)) %>%
  ungroup()

# Bar chart
p5 <- ggplot(tenure_summary,
             aes(x = TenureGroup,
                 y = AvgCareCalls,
                 fill = RiskSegment)) +
  geom_col(position = "dodge", width = 0.75) +
  scale_fill_manual(values = c("Low" = "#88CCEE",
                               "Medium" = "#44AA99",
                               "High" = "#CC6677")) +
  theme_minimal(base_size = 14) +
  labs(title = "Avg Customer Care Calls by Tenure Group and Risk Segment",
       x = "Tenure Group (Months)",
       y = "Average Customer Care Calls",
       fill = "Risk Segment") +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    axis.text.x = element_text(angle = 20, hjust = 1)
  )

ggsave("plots/tenure_vs_carecalls_bar.png", p5, width = 9, height = 6)


## 4.6 Usage Decline Flag vs Churn -----------------------------
p6 <- df %>%
  group_by(UsageDeclineFlag) %>%
  summarise(ChurnRate = mean(Churn_flag == "Churned")) %>%
  mutate(UsageDeclineFlag = factor(UsageDeclineFlag,
                                   labels = c("No Decline", "Usage Dropped >20%"))) %>%
  ggplot(aes(x = UsageDeclineFlag, y = ChurnRate, fill = UsageDeclineFlag)) +
  geom_col() +
  scale_y_continuous(labels = percent) +
  theme_minimal() +
  labs(title = "Churn Rate by Usage Decline",
       x = "", y = "Churn Rate")

ggsave("plots/usage_decline_vs_churn.png", p6, width = 7, height = 5)

## 4.7 Churn Probability Distribution --------------------------
p7 <- ggplot(df, aes(pred_prob)) +
  geom_histogram(bins = 40, fill = "steelblue", alpha = 0.7) +
  geom_density(color = "tomato", size = 1) +
  theme_minimal() +
  labs(title = "Distribution of Predicted Churn Probabilities",
       x = "Predicted Churn Probability",
       y = "Density")

ggsave("plots/churn_probability_distribution.png", p7, width = 8, height = 5)

###############################################################
cat("All plots created and saved in /plots folder.\n")
