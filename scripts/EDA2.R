###############################################################
# Cell2Cell Churn Challenge — Business-Focused EDA
# Objective: Identify patterns and potential churn drivers
# Author: [Your Name]
# Date: [Today’s Date]
###############################################################

# --- 0. SETUP ------------------------------------------------
# Install and load required packages
pkgs <- c("tidyverse", "corrplot", "skimr", "scales", "ggthemes", "gridExtra")
install.packages(setdiff(pkgs, rownames(installed.packages())), dependencies = TRUE)
lapply(pkgs, library, character.only = TRUE)


# --- 1. LOAD AND CLEAN DATA ----------------------------------
train <- read.csv("data/cell2celltrain.csv", stringsAsFactors = FALSE)

# Preview
glimpse(train)

# Clean target variable
train <- train %>%
  mutate(Churn_flag = ifelse(tolower(trimws(Churn)) %in% c("yes", "y", "1", "true"), 1, 0))

# Combine dropped + blocked calls (service quality)
train <- train %>%
  mutate(DroppedBlockedCalls = DroppedCalls + BlockedCalls)

# Extract numeric part from credit rating (1 = best)
train <- train %>%
  mutate(CreditRating_num = as.numeric(str_extract(CreditRating, "\\d+")))

# Handle missing values (drop rows with >2 missing predictors)
train <- train[rowSums(is.na(train)) <= 2, ]

# --- 2. BASIC OVERVIEW ---------------------------------------
cat("Dataset size:", nrow(train), "rows\n")
cat("Churn rate:", round(mean(train$Churn_flag) * 100, 1), "%\n")

# Quick overview of numeric distributions
skimr::skim(train %>% select_if(is.numeric))

# Churn balance
train %>%
  count(Churn_flag) %>%
  mutate(Percent = n / sum(n) * 100)

# --- 3. CORE SERVICE VARIABLES -------------------------------
service_vars <- train %>%
  select(Churn_flag,
         CustomerCareCalls,
         DroppedBlockedCalls,
         MonthlyRevenue,
         MonthlyMinutes,
         OverageMinutes,
         PercChangeMinutes,
         PercChangeRevenues,
         MonthsInService,
         CreditRating_num)

summary(service_vars)

# --- 4. CHURN VS NON-CHURN COMPARISONS ------------------------

# Function for summary by churn status
summ_by_churn <- service_vars %>%
  group_by(Churn_flag) %>%
  summarise(across(where(is.numeric),
                   list(mean = mean, median = median, sd = sd),
                   .names = "{.col}_{.fn}"))

print(summ_by_churn)

# --- 5. VISUAL EXPLORATION -----------------------------------

# Churn distribution
ggplot(service_vars, aes(factor(Churn_flag), fill = factor(Churn_flag))) +
  geom_bar() +
  scale_fill_manual(values = c("steelblue", "tomato"), labels = c("Stayed", "Churned")) +
  theme_minimal() +
  labs(title = "Customer Churn Distribution", x = "Churn Status", y = "Count")

# Boxplots for churn vs non-churn
service_vars %>%
  pivot_longer(-Churn_flag, names_to = "Variable", values_to = "Value") %>%
  ggplot(aes(x = factor(Churn_flag), y = Value, fill = factor(Churn_flag))) +
  geom_boxplot(alpha = 0.7, outlier.color = "red") +
  facet_wrap(~Variable, scales = "free", ncol = 3) +
  scale_fill_manual(values = c("steelblue", "tomato"), labels = c("Stayed", "Churned")) +
  labs(title = "Distribution of Key Service Variables by Churn Status",
       x = "Customer Status", y = "Value") +
  theme_minimal()

# --- 6. CORRELATION AND MULTICOLLINEARITY --------------------

corr_data <- service_vars %>% select(-Churn_flag)
corr_matrix <- cor(corr_data, use = "complete.obs")
corrplot(corr_matrix, method = "color", type = "lower", tl.cex = 0.7,
         title = "Correlation Matrix of Service Predictors", mar = c(0,0,2,0))

# Identify highly correlated pairs (|r| > 0.7)
high_corr <- which(abs(corr_matrix) > 0.7 & abs(corr_matrix) < 1, arr.ind = TRUE)
if (length(high_corr) > 0) {
  cat("High correlations detected:\n")
  print(rownames(corr_matrix)[unique(high_corr[,1])])
} else {
  cat("No extreme multicollinearity detected.\n")
}

# --- 7. KEY FINDINGS (Quantitative Summaries) ----------------

# Example: Mean difference in overage minutes between churners & stayers
agg_diff <- service_vars %>%
  group_by(Churn_flag) %>%
  summarise(across(c(OverageMinutes, MonthlyRevenue, PercChangeMinutes,
                     CustomerCareCalls, DroppedBlockedCalls),
                   mean, na.rm = TRUE)) %>%
  mutate(Group = ifelse(Churn_flag == 1, "Churned", "Stayed"))

print(agg_diff)

# Save clean service dataset
write.csv(service_vars, "data/clean_service_vars_new.csv", row.names = FALSE)
 

