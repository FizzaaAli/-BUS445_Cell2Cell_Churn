###############################################################
# BUS 445 - Storytelling with Data
# Case: Cell2Cell Churn Challenge
# Pathway 2: "The What" – Diagnosing Service Factors & Heterogeneity
# Author: Fizza ALi
###############################################################



# --- 0. SETUP ------------------------------------------------
# Install packages if not already installed
pkgs <- c("tidyverse", "car", "pROC", "randomForest", "FactoMineR", "factoextra")
install.packages(setdiff(pkgs, rownames(installed.packages())), dependencies=TRUE)

# Load libraries
library(tidyverse)
library(car)
library(pROC)
library(randomForest)
library(FactoMineR)
library(factoextra)
library(caret)



# --- 1. LOAD & CLEAN DATA ------------------------------------
train <- read.csv("data/cell2celltrain.csv", stringsAsFactors = FALSE)

# Preview
glimpse(train)

# Check missing values by column
colSums(is.na(train))

# No variable exceeds 5% missing data, which means we don’t need to remove or 
# impute aggressively — we can either drop those few NAs during modeling or replace 
# them with median/mean if needed later.

# Convert Churn to binary flag where 1 = Yes and 0 = No
train <- train %>%
  mutate(Churn_flag = ifelse(tolower(trimws(Churn)) %in% c("yes","y","1","true"), 1, 0))

# Coerce CreditRating (1-Highest, 2-High, 3-Good, 4-Medium, 5-Low, 6-VeryLow, 7-Lowest) into numeric rank
train <- train %>%
  mutate(CreditRating_num = as.numeric(str_extract(CreditRating, "\\d+")))


# Select relevant variables for "The What" pathway
service_vars <- train %>%
  select(
    Churn_flag,                    # target
    DroppedBlockedCalls,           # overall network reliability (sum of dropped + blocked)
    CustomerCareCalls,             # customer service experience
    OverageMinutes,                # pricing fairness / overage behavior
    MonthlyRevenue,                # value perception
    MonthlyMinutes,                # usage intensity
    PercChangeMinutes,             # behavioral change (usage fluctuation)
    PercChangeRevenues,            # revenue fluctuation
    MonthsInService,               # tenure / loyalty measure
    CreditRating_num               # customer quality / financial health proxy
  )


# After selecting variables, ensure numeric columns are truly numeric
service_vars <- service_vars %>%
  mutate(across(-Churn_flag, as.numeric))

# Put Churn_flag at the end (optional aesthetic choice)
service_vars <- service_vars %>%
  relocate(Churn_flag, .after = last_col())


# Check structure and summary of these variables
glimpse(service_vars)
summary(service_vars)

# Handle missing values:
# Since your earlier check showed no variable exceeds 5% NA, we can drop rows with any NA for simplicity
service_vars <- na.omit(service_vars)

# Confirm no missing values remain
colSums(is.na(service_vars))

# Optional: check for extreme outliers that might distort analysis
service_vars %>%
  gather(Variable, Value, -Churn_flag) %>%
  ggplot(aes(x = Variable, y = Value)) +
  geom_boxplot(outlier.colour = "red") +
  theme_minimal() +
  labs(title = "Boxplots of Service-Related Variables", x = "", y = "Value")

# Save a clean version
write.csv(service_vars, "data/clean_service_vars.csv", row.names = FALSE)




# --- 2. EXPLORATORY DATA ANALYSIS -----------------------------

# Churn Rate Overview
table(service_vars$Churn_flag)
prop.table(table(service_vars$Churn_flag))

# Numeric summary by churn group
summ <- service_vars %>%
  group_by(Churn_flag) %>%
  summarise(across(where(is.numeric), 
                   list(mean = mean, median = median, sd = sd), 
                   .names = "{.col}_{.fn}"))


# Histograms of key service variables
service_vars %>%
  gather(Variable, Value, -Churn_flag) %>%
  ggplot(aes(x = Value, fill = factor(Churn_flag))) +
  geom_histogram(alpha = 0.6, bins = 30, position = "identity") +
  facet_wrap(~Variable, scales = "free") +
  scale_fill_manual(values = c("steelblue", "tomato"),
                    labels = c("Stayed", "Churned"),
                    name = "Customer Status") +
  theme_minimal() +
  labs(title = "Distribution of Service Variables by Churn Status",
       x = "Value", y = "Count")

# Boxplots comparing churn vs non-churn
service_vars %>%
  gather(Variable, Value, -Churn_flag) %>%
  ggplot(aes(x = factor(Churn_flag), y = Value, fill = factor(Churn_flag))) +
  geom_boxplot(alpha = 0.7, outlier.colour = "red") +
  facet_wrap(~Variable, scales = "free") +
  scale_fill_manual(values = c("steelblue", "tomato"),
                    labels = c("Stayed", "Churned"),
                    name = "Customer Status") +
  theme_minimal() +
  labs(title = "Service Variable Differences by Churn", x = "", y = "Value")



# Numeric correlation matrix (excluding churn flag)
corr_matrix <- cor(service_vars %>% select(-Churn_flag), use = "complete.obs")

# Visualize correlations
corrplot::corrplot(corr_matrix, method = "color", tl.cex = 0.8,
                   title = "Correlation Matrix of Service Variables", mar = c(0,0,2,0))

# Identify high correlations (> 0.7)
high_corr <- findCorrelation(corr_matrix, cutoff = 0.7, names = TRUE)
high_corr



# Fit an initial logistic regression model with all predictors
logit_full <- glm(Churn_flag ~ ., data = service_vars, family = binomial)

# Compute Variance Inflation Factors (VIF)
vif_values <- car::vif(logit_full)
vif_values

# Flag variables with VIF > 5
vif_values[vif_values > 5]





