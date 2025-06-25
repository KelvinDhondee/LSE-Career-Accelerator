## LSE Data Analytics Online Career Accelerator 
# DA301:  Advanced Analytics for Organisational Impact
# Kelvin Dhondee
################################################################################

# Assignment 5 scenario
## Turtle Games’s sales department has historically preferred to use R when performing 
## sales analyses due to existing workflow systems. As you’re able to perform data analysis 
## in R, you will perform exploratory data analysis and present your findings by utilising 
## basic statistics and plots. You'll explore and prepare the data set to analyse sales per 
## product. The sales department is hoping to use the findings of this exploratory analysis 
## to inform changes and improvements in the team. (Note that you will use basic summary 
## statistics in Module 5 and will continue to go into more detail with descriptive 
## statistics in Module 6.)

################################################################################

## Assignment 5 objective
## Load and wrangle the data. Use summary statistics and groupings if required to sense-check
## and gain insights into the data. Make sure to use different visualisations such as scatterplots, 
## histograms, and boxplots to learn more about the data set. Explore the data and comment on the 
## insights gained from your exploratory data analysis. For example, outliers, missing values, 
## and distribution of data. Also make sure to comment on initial patterns and distributions or 
## behaviour that may be of interest to the business.

################################################################################

# Module 5 assignment: Load, clean and wrangle data using R

## It is strongly advised that you use the cleaned version of the data set that you created and 
##  saved in the Python section of the course. Should you choose to redo the data cleaning in R, 
##  make sure to apply the same transformations as you will have to potentially compare the results.
##  (Note: Manual steps included dropping and renaming the columns as per the instructions in module 1.
##  Drop ‘language’ and ‘platform’ and rename ‘remuneration’ and ‘spending_score’) 

## 1. Open your RStudio and start setting up your R environment. 
## 2. Open a new R script and import the turtle_review.csv data file, which you can download from 
##      Assignment: Predicting future outcomes. (Note: You can use the clean version of the data 
##      you saved as csv in module 1, or, can manually drop and rename the columns as per the instructions 
##      in module 1. Drop ‘language’ and ‘platform’ and rename ‘remuneration’ and ‘spending_score’) 
## 3. Import all the required libraries for the analysis and view the data. 
## 4. Load and explore the data.
##    - View the head the data.
##    - Create a summary of the new data frame.
## 5. Perform exploratory data analysis by creating tables and visualisations to better understand 
##      groupings and different perspectives into customer behaviour and specifically how loyalty 
##      points are accumulated. Example questions could include:
##    - Can you comment on distributions, patterns or outliers based on the visual exploration of the data?
##    - Are there any insights based on the basic observations that may require further investigation?
##    - Are there any groupings that may be useful in gaining deeper insights into customer behaviour?
##    - Are there any specific patterns that you want to investigate
## 6. Create
##    - Create scatterplots, histograms, and boxplots to visually explore the loyalty_points data.
##    - Select appropriate visualisations to communicate relevant findings and insights to the business.
## 7. Note your observations and recommendations to the technical and business users.

################################################################################

# libraries
library(dplyr) # Data manipulation
library(ggplot2) # Data visualisation
library(moments) # Skewness and kurtosis
library(corrplot) # Correlation visualisation
library(car) # VIF calculation
library(lmtest) # Statistical tests
library(caret) # Classification and REgression Training
library(randomForest) # Random Forest modeling

################################################################################

# 5.1 Data Ingestion and Wrangling

# Set working directory.
setwd("/Users/kelvin/Desktop/LSE Data Analytics Career Accelerator - Course 3 files/LSE_DA301_assignment_files_new")

df <- read.csv("reviews.csv")

# View first few rows of df.
head(df)
dim(df)

# Check for missing values
missing_values <- colSums(is.na(df))
cat("Missing values by column:\n")
print(missing_values)

# Check for duplicates
duplicate_count <- sum(duplicated(df))
cat("Number of duplicate rows:", duplicate_count, "\n")

# Drop unnecessary columns.
df <- df %>%
  select(-review, -summary)

# View first ten entries and view variable types.
as_tibble(df)

# Alternative view.
glimpse(df)

# product is an identifier. Convert to character.
df$product <- as.character(df$product)

# gender - convert to factor.
df$gender <- as.factor(df$gender)

# education - convert to factor.
df$education <- as.factor(df$education)

# Verify data structure
str(df)

################################################################################
################################################################################

# 5.2 Exploratory data analysis

# View descriptive statistics.
summary(df)
DataExplorer::create_report(df)

#unique(df$gender)

gender_frequency <- table(df$gender)
print(gender_frequency)

education_frequency <- table(df$education)
print(education_frequency)

product_frequency <- table(df$product)
print(product_frequency)

################################################################################

# 5.2a Group statistics

group_by_gender <- df %>%
  group_by(gender) %>%
  summarize(
    mean_loyalty = mean(loyalty_points),
    median_loyalty = median(loyalty_points),
    count = n()
  )

group_by_education <- df %>%
  group_by(education) %>%
  summarize(
    mean_loyalty = mean(loyalty_points),
    median_loyalty = median(loyalty_points),
    count = n()
  )

cat("Loyalty points by gender:\n")
print(group_by_gender)
cat("\nLoyalty points by education:\n")
print(group_by_education)

################################################################################

# 5.2b Plot Generation

# Plot 1: Histogram of Loyalty Points
loyalty_points_plot <- ggplot(data = df, aes (x = loyalty_points)) +
  geom_histogram(bins = 20, fill = 'cornflowerblue', color = "black") +
  labs(title = "Distribution of Loyalty Points", x = "Loyalty Points", y = "Frequency") +
  theme_minimal()

loyalty_points_plot

# Turtle Games would not want right-skewed data.

# Plot 2: Scatter plot of Age v Loyalty Points.
age_loyalty_scatter <- ggplot(data = df, aes( x = age, y = loyalty_points)) +
  geom_point(color = "coral2", alpha = 0.5) +
  labs(title = "Age vs Loyalty Points", x = "Age", y = "Loyalty Points") +
  theme_minimal()

age_loyalty_scatter

# Plot 3: Scatter plot of Remuneration v Loyalty Points
remuneration_loyalty_scatter <- ggplot(data = df, aes( x = remuneration, y = loyalty_points)) +
  geom_point(color = "darkslategray", alpha = 0.5) +
  labs(title = "Remuneration vs Loyalty Points", x = "Remuneration £k", y = "Loyalty Points") +
  theme_minimal()

remuneration_loyalty_scatter

# Plot 4: Scatter Plot of Spending Score v Loyalty Points
spending_score_loyalty_scatter <- ggplot(data = df, aes( x = spending_score, y = loyalty_points)) +
  geom_point(color = "darksalmon", alpha = 0.5) +
  labs(title = "Spending Score vs Loyalty Points", x = "Spending Score", y = "Loyalty Points") +
  theme_minimal()

spending_score_loyalty_scatter

# Plot 5: Boxplot of Loyalty Points by Gender
gender_loyalty_points_boxplot <- ggplot(data = df, aes(x = gender, y = loyalty_points)) +
  geom_boxplot(fill = "cyan", color = "black") +
  labs(title = "Loyalty Points by Gender", y = "Loyalty Points") +
  theme_minimal()

gender_loyalty_points_boxplot

# Plot 6: Violin Plot Loyalty Points by Educational Level 
edu_loyalty_points_violin <- ggplot(df, aes(x = education, y = loyalty_points)) +
  geom_violin(fill = 'dodgerblue4') +  
  geom_boxplot(fill = 'orange', width = 0.25,
               outlier.color = 'orange', outlier.size = 1,
               outlier.shape = 'square') +
  labs(title = "Loyalty Points by Educational Level",
       x = "Educational Level",
       y = "Loyalty Points") + theme_minimal()

edu_loyalty_points_violin

# Plot 7: Correlation Matrix, Correlation Plot, Pair Plot
# Drop unnecessary columns for correlation analysis
numeric_df <- df %>%
  select_if(is.numeric)

head(numeric_df)

# Calculate the correlation matrix using only complete (non-missing) observations
correlation_matrix <- cor(numeric_df, use = "complete.obs") 

# Create the correlation plot with a title
corrplot(correlation_matrix, method = "circle")

# View the correlation matrix
correlation_matrix

# Plot 8: Pair plots to visualize relationships
pairs(df[, c('age', 'remuneration', 'spending_score', 'loyalty_points')])

################################################################################

# 5.2c Measures of Shape

# Shapiro-Wilk test for normality
shapiro.test(df$loyalty_points)

# Skewness and Kurtosis
skewness(df$loyalty_points)
kurtosis(df$loyalty_points)

# Calculate Range
range_loyalty_points <- range(df$loyalty_points)

# Calculate Difference between highest and lowest values
difference_high_low <- diff(range_loyalty_points)

# Calculate Interquartile Range (IQR)
iqr_loyalty_points <- IQR(df$loyalty_points)

# Calculate Variance
variance_loyalty_points <- var(df$loyalty_points)

# Calculate Standard Deviation
std_deviation_loyalty_points <- sd(df$loyalty_points)

# Display results
list(
  Range = range_loyalty_points,
  Difference = difference_high_low,
  IQR = iqr_loyalty_points,
  Variance = variance_loyalty_points,
  Standard_Deviation = std_deviation_loyalty_points
)

################################################################################

# 5.2d More Measures of Shape

scores <- df$loyalty_points

# Calculate mean, median, and mode
mean_score <- mean(scores)
median_score <- median(scores)
mode_score <- as.numeric(names(sort(table(scores), decreasing = TRUE)[1]))

# Print the results
cat("Mean:", mean_score, "\n")
cat("Median:", median_score, "\n")
cat("Mode:", mode_score, "\n")

################################################################################

# 5.2e Outlier detection
outliers_iqr <- function(x) {
  q1 <- quantile(x, 0.25)
  q3 <- quantile(x, 0.75)
  iqr <- q3 - q1
  lower_bound <- q1 - 1.5 * iqr
  upper_bound <- q3 + 1.5 * iqr
  return(sum(x < lower_bound | x > upper_bound))
}

outlier_counts <- sapply(df[sapply(df, is.numeric)], outliers_iqr)
cat("Outlier counts by variable (IQR method):\n")
print(outlier_counts)

# Calculate IQR for the loyalty_points column
Q1 <- quantile(df$loyalty_points, 0.25)
Q3 <- quantile(df$loyalty_points, 0.75)
IQR_value <- Q3 - Q1

# Determine lower and upper bounds
lower_bound <- Q1 - 1.5 * IQR_value
upper_bound <- Q3 + 1.5 * IQR_value

# Identify outliers (loyalty_points outside of the bounds)
outliers <- df$loyalty_points < lower_bound | df$loyalty_points > upper_bound

# Count the outliers
sum(outliers)

# Identify number of customers with loyalty_points above the upper bound
upper_outliers <- df$loyalty_points > upper_bound

# Count how many customers exceed the upper bound
num_upper_outliers <- sum(upper_outliers)
cat("Number of customers with loyalty points above the upper bound:", num_upper_outliers, "\n")

# Identify number of customers with loyalty_points below the lower bound
lower_outliers <- df$loyalty_points < lower_bound

# Count how many customers are below the lower bound
num_lower_outliers <- sum(lower_outliers)
cat("Number of customers with loyalty points below the lower bound:", num_lower_outliers, "\n")

# Remove rows with outliers in loyalty_points
df_no_outliers <- df[!outliers, ]

# View the cleaned data frame
str(df_no_outliers)

################################################################################
################################################################################
################################################################################
################################################################################

# Assignment 6 scenario

## In Module 5, you were requested to redo components of the analysis using Turtle Games’s preferred 
## language, R, in order to make it easier for them to implement your analysis internally. As a final 
## task the team asked you to perform a statistical analysis and create a multiple linear regression 
## model using R to predict loyalty points using the available features in a multiple linear model. 
## They did not prescribe which features to use and you can therefore use insights from previous modules 
## as well as your statistical analysis to make recommendations regarding suitability of this model type,
## the specifics of the model you created and alternative solutions. As a final task they also requested 
## your observations and recommendations regarding the current loyalty programme and how this could be 
## improved. 

################################################################################

## Assignment 6 objective
## You need to investigate customer behaviour and the effectiveness of the current loyalty program based 
## on the work completed in modules 1-5 as well as the statistical analysis and modelling efforts of module 6.
##  - Can we predict loyalty points given the existing features using a relatively simple MLR model?
##  - Do you have confidence in the model results (Goodness of fit evaluation)
##  - Where should the business focus their marketing efforts?
##  - How could the loyalty program be improved?
##  - How could the analysis be improved?

################################################################################

## Assignment 6 assignment: Making recommendations to the business.

## 1. Continue with your R script in RStudio from Assignment Activity 5: Cleaning, manipulating, and 
##     visualising the data.
## 2. Load and explore the data, and continue to use the data frame you prepared in Module 5.
## 3. Perform a statistical analysis and comment on the descriptive statistics in the context of the 
##     review of how customers accumulate loyalty points.
##  - Comment on distributions and patterns observed in the data.
##  - Determine and justify the features to be used in a multiple linear regression model and potential
##.    concerns and corrective actions.
## 4. Create a Multiple linear regression model using your selected (numeric) features.
##  - Evaluate the goodness of fit and interpret the model summary statistics.
##  - Create a visual demonstration of the model
##  - Comment on the usefulness of the model, potential improvements and alternate suggestions that could 
##     be considered.
##  - Demonstrate how the model could be used to predict given specific scenarios. (You can create your own 
##     scenarios).
## 5. Perform exploratory data analysis by using statistical analysis methods and comment on the descriptive 
##     statistics in the context of the review of how customers accumulate loyalty points.
## 6. Document your observations, interpretations, and suggestions based on each of the models created in 
##     your notebook. (This will serve as input to your summary and final submission at the end of the course.)

################################################################################

# 6.1 Multiple Linear Regression (model1)

# Create the multiple linear regression model
model1 <- lm(loyalty_points ~ remuneration + spending_score, data  = df)

# Summarize the model
summary(model1)

# Plot actual vs. predicted values (lmP smoothing method)
ggplot(df, aes(y = loyalty_points, x = predict(model1, df))) +
  geom_point(alpha = 0.5) +
  stat_smooth(method = "lm") +  
  labs(y = 'Actual Loyalty Points', x = 'Predicted Loyalty Points') +
  ggtitle('Predicted vs. Actual Loyalty Points model1') + theme_minimal()

# Check for multicollinearity
vif_values <- vif(model1)
cat("Variance Inflation Factors:\n")
print(vif_values)
# VIF between 1-5, therefore, low multicollinearity. 

# Test for heteroscedasticity
bp_test <- bptest(model1)
cat("Breusch-Pagan Test for Heteroscedasticity:\n")
print(bp_test)
# p < 0.05, therefore, evidence of heteroskedasticity in the regression model.

plot(fitted(model1), residuals(model1),
     main = "Fitted values vs Residuals - model1",
     xlab = "Fitted values",
     ylab = "Residuals")
abline(h = 0, col = "red")

# Test for autocorrelation of residuals
dw_test <- durbinWatsonTest(model1)
cat("Durbin-Watson Test for Autocorrelation:\n")
print(dw_test)

# Check for normality of residuals
# Q-Q plot
qqnorm(residuals(model1))
qqline(residuals(model1), col = "red")   
xlab("Theoretical Quantiles")                                
ylab("Sample Quantiles")                                    

# Shapiro-Wilk normality test
shapiro_test <- shapiro.test(residuals(model1))
cat("Shapiro-Wilk Test for Normality:\n")
print(shapiro_test)
# Although the model can still be valid with non-normal residuals, violations can affect the accuracy of p-values and confidence intervals.

# Create scenarios for prediction
scenarios <- data.frame(
  remuneration = c(13.94, 14.76, 14.76, 15.58),
  spending_score = c(76, 6, 94, 3)
)

# Make predictions
scenarios$predicted_loyalty <- predict(model1, newdata = scenarios)
print(scenarios)

################################################################################
################################################################################

# 6.2 Random forest (rf_model)

# Prepare data
set.seed(42)
train_index <- createDataPartition(df$loyalty_points, p = 0.8, list = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]

# Random Forest model
rf_model <- randomForest(loyalty_points ~ remuneration + spending_score,
                         data = train_data,
                         ntree = 500,
                         importance = TRUE)
print(rf_model)
importance(rf_model)

# Predictions on test data
rf_predictions <- predict(rf_model, test_data)
rf_rmse <- sqrt(mean((test_data$loyalty_points - rf_predictions)^2))
rf_mape <- mean(abs((test_data$loyalty_points - rf_predictions) / test_data$loyalty_points)) * 100

rf_rmse
rf_mape
# test metrics are relatively reasonable, so it may not be severe overfitting.

# Create scenarios for prediction
scenariosrf <- data.frame(
  remuneration = c(13.94, 14.76, 14.76, 15.58),
  spending_score = c(76, 6, 94, 3)
)

# Make predictions
scenariosrf$predicted_loyalty <- predict(rf_model, newdata = scenarios)
print(scenariosrf)

# Random Forest commentary:
# 500 trees used in the model, resulting in high R-squared value of 98.42%.
# Model explains about 98.42% of the variance in loyalty_points.
# %IncMSE measures how much the model accuracy decreases when a variable is excluded.
# %IncMSE suggests spending_score contributes more to model's predictive accuracy.
# IncNodePurity measures total decrease in node impurity from splitting on a variable.
# IncNodePurity confirms spending_score is more useful for creating homogeneous nodes in the trees
# rf_mse pf 147.4447 represents average squared difference between predicted and actual loyalty_points.
# rf_mape of 8.856101 - model's predictions deviate from the actual loyalty points by about 8.86%.
# rf_mape reasonably good. 

################################################################################
################################################################################

# 6.3 Multiple Linear Regression (model2)

# Create the multiple linear regression model(include age)
model2 <- lm(loyalty_points ~ age+ remuneration + spending_score, data  = df)

# Summarize the model
summary(model2)

# Plot actual vs. predicted values (lm smoothing method)
ggplot(df, aes(y = loyalty_points, x = predict(model2, df))) +
  geom_point(alpha=0.5) +
  stat_smooth(method = "lm") +  
  labs(y = 'Actual Loyalty Points', x = 'Predicted Loyalty Points') +
  ggtitle('Predicted vs. Actual Loyalty Points for model2') + theme_minimal()

# Create scenarios for prediction
scenarios2 <- data.frame(
  age = c(24, 37, 25, 66),
  remuneration = c(13.94, 14.76, 14.76, 15.58),
  spending_score = c(76, 6, 94, 3)
)

# Make predictions
scenarios2$predicted_loyalty <- predict(model2, newdata = scenarios2)
print(scenarios2)

################################################################################
################################################################################


# 6.4 Multiple Linear Regression (model3)

# Create the multiple linear regression model (with outliers removed)
model3 <- lm(loyalty_points ~ age + remuneration + spending_score, data  = df_no_outliers)

# Summarize the model
summary(model3)

# Possible reasons for deterioration in results: (i) outliers are valid and informative, 
# i.e. they reveal important pattern or subgroup that the model should account for.
# Due to small dataset, removing outliers distorted the regression fit.

# Plot actual vs. predicted values (lm smoothing method)
ggplot(df, aes(y = loyalty_points, x = predict(model3, df))) +
  geom_point(alpha=0.5) +
  stat_smooth(method = "lm") +  
  labs(y = 'Actual Loyalty Points', x = 'Predicted Loyalty Points') +
  ggtitle('Predicted vs. Actual Loyalty Points for model3') + theme_minimal()


###############################################################################
###############################################################################

# 6.5 Multiple Linear Regression (modelfe)

# Feature engineering (remuneration * spending_score = potential value of a customer)
df2 <- df %>%
  mutate(potential_value = spending_score * remuneration)

head(df2)

# Fit the linear regression model with the new variable
#modelfe <- lm(loyalty_points ~ age + remuneration + spending_score + potential_value, data = df2)
#modelfe <- lm(loyalty_points ~ remuneration + spending_score + potential_value, data = df2)
modelfe <- lm(loyalty_points ~ spending_score + potential_value, data = df2)

# View the model summary
summary(modelfe)

# Plot actual vs. predicted values (lm smoothing method)
ggplot(df, aes(y = loyalty_points, x = predict(modelfe, df2))) +
  geom_point(alpha=0.5) +
  stat_smooth(method = "lm") +  
  labs(y = 'Actual Loyalty Points', x = 'Predicted Loyalty Points') +
  ggtitle('Predicted vs. Actual Loyalty Points for modelfe') + theme_minimal()

# Check for multicollinearity
vif_valuesfe <- vif(modelfe)
cat("Variance Inflation Factors:\n")
print(vif_valuesfe)
# VIF between 1-5, therefore, low multicollinearity. 

# Test for heteroscedasticity
bp_testfe <- bptest(modelfe)
cat("Breusch-Pagan Test for Heteroscedasticity:\n")
print(bp_testfe)
# p < 0.05, therefore, evidence of heteroskedasticity in the regression model.

# Plot Residuals vs. Fitted Values
plot(fitted(modelfe), residuals(modelfe))
abline(h = 0, col = "red") + theme_minimal()

# Test for autocorrelation of residuals
dw_testfe <- durbinWatsonTest(modelfe)
cat("Durbin-Watson Test for Autocorrelation:\n")
print(dw_testfe)

# Check for normality of residuals
# Q-Q plot
qqnorm(residuals(modelfe))
qqline(residuals(modelfe), col = "red")   
xlab("Theoretical Quantiles")                                
ylab("Sample Quantiles")                                    

# Shapiro-Wilk normality test
shapiro_testfe <- shapiro.test(residuals(modelfe))
cat("Shapiro-Wilk Test for Normality:\n")
print(shapiro_test)

# Create scenarios for prediction
scenariosfe <- data.frame(
  spending_score = c(76, 6, 94, 3),
  potential_value = c(1059.44, 88.56, 1387.44, 46.74)
)

# Make predictions
scenariosfe$predicted_loyalty <- predict(modelfe, newdata = scenariosfe)
print(scenariosfe)

# 6.5a Evaluate predictive performance of modelfe

# Set seed for reproducibility
set.seed(42)

# Create an index for splitting (80% training, 20% testing)
sample_index <- sample(1:nrow(df2), size = round(0.8 * nrow(df2)))

# Split the data
train_data <- df2[sample_index, ]
test_data <- df2[-sample_index, ]

# Fit the model on training data
model_train <- lm(loyalty_points ~ spending_score + potential_value, data = train_data)

# Summarize the model
summary(model_train)

# Make predictions on test data
predictions_train_fe <- predict(model_train, newdata = test_data)

# Evaluate model performance
# Calculate Root Mean Squared Error (RMSE)
rmse <- sqrt(mean((test_data$loyalty_points - predictions_train_fe)^2))
print(paste("RMSE:", rmse))

# Calculate Mean Absolute Error (MAE)
mae <- mean(abs(test_data$loyalty_points - predictions_train_fe))
print(paste("MAE:", mae))

# Calculate R-squared on test data
ss_total <- sum((test_data$loyalty_points - mean(test_data$loyalty_points))^2)
ss_residual <- sum((test_data$loyalty_points - predictions_train_fe)^2)
r_squared <- 1 - (ss_residual / ss_total)
print(paste("Test R-squared:", r_squared))

# retaining the original variables captures the independent effects of each variable and
# the joint effect of their interaction.
# With only the interaction term, it would mean an expectation that the effect of 
# spending_score to be proportional to remuneration (and vice versa) which is unlikely.

# The model will account for both spending_score directly and its interaction 
# with remuneration through potential_value.Essentially, potential_value could be interpreted 
# as a proxy for the effect of both remuneration and spending_score on loyalty_points, 
# while spending_score captures its direct impact as well.

################################################################################
# Where the business should focus marketing efforts:

# Customers with high spending_score. spending_score is the most important predictor of
# loyalty points. Marketing efforts should prioritise customers with high spending_scores as
# they generate more loyalty_points.

# Feature engineering created potential_value variable (spending_score x remuneration) which
# appears to be valuable.
# Marketing campaigns should focus on customers with high potential_value scores.

# Whilst less influential than spending_score, remuneration still impacts loyalty. 
# Consider different marketing strategies for different income segments. 
# K-Means clusters work conducted during Python analysis may be used as basis for segments. 

###############################################################################
# How the loyalty program could be improved:

# Given strong relationship between spending _score and loyality_points,
# incentivise customers to increase spending_score by offering tiered-rewards

# Use feature_engineered metric potential_value to create personalised loyalty offers
# that maximise engagement based on both spending habits (spending_score) and remuneration.

# Retain valuable outliers as removing outliers worsened MLR model fit, suggesting the 266 high-value
# customers represent important segments. Design special loyalty tiers or exclusive benefits for
# customers with high loyalty_points.

################################################################################
################################################################################

## 6.6 T-Test for Gender Differences in Loyalty Points

# Null Hypothesis (H₀): There is no significant difference in mean loyalty points between female and male customers.
# Alternative Hypothesis (H₁): There is a significant difference in mean loyalty points between female and male customers.

# Extract loyalty points by gender
female_loyalty <- df$loyalty_points[df$gender == "Female"]
male_loyalty <- df$loyalty_points[df$gender == "Male"]

# Check sample sizes
female_count <- length(female_loyalty)
male_count <- length(male_loyalty)
cat("Female customers:", female_count, "\n")
cat("Male customers:", male_count, "\n")

# Calculate descriptive statistics
female_mean <- mean(female_loyalty)
male_mean <- mean(male_loyalty)
female_sd <- sd(female_loyalty)
male_sd <- sd(male_loyalty)

cat("Female mean loyalty points:", female_mean, "\n")
cat("Male mean loyalty points:", male_mean, "\n")
cat("Female SD:", female_sd, "\n")
cat("Male SD:", male_sd, "\n")

# Visualize distributions
par(mfrow=c(1,2))
hist(female_loyalty, main="Female Loyalty Points", col="pink")
hist(male_loyalty, main="Male Loyalty Points", col="lightblue")

# Check for normality
shapiro.test(female_loyalty)
shapiro.test(male_loyalty)

# QQ plots to visually assess normality
par(mfrow=c(1,2))
# Female QQ plot with punk color (hot pink)
qqnorm(female_loyalty, main="QQ Plot - Female Loyalty Points", col="pink", pch=19)
qqline(female_loyalty, col="purple", lwd=2)
# Male QQ plot with light blue
qqnorm(male_loyalty, main="QQ Plot - Male Male Loyalty Points", col="lightblue", pch=19)
qqline(male_loyalty, col="skyblue", lwd=2)

# Perform the t-test
t_test_result <- t.test(female_loyalty, male_loyalty)
print(t_test_result)

# Calculate effect size (Cohen's d)
pooled_sd <- sqrt(((female_count-1)*female_sd^2 + (male_count-1)*male_sd^2) / 
                    (female_count + male_count - 2))
cohen_d <- abs(female_mean - male_mean) / pooled_sd
cat("Cohen's d effect size:", cohen_d, "\n")

# The difference in loyalty points between women and men is extremely small (<0.2).

# T-Test commentary:
# p-value of 0.3664 (which is > 0,05). The raw difference of 52.58 in raw loyalty_points
# between the two genders is not statistically significant.
# Data violated the normality assumption. Use non-parametric to confirm results.

################################################################################
################################################################################

## 6.7 Non-parametric alternative (Wilcoxon-rank-sum test)

wilcox_test_result <- wilcox.test(female_loyalty, male_loyalty)
print(wilcox_test_result)

# Unlike the t-test, the Wilcoxon test indicates there is a statistically 
# significant difference in the distribution of loyalty points between female and male customers (p = 0.02675).
# Results of Wilcoxon test favoured as (i) loyalty_points data significantly deviated 
# from normal distribution as per Shapiro-Wilk test.
# (ii) Wilcoxon test compares median distributions rather than mean (less sensitive to outliers).
# Female customers tend to have higher loyalty point values than male customers, 
# and this difference is statistically significant (p = 0.02675).
# But effect size is extremely small. Cohen's d effect size.

# Recommendations: (i) female-oriented campaigns to acknowledge their consistent participation,
# (ii) male-oriented campaigns to boost their participation.
# (iii) A/B test different messaging styles & visuals based on gender (even if core offers similar).
# (iv) Avoid overly different approaches given extremely small effect size. 

################################################################################
################################################################################

# 6.8 Random forest (attempt to replicate rfg2 model on Python)

# Prepare data
set.seed(42)
train_index <- createDataPartition(df$loyalty_points, p = 0.7, list = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]

# Random Forest model
rf_model1 <- randomForest(loyalty_points ~ age + remuneration + spending_score,
                         data = train_data,
                         #ntree = 100,
                         importance = TRUE)
print(rf_model1)
importance(rf_model1)

# Predictions on test data
rf_predictions1 <- predict(rf_model1, test_data)
rf_rmse1 <- sqrt(mean((test_data$loyalty_points - rf_predictions1)^2))
rf_mape1 <- mean(abs((test_data$loyalty_points - rf_predictions1) / test_data$loyalty_points)) * 100

rf_rmse1
rf_mape1

# Create scenarios for prediction
scenariosrf1 <- data.frame(
  age = c(27),
  remuneration = c(19.68),
  spending_score = c(73)
)

# Make predictions
scenariosrf1$predicted_loyalty <- predict(rf_model1, newdata = scenariosrf1)
print(scenariosrf1)

# Random Forest commentary ntree=100:
# Training performance MSR (Mean Squared Error): 14,808.43
# Variance Explained 99.1% (excellent fit on training data)

# Test performance RMSE: 130.81, MAPE 10.48% (Mean Absolute Percentage error)
# i.e. predictions are off by 10.48% on average.

# Variable importance: spending_score most important.

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################