#Loading the data into R
library(readr)
cc_fraud_trainingset <- read_csv("~/ADA_Assignemnt2/cc_fraud_trainingset.csv")
View(cc_fraud_trainingset)
cc_fraud_testset <- read_csv("~/ADA_Assignemnt2/cc_fraud_testset.csv")
View(cc_fraud_testset)

## Handling Missing Values
install.packages("mice")
library(mice)

#Imputing missing values using Mice package

##Training data

imputed_data <-mice(data =cc_fraud_trainingset , m = 5, method = "mean", maxit=10)
cc_fraud_trainingsetImputed <-complete(imputed_data, 2)
md.pattern(cc_fraud_trainingsetImputed)
train<-cc_fraud_trainingsetImputed

##Testing

imputed_data2 <-mice(data =cc_fraud_testset, m = 5, method = "mean", maxit=10)
cc_fraud_testsetImputed <-complete(imputed_data2, 2)
md.pattern(cc_fraud_testsetImputed)
test<-cc_fraud_testsetImputed 


#To combine datasets and preserve information from which datasets 
#records are we need to create the artificial variable
#the variable will be called isTrain and will have value "yes" for training records
#and "no" for testing records

train$isTrain<-"yes"
test$isTrain<-"no"

#combine datasets
comb<- rbind(train, test)

#use comb variabke for EDA nad preprocessing analysis.

#remember isTrain is indicating from which dataset the record comes from. Do not report that variable
#in analysis
# Load required packages
library(ggplot2)
library(dplyr)
library(factoextra)

# Read the dataset (replace "comb.csv" with the actual file name of your dataset)
# Data introduction and presentation
cat("Credit Card Fraud Dataset:\n")
cat("Number of observations: ", nrow(comb), "\n")
cat("Number of variables: ", ncol(comb), "\n")
cat("\n")

cat("Variables in the dataset:\n")
str(comb)
cat("\n")

# Identify and present the target variable in terms of distribution
target_variable <- "Class"  # Replace "Class" with your actual target variable name

cat("Distribution of the target variable (", target_variable, "):\n")
summary(comb[[target_variable]])
cat("\n")

# Select only numeric variables for PCA
numeric_vars <- comb %>% select_if(is.numeric)

# Perform PCA on numeric variables
pca_data <- prcomp(numeric_vars, scale. = TRUE)

# Extract the first two principal components
pc1 <- pca_data$x[, 1]
pc2 <- pca_data$x[, 2]

# Create a scatter plot of the data using the first two principal components
plot_data <- data.frame(PC1 = pc1, PC2 = pc2, Class = comb$Class)
ggplot(plot_data, aes(x = PC1, y = PC2, color = factor(Class))) +
  geom_point() +
  labs(color = "Class") +
  theme_minimal()

# when you finish task #undertanding data and #preprocess and this is what you submit for PART I

#for Part II
#you need to split  comb back to test and train before #task 6

train<-comb[comb$isTrain=="yes",]
test<-comb[comb$isTest=="no", ]

#remove variable isTrainfrom both train and test
train$isTrain<-NULL
test$isTrain<-NULL
##################################################
# Load required packages
library(ggplot2)
library(dplyr)
library(tidyr)

# Numeric variables in the dataset
numeric_vars <- comb %>% select_if(is.numeric)

# Generate a grid of boxplots for each numeric variable
grid <- numeric_vars %>% 
  pivot_longer(everything(), names_to = "Variable", values_to = "Value") %>%
  ggplot(aes(x = Variable, y = Value)) +
  geom_boxplot() +
  labs(x = "Variable", y = "Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Display the grid of boxplots
print(grid)

# Analyze outliers for the most influential variable
most_influential_variable <- "Amount"  # Replace with the most influential variable discovered during EDA

# Calculate deviation from the mean for outliers
mean_value <- mean(comb[[most_influential_variable]])
outliers <- comb %>% 
  filter(comb[[most_influential_variable]] > mean_value) 

# Calculate the number of outliers per class
outliers_per_class <- table(outliers$Class)

# Print the analysis results
cat("Analysis of Outliers for Variable:", most_influential_variable, "\n")
cat("Deviation from Mean:", mean_value, "\n")
cat("Number of Outliers per Class:\n")
print(outliers_per_class)


##############################################correlation
# Load required packages
library(dplyr)
library(tibble)

# Select only numeric variables for correlation analysis
numeric_vars <- comb %>% select_if(is.numeric)

# Calculate correlation matrix
cor_matrix <- cor(numeric_vars)

# Display the correlation matrix
print(cor_matrix)

# Identify high correlation between variables and the target variable
target_variable <- "Class"  # Replace "Class" with your actual target variable name
target_corr <- cor_matrix[, target_variable]
high_corr_vars <- names(target_corr[abs(target_corr) > 0.8])  # Change threshold to 0.8 or 0.9 as per your choice

# Identify high correlation among predictors
high_corr_predictors <- numeric_vars %>% 
  select(all_of(high_corr_vars)) %>%
  cor() %>%
  as.data.frame() %>%
  tibble::rownames_to_column(var = "Variable") %>%
  filter(rowSums(abs(.[-1])) > 0.8)  # threshold to 0.8 

# Remove variables with high correlation
removed_vars <- high_corr_predictors$Variable
comb <- comb %>% select(-all_of(removed_vars))

# Print the list of removed variables
cat("Removed variables due to high correlation:\n")
print(removed_vars)

################################################## scalability
# Load required packages
library(dplyr)

# Select only numerical variables for scaling
numeric_vars <- comb %>% select_if(is.numeric)

# Scale the numerical variables
scaled_vars <- scale(numeric_vars)

# Replace the original numerical variables with the scaled variables in the dataset
comb <- comb %>% 
  select(-starts_with("V")) %>%  # Exclude the "V" variables as they are already PCA components
  bind_cols(as.data.frame(scaled_vars))


##############################problematic variables
# Remove identifier variables
identifier_vars <- c("Identifier_Variable1", "Identifier_Variable2", ...)  # Replace with actual identifier variable names
comb <- comb[, !colnames(comb) %in% identifier_vars]

# Remove zero variance variables
zero_variance_vars <- names(comb)[sapply(comb, var) == 0]
comb <- comb[, !names(comb) %in% zero_variance_vars]

# Remove duplicate rows
comb <- comb[!duplicated(comb), ]

###PART II
########
library(caret)
library(pROC)
library(rpart)
library(e1071)
library(class)
library(randomForest)
library(kknn)
library(nnet)
library(tidymodels)

#Set the seed for reproducibility
set.seed(123)
split_comb<- initial_split(comb)
train <- training(split_comb)
test <- testing(split_comb)
##################################
# Load required packages
library(caret)
library(e1071)
library(randomForest)
library(class)
library(neuralnet)
library(pROC)

# Fit the models on the training data
# Logistic Regression
logistic_model <- train(Class ~ ., data = train, method = "glm",
                        family = "binomial")

# Naive Bayes
naive_bayes_model <- train(Class ~ ., data = train, method = "naive_bayes")

# Random Forest
rf_model <- train(Class ~ ., data = train, method = "rf",
                  ntree = 500, mtry = sqrt(ncol(train)-1))

# K-Nearest Neighbors
knn_model <- train(Class ~ ., data = train, method = "knn", 
                   tuneLength = 10, preProcess = c("center", "scale"))

# Neural Network
nn_model <- train(Class ~ ., data = train, method = "neuralnet", 
                  linear.output = FALSE, hidden = c(10, 5))

# Evaluate the models using 5-fold cross-validation
set.seed(123)
ctrl <- trainControl(method = "cv", number = 5)
logistic_cv <- train(Class ~ ., data = train, method = "glm", 
                     family = "binomial", trControl = ctrl)
naive_bayes_cv <- train(Class ~ ., data = train, method = "naive_bayes", 
                        trControl = ctrl)
rf_cv <- train(Class ~ ., data = train, method = "rf", 
               ntree = 500, mtry = sqrt(ncol(train)-1), trControl = ctrl)
knn_cv <- train(Class ~ ., data = train, method = "knn", 
                tuneLength = 10, preProcess = c("center", "scale"), trControl = ctrl)
nn_cv <- train(Class ~ ., data = train, method = "neuralnet", 
               linear.output = FALSE, hidden = c(10, 5), trControl = ctrl)

# Choose the best performing model based on the cross-validation results
models <- list(logistic_cv, naive_bayes_cv, rf_cv, knn_cv, nn_cv)
results <- resamples(models)
summary(results)
bwplot(results)
best_model <- getModelMetaData(results, "best")$model

# Evaluate the selected model using appropriate performance metrics
# Confusion matrix
confusionMatrix(predict(best_model, newdata = test),
                test$Class)

# ROC curve and AUC
roc_obj <- roc(test$Class, predict(best_model, newdata = test, type = "prob")[,2])
plot(roc_obj)
auc(roc_obj)

# Fine-tune the model hyperparameters to improve the model's performance
# Random Forest
rf_grid <- expand.grid(mtry = c(2, 5, 10, 15))
rf_tune <- train(Class ~ ., data = train, method = "rf",
                 ntree = 500, tuneGrid = rf_grid, trControl = ctrl)
best_rf_model <- rf_tune$bestTune

# Tabulate the key performances and AUC values of each model
results_table <- data.frame(Model = c("Logistic Regression", "Naive Bayes", 
                                      "Random Forest", "K-Nearest Neighbors", 
                                      "Neural Network"),
                            Accuracy = sapply(models, function(m) {
                              max(m$results$Accuracy)
                            }),
                            AUC = sapply(models, function(m) {
                              roc_obj <- roc(test$Class, 
                                             predict(m, newdata = test, type = "prob")[,2])
                              auc(roc_obj)
                            }))
print(results_table)


# Create a data frame to store the performance metrics
performance_table <- data.frame(
  Model = c("Logistic Regression", "Naive Bayes", "Random Forest", "K-Nearest Neighbors", "Neural Network"),
  R2 = c(lr_r2, nb_r2, rf_r2, knn_r2, nn_r2),
  Adjust_R2 = c(lr_adjusted_r2, nb_adjusted_r2, rf_adjusted_r2, knn_adjusted_r2, nn_adjusted_r2),
  MSE = c(lr_mse, nb_mse, rf_mse, knn_mse, nn_mse),
  RMSE = c(lr_rmse, nb_rmse, rf_rmse, knn_rmse, nn_rmse),
  MAE = c(lr_mae, nb_mae, rf_mae, knn_mae, nn_mae)
)

# Print the performance table
print(performance_table)
