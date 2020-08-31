# library(caret)
df_pass <- read.csv("train.csv")

# Remove outlier (large fare)
df_pass <- df_pass[df_pass$Fare < 400,]

# Update missing values for Embarkation point
df_pass$Embarked[df_pass$Embarked == ""] <- "C"
df_pass$Embarked <- as.factor(as.character(df_pass$Embarked))

# Combine siblings and parent/child columns into single "family size" column
df_pass$FamSize <- df_pass$SibSp + df_pass$Parch

# Remove Name, Passenger Id, Ticket, SibSp, Parch and Cabin columns
df_pass <- df_pass[,-c(1,4,7,8,9,11)]

# Split into training and test data
set.seed(2903)
trainRowNumbers <- createDataPartition(df_pass$Survived, p = 0.7, list = FALSE)
df_train <- df_pass[trainRowNumbers,]
df_test <- df_pass[-trainRowNumbers,]

# Store Survived value for later use
train_y <- as.data.frame(list(as.factor(df_train$Survived)), col.names = c("Survived"))
test_y <- as.data.frame(list(as.factor(df_test$Survived)), col.names = c("Survived"))

# Impute missing rows
missingDataModel <- preProcess(df_train[,-1],method='bagImpute')

# Data Transformations (training data)
  # Missing values
    df_train <- predict(missingDataModel, newdata = df_train[,-1])
  # Categorical features to numeric features
    ohe_model <- dummyVars(~ ., data = df_train)
    df_train <- data.frame(predict(ohe_model, newdata = df_train))
  # Centre and Scale data
    scale_model <- preProcess(df_train, method = c("center","scale"))
    df_train <- predict(scale_model, newdata = df_train)

# Data Transformations (test data)
df_test <- predict(missingDataModel, newdata = df_test[,-1])
df_test <- data.frame(predict(ohe_model, newdata = df_test))
df_test <- predict(scale_model, newdata = df_test)

# Generate random forest model
set.seed(2903)
options(warn = -1)
model_rf_4 <- train(Survived ~ Sex.female + Sex.male + Fare + Age + Pclass, 
                    data = bind_cols(train_y,df_train), 
                    method = "rf", 
                    prox = TRUE)

# Apply to test data and check results
df_test_predicted <- predict(model_rf_4,df_test)
cm_rf_4 <- confusionMatrix(data = df_test_predicted, 
                           reference = test_y$Survived,
                           mode = "everything",
                           positive = "1")
cm_rf_4

# Get Kaggle data and transform the same way
df_kaggle <- read.csv("test.csv")
df_kaggle$Embarked[df_kaggle$Embarked == ""] <- "C"
df_kaggle$FamSize <- df_kaggle$SibSp + df_kaggle$Parch
PassengerId <- df_kaggle$PassengerId
df_kaggle <- df_kaggle[,-c(1,3,6,7,8,10)]
df_kaggle <- predict(missingDataModel, newdata = df_kaggle)
df_kaggle <- data.frame(predict(ohe_model, newdata = df_kaggle))
df_kaggle <- predict(scale_model, newdata = df_kaggle)

# Apply Kaggle data to model
df_kaggle_predicted <- predict(model_rf_4,df_kaggle)
df_kaggle_predicted_df <- as.data.frame(PassengerId)
df_kaggle_predicted_df$Survived <- df_kaggle_predicted
write.csv(df_kaggle_predicted_df, "predictions.csv", row.names = FALSE)


# Now try xgboost
tune_grid <- expand.grid(
    nrounds = 100 # seq(from = 50, to = 500, by = 50)        # default = 100
  , eta = 0.5 # c(0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9) # default = 0.3 range = 0-1   typically 0.01 - 0.3
  , max_depth = 6 # c(2, 3, 4, 5, 6, 7, 10, 100)             # default = 6   range = 0-Inf
  , gamma = 5 # c(0, 0.05, 0.1, 0.5, 0.7, 1, 3, 5)           # default = 0   range = 0-Inf higher = higher reg'n
  , colsample_bytree = 1 # c(0.3, 0.5, 0.7, 0.9, 1)          # default = 1   range = 0-1   typically 0.5 - 0.9
  , min_child_weight = 1 # c(1, 2, 3, 5, 10, 100)            # default = 1   range = 0-Inf
  , subsample = 1 # c(0.3, 0.5, 0.65, 0.8, 1.0)              # default = 1   range = 0-1   typically 0.5 - 0.8
)

run_xgb()

tune_control <- caret::trainControl(
    method = "cv", # cross-validation
    number = 5, # with n folds 
    verboseIter = FALSE, # no training log
    allowParallel = TRUE # FALSE for reproducible results 
)

set.seed(2903)
model_xgb <- caret::train(
    x = df_train,
    y = train_y$Survived,
    trControl = tune_control,
    tuneGrid = tune_grid,
    method = "xgbTree",
    verbose = TRUE
)

model_xgb$bestTune

df_test_predicted <- predict(model_xgb,df_test)
cm_xgb <- confusionMatrix(data = df_test_predicted, 
                           reference = test_y$Survived,
                           mode = "everything",
                           positive = "1")
cm_xgb

# Apply Kaggle data to model
df_kaggle_predicted <- predict(model_xgb,df_kaggle)
df_kaggle_predicted_df <- as.data.frame(PassengerId)
df_kaggle_predicted_df$Survived <- df_kaggle_predicted
write.csv(df_kaggle_predicted_df, "predictions.csv", row.names = FALSE)


run_xgb <- function(parms = tune_grid, control = tune_control, feats = df_train, vals = train_y$Survived) {
    set.seed(2903)
    model_xgb <- caret::train(
        x = feats,
        y = vals,
        trControl = control,
        tuneGrid = parms,
        method = "xgbTree",
        verbose = TRUE
    )
    df_train_predicted <- predict(model_xgb,df_train)
    cm_xgb_train <- confusionMatrix(data = df_train_predicted, 
                                    reference = train_y$Survived,
                                    mode = "everything",
                                    positive = "1")
    
    df_test_predicted <- predict(model_xgb,df_test)
    cm_xgb_test <- confusionMatrix(data = df_test_predicted, 
                                   reference = test_y$Survived,
                                   mode = "everything",
                                   positive = "1")
    print(model_xgb$bestTune)
    print(paste('Accuracy Train:', cm_xgb_train$overall[[1]]))
    print(paste('Accuracy Test: ', cm_xgb_test$overall[[1]]))
    }
