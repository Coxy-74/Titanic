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
set.seed(2903)
options(warn = -1)
model_glm_3 <- train(Survived ~ Sex.female + Pclass + Age, 
                     data = df_train, 
                     method = "glm", 
                     family = "binomial")
# Apply to test data and check results
df_test_predicted <- predict(model_rf_4,df_test)
df_test_predicted <- predict(model_glm_3,df_test)
cm_glm_3 <- confusionMatrix(data = df_test_predicted, 
                            reference = df_test$Survived,
                            mode = "everything",
                            positive = "1")
cm_glm3

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


