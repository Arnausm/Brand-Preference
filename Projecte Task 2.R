
# Libraries

library(ggplot2,lattice,readr)
library(caret)
library(C50)
library(reshape2)

#### Data preprocessing ####

CR<- read.csv("C:/Users/Arnau/Documents/Task 2/CompleteResponses.csv")

CRN<- read.csv("C:/Users/Arnau/Documents/Task 2/SurveyIncomplete.csv")

  # Defining levels of factors

CR$elevel<- factor(CR$elevel,levels = c(0,1,2,3,4),labels = c(
                    "Primary School",
                    "High School",
                    "One Year Degree",
                    "College Degree",
                    "Professional Degree"))
CR$car<- factor(CR$car,
                levels = c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20),
                labels = c("BMW","Buick","Cadillac","Chevrolet","Chrysler",
                           "Dodge","Ford","Honda","Hyundai","Jeep","Kia","Lincoln",
                           "Mazda","Mercedes","Mitsubishi","Nissan","Ram","Subaru","Toyota", "None"))
CR$zipcode <- as.factor(CR$zipcode)
# Values (0,1) doesn't mean that it's a logical value.
# It's a factor with 2 levels

CR$brand <- factor(CR$brand, levels = c(0,1), labels = c("Acer","Sony"))


CRN$elevel<- factor(CRN$elevel,levels = c(0,1,2,3,4),labels = c(
  "Primary School",
  "High School",
  "One Year Degree",
  "College Degree",
  "Professional Degree"))
CRN$car<- factor(CRN$car,
                levels = c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20),
                labels = c("BMW","Buick","Cadillac","Chevrolet","Chrysler",
                           "Dodge","Ford","Honda","Hyundai","Jeep","Kia","Lincoln",
                           "Mazda","Mercedes","Mitsubishi","Nissan","Ram","Subaru","Toyota", "None"))
CRN$zipcode <- factor(CRN$zipcode)
CRN$brand <- factor(CRN$brand, levels = c(0,1), labels = c("Acer","Sony"))

## Remove columns ##

CR$car <- NULL
CR$credit <- NULL
CR$elevel <- NULL
CR$zipcode <- NULL

CRN$car <- NULL
CRN$credit <- NULL
CRN$elevel <- NULL
CRN$zipcode <- NULL

#### Random forest with Automatic Grid ####

# "createDataPartition" is needed because respect the proportion of the population
# "Stratified" randomly split of the data. "Sample" take a random sample.

set.seed(998)
inTraining <- createDataPartition(CR$brand, p = .75, list = FALSE)
training <- CR[inTraining,]
testing <- CR[-inTraining,]
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)
system.time(RandomForest1 <- train(brand~.,
                                   data = training,
                                   method = "rf",
                                   trControl=fitControl,
                                   tuneLength = 2))
# Prediction training
RF1_pred_train <- predict(RandomForest1,training)

# Confusion matrix. Evaluate obtained results
CM_RF1_pred_train <- confusionMatrix(RF1_pred_train,training$brand)
CM_RF1_pred_train

# Prediction testing
RF1_pred_test <- predict(RandomForest1,testing)

# Confusion matrix. Evaluate obtained results
CM_RF1_pred_test <- confusionMatrix(RF1_pred_test,testing$brand)
CM_RF1_pred_test

# Importance of variables
RF1varImp <- varImp(RandomForest1, scale = FALSE)
RF1varImp
plot(RF1varImp)

##### Random forest Manual Grid ####

set.seed(998)
inTraining <- createDataPartition(CR$brand, p = .75, list = FALSE)
training <- CR[inTraining,]
testing <- CR[-inTraining,]
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)
rfGrid <- expand.grid(mtry=c(3))
system.time(RandomForest2 <- train(brand~., data = training,
                                   method = "rf",
                                   trControl=fitControl,
                                   tuneGrid=rfGrid))
# Prediction training
RF2_pred_train <- predict(RandomForest2,training)

# Evaluate obtained results
CM_RF2_pred_train <- confusionMatrix(RF2_pred_train,training$brand)
CM_RF2_pred_train

# Prediction testing
RF2_pred_test <- predict(RandomForest2,testing)

# Evaluate obtained results
CM_RF2_pred_test <- confusionMatrix(RF2_pred_test,testing$brand)
CM_RF2_pred_test

# Importance of variables
RF2varImp <- varImp(RandomForest2, scale = FALSE)
plot(RF2varImp)

#### mtry ####

# mtry = caret::var_seq(p = ncol(x), 
#                     classification = is.factor(y), 
#                     len = len)

mtry = caret::var_seq(p = 3, 
                      classification = "brand", 
                      len = 5)
mtry # 2 3 

#### Decision tree rpart ####

set.seed(998)
inTraining <- createDataPartition(CR$brand, p = .75, list = FALSE)
training <- CR[inTraining,]
testing <- CR[-inTraining,]
fitControl_distre <- trainControl(method = "repeatedcv", 
                                  number = 10, 
                                  repeats = 1,
                                  classProbs = TRUE)
DecisionTree <- train(brand~.,
                      data = training, 
                      method = "rpart",
                      trControl=fitControl_distre,
                      tuneLength = 2)
DecisionTree

# Importance of variables
varImp(DecisionTree, scale = TRUE)

# Prediction training
DT_pred_train <- predict(DecisionTree,training)

# Evaluate results obtained
CM_DT_pred_train <- confusionMatrix(DT_pred_train,training$brand)
CM_DT_pred_train
# Prediction testing
DT_pred_test <- predict(DecisionTree,testing)
# Evaluate results obtained
CM_DT_pred_test <- confusionMatrix(DT_pred_test,testing$brand)
CM_DT_pred_test

## Visual representation decision tree
# FONT: 2 to bold face, 3 to italic and 4 to bold italic 

plot(DecisionTree$finalModel, uniform=TRUE,
     main="Classification Tree")
text(DecisionTree$finalModel, use.n.=TRUE, font=4, all=TRUE, cex=.7)

summary(DecisionTree$finalModel)

#### Decision Tree c5.0 ####

set.seed(998)
inTraining <- createDataPartition(CR$brand, p = .75, list = FALSE)
training <- CR[inTraining,]
testing <- CR[-inTraining,]
fitControl_distre2 <- trainControl(method = "repeatedcv", number = 10, repeats = 1,classProbs = TRUE)
DT5.0 <- train(brand~.,
              data = training, 
              method = "C5.0",
              trControl=fitControl_distre2,
              tuneLength = 2)
DT5.0

# Importance of variables
varImp(DT5.0, scale = TRUE)
# Prediction training
DT5.0_pred_train <- predict(DT5.0,training)
# Evaluate results
CM_DT5.0_pred_train <- confusionMatrix(DT5.0_pred_train,training$brand)
CM_DT5.0_pred_train
# Prediction testing
DT5.0_pred_test <- predict(DT5.0,testing)
# Evaluate results
CM_DT_pred_test <- confusionMatrix(DT5.0_pred_test,testing$brand)
CM_DT_pred_test

#### Final prediction Random forest1 ####
Surveyimcomplete2 <- predict(RandomForest1,CRN)
Surveyimcomplete2
summary(Surveyimcomplete2)

## Postresample Random forest1
# postResample(pred,obs)
postResample(Surveyimcomplete2,testing$brand)

#### Final prediction Decision Tree c5.0 ####
Surveyimcomplete1 <- predict(DT5.0,CRN)
summary(Surveyimcomplete1)

## Postresample Decision Tree c5.0
# postResample(pred,obs)
postResample(Surveyimcomplete1,testing$brand)

#### rfe recursive feature elimination ####

set.seed(998)
inTraining <- createDataPartition(CR$brand, p = .75, list = FALSE)
Training <- CR[inTraining,]
testing <-- CR[-inTraining,]

ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)



lmProfile <- rfe(x=Training, y=CR$Brand,
                 sizes = subsets,
                 rfeControl = ctrl)

lmProfile

# rfe 2

control <- rfeControl(functions=rfFuncs, method="cv", number=10)
results <- rfe(feature_selection_data
               , feature_selection_target$value
               , sizes = c(1:12)
               , rfeControl = control)

#### Histograms #### 
# COLORS = LIGHT BLUE, DARK BLUE.
# Line types = blank, solid, dashed, dotted, dotdash, longdash, twodash
histogram(Surveyimcomplete1, 
          main = "Percent of total",
          xlab = "",
          border = c(Acer ="dark blue", Sony = " dark green"),
          col = c(Acer ="light blue", Sony ="light green"),
          type = "percent")

hist(CR$salary, col = "blue", main = "Histogram of Salary", xlab = "Salary")
hist(CR$age, col = "red", main = "Histogram of Age", xlab = "Age")

#### Density Graphics ####

qplot(salary, data=CR, geom="density", fill= brand, alpha=I(.5), 
      main="Distribution of Salary", xlab="Salary",
      ylab="Density")

ggplot(CR, aes(x = salary,y= age,fill=brand)) + 
  geom_point(aes(color=brand)) + 
  xlab("Salary") +
  ylab("Age")

# training 

ggplot(training, aes(x = salary,y= age,fill=brand)) + 
  geom_point(aes(color=brand)) + 
  xlab("Salary") +
  ylab("Age")

# testing

ggplot(testing, aes(x = salary,y= age,fill=brand)) + 
  geom_point(aes(color=brand)) + 
  xlab("Salary") +
  ylab("Age")


qplot(salary, age, data=CR, geom=c("boxplot"), 
      fill=brand, main="",
      xlab="Salary", ylab="Age")


ggplot(CR, aes(x = age)) +   geom_density(aes(color = brand)) + xlab("Age") + ylab("Density")

