setwd("/media/gimin/vol2/DS_Lab/Titanic")

library(doMC)
registerDoMC(cores = 2)

# read data
# train <- read.csv(file.choose(), na.strings = c("NA", "", " ", NULL))
# test <- read.csv(file.choose(), na.strings = c("NA", "", " ", NULL))

train <- read.csv("data/train.csv", na.strings = c("NA", "", " ", NULL))
test <- read.csv("data/test.csv", na.strings = c("NA", "", " ", NULL))

#-----------------------------------------  Plotting ---------------------------------------------
library(Amelia)
missmap(train, main="Titanic Training Data - Missings Map", legend=F) # plot NAs

barplot(table(train$Survived), names.arg = c("Died", "Survided"), main="Survived (passenger fate)")
barplot(table(train$Pclass), names.arg = c("1st class", "2nd class", "3rd class"),
        main="Pclass (passenger traveling class)", col="firebrick")
barplot(table(train$Sex), main="Sex (gender)", col=c("pink", "blue"))
hist(train$Age, main="Age", xlab = NULL, col="brown", density = 20, breaks = 40)
plot(table(train$Age))
barplot(table(train$SibSp), main="SibSp (siblings + spouse aboard)", col="darkblue")
barplot(table(train$Parch), main="Parch (parents + kids aboard)", col="gray50")
hist(train$Fare, main="Fare (fee paid for ticket[s])", xlab = NULL, col="darkgreen")
barplot(table(train$Embarked), names.arg = c("Cherbourg", "Queenstown", "Southampton"), main="Embarked (port of embarkation)", col="sienna")
mosaicplot(train$Pclass ~ train$Survived, 
           main="Passenger Survival vs Traveling Class", shade=FALSE, 
           color=TRUE, xlab="Pclass", ylab="Survived, 1 = yes / 0 = No")
mosaicplot(train$Sex ~ train$Survived, 
           main="Passenger Fate by Sex", shade=F, 
           color=TRUE, xlab="Sex", ylab="Survived, 1 = yes / 0 = No")
boxplot(train$Age~train$Sex+train$Survived, varwidth = T, 
        names = c("F/Died", "Male/Died", "F/Survived", "M/Survived"), border = c("red", "red", "green", "green"))
#-----------------------------------------------------------------------------------------------------

library(caret)
library(plyr)
library(dplyr)
library(stringr)
library(party); library(partykit)

str(train)
### Notes:
#  VARIABLE        MEANING
#  Sibsp           Number of Siblings/Spouses Aboard
#  Parch           Number of Parents/Children Aboard
#  Ticket          Ticket Number
#  Fare            Passenger Fare
#  Cabin           Cabin
#  Embarked        Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
summary(train)

#labelName <- 'Survived'
#predictors <- names(train)[names(train) != labelName]

# Take a look at class distributions. A bit left(0=perished) skweed
table(train$Survived)
prop.table(table(train$Survived))

#-------------- Data Wrangling-----------------------------------------
# combine train & test into 1 data set to save code lines
nrow_spl <- dim(train)[1]
trainSurvived <- as.factor(train$Survived)
allData <- train %>% select(- Survived) %>% bind_rows(test)
str(allData)

# Drop PassengerID, not a predictor 
allData$PassengerId <- NULL

# Cabin has too many NAs, to try to impute so drop it
allData$Cabin <- NULL

# Do factors the variables with few discreet values

allData$Pclass <- as.factor(allData$Pclass)
allData$SibSp <- as.factor(allData$SibSp)
allData$Parch <- as.factor(allData$Parch)
allData$Embarked <- as.factor(allData$Embarked)
allData$Ticket <- as.factor(allData$Ticket)

# Replace Name by a Factor: Mr, Mrs, Miss
# Name is in format "Surname, Title (Mr, Miss, Mrs, other) First name". Keep only the title (if other = Mr)
allData$Title <- regmatches(allData$Name, regexpr("\\b[A-z]{1,12}\\.", allData$Name))
allData$Title <- sapply(allData$Title, function(x){gsub("\\.", "", x)}) # get rid of the dots

# summarize titles to Mr, Mrs, Miss
allData$Title[which(allData$Title %in% c("Mme", "Lady", "Countess", "Dona", "Ms"))] <- "Mrs"
allData$Title[which(allData$Title %in% c("Mlle"))] <- "Miss"
allData$Title[which(allData$Title %in% c("Master", "Don", "Rev", "Major", "Sir", "Col", "Capt", "Jonkheer"))] <- "Mr"
allData$Title[which(allData$Title == "Dr" & allData$Sex == "male")] <- "Mr"
allData$Title[which(allData$Title == "Dr" & allData$Sex == "female")] <- "Mrs"
table(allData$Title)

# remove Name columns
allData$Name <- NULL
allData$Title <- as.factor(allData$Title) # transform to factor

# ------------------Ticket---------------------------
# unique ticket numbers < number of passengers, because passengers share some cabins (families etc.)
# Assign a number of passengers per Ticket
sum(length(unique(allData$Ticket)))

library(data.table)
allData <- as.data.table(allData)
d <- allData[, .N, by = Ticket] # count the number of rows for every group in Ticket
allData$Ticket <- sapply(allData$Ticket, function(x){d[Ticket == x, N]}) # replace in Ticket the number of passengers included
allData$Ticket <- as.factor(allData$Ticket)

# ---------------------------------------------------------

### NAs in train data set
# Age: 177 (~ 20%)
# Embarked: 2 (~ 2%)   ---> replace with most probabletrain

# deal with Embarked, replace with most frequent embarkation point
na_Embarked <- which(is.na(allData$Embarked))
table(allData$Embarked)
allData[c(na_Embarked), "Embarked"] <- c("S", "S")

# deal with age, replace with mean values
# mean Age values
meanMrs <- mean(subset(allData, Title == "Mrs")$Age, na.rm = T)
meanMiss <- mean(subset(allData, Title == "Miss")$Age, na.rm = T)
meanMr <- mean(subset(allData, Title == "Mr")$Age, na.rm = T)

fillAge <- function(r){ # function to replace missing ages
  if (is.na(r[3])){ # if there is a missing age, replace accordingly
        if (r[9] == "Mr"){
          r[3] <- meanMr
        } else if (r[9] == "Miss"){
          r[3] <- meanMiss
        } else if (r[9] == "Mrs") r[3] <- meanMrs
      } else return(r[3])
}

allData$Age <- as.numeric(apply(allData, MARGIN = 1, FUN = fillAge))

# deal with Fare
na_Fare <- which(is.na(allData$Fare))
allData[c(na_Fare), "Fare"] <- 14.45 # replace with median

# ---------------------------------------- End preprocessing -----------------------------------------


## -------------------------------------- Prediction models -------------------------------------------
set.seed(1111)

## (1)  Turn everything into categorical variables and use Bayesian
allCategorical <- allData

# SibSp and Parch are int and easily transformed (the idea holds to be factors in context)
allCategorical$SibSp <- as.factor(allCategorical$SibSp)
allCategorical$Parch <- as.factor(allCategorical$Parch)

# Binning Age
barplot(table(round(allCategorical$Age, 0))); title("table(allCategorical$Age)")
allCategorical$Age_range <- cut(allCategorical$Age, c(0, 12, 20.0, 40.0, 60.0, 80.0))

barplot(table(round(allCategorical$Fare, 0))); title("table(allCategorical$Fare)")
allCategorical$Fare_range <- cut(allCategorical$Fare, c(0, 20, 30, 50, 100, 520),
                         include.lowest = T, right = F) # >0 starts at 4.0125

allCategorical[, c("Age", "Fare") := NULL] #remove Age and Fare numerical columns

# Split allCategorical into train and test set
train1 <- allCategorical[c(1:nrow_spl), ]; train1$Survived <- trainSurvived
test1 <- allCategorical[- c(1:nrow_spl), ]

# train1$Survived <- ifelse(train1$Survived == 1, "yes", "no"); train1$Survived <- as.factor(train1$Survived)

# transform to data.frame only to use in caret 
train1 <- as.data.frame(train1)
test1 <-as.data.frame(test1)

labelName <- 'Survived'
predictors_1 <- names(train1)[names(train1) != labelName]
# predictors_1 <- names(train1)[c(2, 3, 4, 5)]

# fit naiveBayes
myCtrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5)

naiveBayes <- train(x = train1[, predictors_1], y = train1[, labelName], method = 'nb', trControl = myCtrl)
naiveBayes$results
naiveBayes$bestTune

# fit rpart
rpart <- train(x = train1[, predictors_1], y = train1[, labelName], method = 'rpart', trControl = myCtrl)
rpart$results

# fit C5.0
c_50 <- train(x = train1[, predictors_1], y = train1[, labelName], method = 'C5.0', trControl = myCtrl)
c_50$results

# -----------------------------------------------------------------------------------
# Split allData into train and test set
# allData <- as.data.frame(allData)
train <- allData[c(1:nrow_spl), ]; train$Survived <- trainSurvived
test <- allData[- c(1:nrow_spl), ]

# rearrange the columns
setcolorder(train, c("Pclass", "Title", "Sex", "SibSp", "Parch", "Ticket", "Embarked", "Age", "Fare", "Survived"))
setcolorder(test, c("Pclass", "Title", "Sex", "SibSp", "Parch", "Ticket", "Embarked", "Age", "Fare"))

## (2) with the train set (factors and nums)
#labelName <- 'Survived'
train$Survived <- ifelse(train$Survived == 1, "yes", "no");  train$Survived <- as.factor(train$Survived)
predictors_2 <- names(train)[names(train) != labelName]

# transform back to data.grame class only so it works with caret indexing
train <- as.data.frame(train)
test <- as.data.frame(test)

c5o <- train(x = train[, predictors_2], y = train[, labelName], method = 'C5.0', trControl = myCtrl)
c5o$results
c5o$bestTune

rfo <- train(x = train[, predictors_2], y = train[, labelName], method = 'rf', trControl = myCtrl)
rfo$results  # this gives the best results on the train set with 10-fold CV
rfo$bestTune
# pp <- predict(object = rfo, test)

cTree <- train(x = train[, predictors_2], y = train[, labelName], method = 'ctree', trControl = myCtrl)
cTree$results

treeBag <- train(x = train[, predictors_2], y = train[, labelName], method = 'treebag', trControl = myCtrl)

gradBM <- train(x = train[, predictors_2], y = train[, labelName], method = 'gbm', trControl = myCtrl)
gradBM$results

# ----------------------------------------------------------------------------------------------------
# Let's do some ensembles
splitTrain <- createDataPartition(train[, labelName], p = 0.2, list = F)
training <- train[splitTrain, ]
blender <- train[- splitTrain,] 

##### {Models}
#(m1) C5.0
c5o <- train(x = training[, predictors_2], y = training[, labelName], method = 'C5.0')
#(m2) Random Forest
rfo <- train(x = training[, predictors_2], y = training[, labelName], method = 'rf')
#(m3) Conditiona Tree
cTree <- train(x = training[, predictors_2], y = training[, labelName], method = 'ctree')
#(m4) Tree bagging
treeBag <- train(x = training[, predictors_2], y = training[, labelName], method = 'treebag')
#(m5) Naive Bayes on the Categorical predictors
nb <- train(x = training[, predictors_1], y = training[, labelName], method = 'nb') 
######
# Use trained models to predict on blender and test data
blender$c5oPr <- predict(c5o, blender)
blender$rfoPr <- predict(rfo, blender)
blender$cTreePr <- predict(cTree, blender)
blender$treeBagPr <- predict(treeBag, blender)
blender$nbPr <- predict(nb, blender)

test$c5oPr <- predict(c5o, test)
test$rfoPr <- predict(rfo, test)
test$cTreePr <- predict(cTree, test)
test$treeBagPr <- predict(treeBag, test)
test$nbPr <- predict(nb, test)

new_predictors <- names(blender)[names(blender) != labelName]

fit_ens <- train(x = blender[, new_predictors], y = blender[, labelName], method = "gbm") # Ensemble with GradientBoostingMachine
pp <- predict(fit_ens, test)
pp <- ifelse(pp == "yes", 1, 0)


######-----------------Write to disk for submission--------------------------------------------------
TEST <- read.csv("data/test.csv", na.strings = c("NA", "", " ", NULL))
submit <- data.frame(PassengerId = TEST$PassengerId, Survived = pp)
write.csv(x = submit, file = "submit25112015_2.csv", quote = F, row.names = F)






