

#This code analyzes the historic data on bike rentals and attempts to create a predicitve model. 
#Data are provided in the form of 'training.csv' and 'test.csv'. The aim was to predict the total counts of bike rentals. 
#In what follows, I present a number of methods, all of them using the tree approach. 
#The results for RMSLE are collected in a table 'compare.results', showing the effect of different methods 'm'.
#This model uses trees to predict the total number of counts; a comparison with a linear regression 
#model would be an interesting addition to the exercise.
#
#Requires: train.csv, test.csv in the local folder

library(MASS)
library(tree)
library(randomForest)
library(fitdistrplus)
library(logspline)
library(gbm)

######################### A. DATA & PRELIMINARY ANALYSIS ######################################

#Load the data into a dataframe 'bikes'
bikes <- read.csv("train.csv")
#check: properties of the data set
dim(bikes)
if(sum(is.na(bikes)) > 0){bikes <- omit.na(bikes)}
colnames(bikes)
head(bikes)
summary(bikes)

#looking at the count of rentals
hist(bikes$count,breaks=100)

#Changing the first column into data type 
mydates <- as.character(bikes[, 1])
mydates <- strptime(mydates, "%Y-%m-%d %H:%M:%S")

#Adding columns to 'bikes' with specific date-time information: 
bikes$bikesHour <- as.numeric(format(mydates, "%H"))
bikes$bikesDates <- as.Date(mydates)
bikes$bikesMonth <- as.numeric(format(mydates, "%m"))
bikes$bikesYear <- as.numeric(format(mydates, "%Y"))

#Remove first column, 
bikes <- bikes[,-1]
names(bikes)[11]<-"Counts"

#I could additionally divide the days into separate weekdays
#bikes$Day<-weekdays(bikes$Dates)  #remember to change the class from character to factor or numeric

#Keeping original values of counts in:
bikesCountsOriginal<-data.frame(bikes$casual,bikes$registered,bikes$Counts)

#Chaning the counts into log(...)
bikes$casual<-log(bikes$casual+1)
bikes$registered<-log(bikes$registered+1)
bikes$Counts<-log(bikes$Counts)

###### PLOTS
#Plot the behaviour of total counts
plot(c(1:(200*24)),bikes$Counts[1:(200*24)],typ="l")

#Plot the mean value of counts for each day
bikes.counts.byday <- by(bikes$Counts, bikes$bikesDates, mean)
plot(c(1:length(bikes.counts.byday)), bikes.counts.byday, type="l")

bikes.counts.byyear <- by(bikes$Counts, bikes$bikesYear, mean)
bikes.counts.byyearSD <- by(bikes$Counts, bikes$bikesYear, sd)
print(bikes.counts.byyear)

######################### B. TESTING THE DISTRIBUTION #######################################################

#What's the distribution of bikes$Counts?
hist(bikes$Counts, breaks=200)
hist(bikes$casual, breaks=200)
hist(bikes$registered, breaks=200)

#Look at specific years
year2011 <- subset(bikes,bikesYear < 2012, select="Counts")
hist(year2011$Counts, breaks=100)

year2012 <- subset(bikes,bikesYear > 2011, select="Counts")
hist(year2012$Counts, breaks=100)

#Compute descriptive parameters of an empirical distribution
descdist(bikes$Counts, discrete=F, boot=1000)
plot(fitdist(bikes$Counts, "norm"))

#Remove casual and registered values
bikes$casual<-NULL
bikes$registered<-NULL

#Try this - to get rid of the two maxima present in hist(bikes$Counts,breaks=100)
 
bikes.day <- subset(bikes, bikesHour > 6.5, select = colnames(bikes))
bikes.night <- subset(bikes, bikesHour < 6.5, select = colnames(bikes))

#bikes.day
descdist(bikes.day$Counts, discrete=F, boot=1000)
fit.weibull.day <- fitdist(bikes.day$Counts, "weibull")
fit.norm.day <- fitdist(bikes.day$Counts, "norm")
fit.lnorm.day <- fitdist(bikes.day$Counts,"lnorm")
plot(fit.norm.day)
plot(fit.weibull.day)

#bikes.night
descdist(bikes.night$Counts, discrete=F, boot=1000)
fit.unif.night <- fitdist(bikes.night$Counts, "unif")
fit.norm.night <- fitdist(bikes.night$Counts, "norm")
plot(fit.unif.night)
plot(fit.norm.night)



######################### C. TREES FOR FULL DATA "BIKES" ##################################################

#First, I create a data frame for storing a comparison of methods & results used 

compare.results <- data.frame(c(0),c(0),c(0))
colnames(compare.results)<-c("DATA TYPE","METHOD USED","RMSLE")

##Subset the data to form bikes.train & test: I choose to have around 75% of data for training, 25% for testing
##This means: testing: 227*12=2724; 2724/10886 ~ 0.25
## I sample 227 entries from each month

exclude.howmany <- round((0.25*nrow(bikes))/12)
set.seed(1)
meas.month <- c(1:12)
bikes.train <- numeric()
for(i in 1:12){
  meas.month[i] <- length(which(bikes$bikesMonth == i))
    sample.Temp <- sample(which(bikes$bikesMonth == i), length(which(bikes$bikesMonth == i))-exclude.howmany)
    bikes.train <- c(bikes.train, sample.Temp)
}

######################### C.1.Building a tree ##################################################

tree.bikes <- tree(Counts~., bikes, subset=bikes.train)
summary(tree.bikes)

plot(tree.bikes)
text(tree.bikes, pretty=0)

#Cross-v
cv.bikes <- cv.tree(tree.bikes, FUN=prune.tree)
plot(cv.bikes$size, cv.bikes$dev, type='b')
#which tree is minimal in error?
cv.bikes$size[which.min(cv.bikes$dev)]

#PRUNING if necessary 
#prune.bikes=prune.tree(tree.bikes,best=...)
#plot(prune.bikes)

#Make predicion for Counts
mybikes1 <- predict(tree.bikes, newdata=bikes[-bikes.train, ])
bikes.test <- bikes[-bikes.train, "Counts"]
plot(mybikes1, bikes.test)
abline(0, 1)
#RMSLE^2
mean((mybikes1 - bikes.test)^2)

#Update the compare.results 
compare.results[1,] <- c("full","m1: cv-tree", round(sqrt(mean((mybikes1 - bikes.test)^2)),digits=5))

######################### C.2. Improving the tree with bagging ##################################################

set.seed(1)
bag.bikes <- randomForest(Counts~., data=bikes, subset=bikes.train, mtry=12, importance=TRUE)

#Make predicion for Counts
mybikes2 <- predict(bag.bikes, newdata=bikes[-bikes.train, ])
plot(mybikes2,bikes.test)
abline(0,1)
#RMSLE^2
mean((mybikes2-bikes.test)^2)

#Update the compare.results 
compare.results <- rbind(compare.results,c("full","m2: bagged tree", round(sqrt(mean((mybikes2-bikes.test)^2)), digits=5)))

######################### C.3. Improving the tree - random forest - with optimal mtry #########################

#Check what value of mtry will lead to the smallest error
rfmeans <- c(2:12)
for(j in 2:12){
    set.seed(1)
    rf.bikes.Temp <- randomForest(Counts~., data=bikes, subset=bikes.train, mtry=j, importance=TRUE)
    mybikes.Temp <- predict(rf.bikes.Temp, newdata=bikes[-bikes.train,])
    rfmeans[j-1] <- mean((mybikes.Temp - bikes.test)^2)
}

plot(c(2:12), rfmeans, type="b")
#What is the best mtry value? (that is - the lowest rfmeans value?)
which.min(rfmeans)
#Best: 
rf.bikes = randomForest(Counts~., data=bikes, subset=bikes.train, mtry=(1 + which.min(rfmeans)), importance=TRUE)

#Make predicion for Counts
mybikes3=predict(rf.bikes, newdata=bikes[-bikes.train, ])
plot(mybikes3, bikes.test)
abline(0, 1)
#RMSLE^2
mean((mybikes3 - bikes.test)^2)

#Update the compare.results 
compare.results <- rbind(compare.results,c("full","m3: random forest", round(sqrt(mean((mybikes3 - bikes.test)^2)), digits=5)))


######################### C.4. IMPORTANCE OF PREDICTORS ##################################################

importance(rf.bikes)
varImpPlot(rf.bikes)


######################### C.5. IMPROVING TREES WITH BOSSTING ##################################################


set.seed(1)
bikes.copy <- bikes
#I need to change bikes.copy$bikesDates to factors before using method 'gbm'
bikes.copy$bikesDates <- as.factor(bikes.copy$bikesDates)

#Below, I am checking roughly what value for the shrinkage parameter will be optimal, I investigate 8 values 
boost.means <- c(1:8)
for(i in 1:8){
              x.temp = (0.04*i-0.04 + 0.001)  #0.001 is default shrinkage
    boost.bikes.temp = gbm(Counts~., data=bikes.copy[bikes.train, ], distribution="gaussian", 
                      n.trees=5000, interaction.depth=4, shrinkage = x.temp, 
                      verbose = F)   
       mybikes4.temp = predict(boost.bikes.temp, newdata=bikes.copy[-bikes.train, ], n.trees=5000)
      boost.means[i] <- mean((mybikes4.temp - bikes.test)^2)
    }

#Choose the best value for shrinkage parameter
best.shrinkage <- 0.04*(which.min(boost.means)-1) + 0.001
boost.bikes = gbm(Counts~., data=bikes.copy[bikes.train, ], distribution="gaussian", 
                       n.trees=5000, interaction.depth=4, shrinkage=best.shrinkage, 
                       verbose = F)   
summary(boost.bikes)

#Make predicion for Counts
mybikes4 <- predict(boost.bikes, newdata=bikes.copy[-bikes.train, ], n.trees=5000)
plot(mybikes4, bikes.test)
abline(0, 1)
#RMSLE^2
mean((mybikes4 - bikes.test)^2)


#Update compare.results
compare.results <- rbind(compare.results,c("full","m4: boosting", round(sqrt(mean((mybikes4 - bikes.test)^2)), digits=5)))



######################### D. TREES FOR DATA BROKEN BY HOUR: DAY (Hour > 6.5) and NIGHT (Hour < 6.5) ############################

set.seed(1)
#Subsets: divide bikes.train into day & night, and also prepare the set of bikes.test.id 
#(which points to bikes[-bikes.train,]) and then divide it into day & night.
#This procedure may not be optimal, as it doesn't guarantee 25% & 75% in the bikes. day and bikes.night, test/training ratio
# but allows me to compare results with the previously obtained 

bikes.train.day <- intersect(bikes.train, which(bikes$bikesHour > 6.5))
bikes.train.night <- intersect(bikes.train, which(bikes$bikesHour < 6.5))     
bikes.test.id <- setdiff(c(1:nrow(bikes)), bikes.train)
bikes.test.id.day <- intersect(bikes.test.id, which(bikes$bikesHour > 6.5))
bikes.test.id.night <- intersect(bikes.test.id, which(bikes$bikesHour < 6.5)) 

###################### D.1. TREES FOR BIKES.DAY ###############################################

###Building a tree
tree.bikes.day = tree(Counts~., bikes, subset=bikes.train.day)
summary(tree.bikes.day)
plot(tree.bikes.day)
text(tree.bikes.day, pretty=0)

#Cross-v
cv.bikes.day <- cv.tree(tree.bikes.day, FUN=prune.tree)
plot(cv.bikes.day$size, cv.bikes.day$dev, type='b')
#check which is the best size of the tree
cv.bikes.day$size[which.min(cv.bikes.day$dev)]

#PRUNING if necessary
#prune.bikes.day=prune.tree(tree.bikes.day,best=...)
#plot(prune.bikes.day)

#Make predicion for Counts
mybikes.day1 <- predict(tree.bikes.day, newdata=bikes[bikes.test.id.day, ])
bikes.test.day <- bikes[bikes.test.id.day, "Counts"]
plot(mybikes.day1, bikes.test.day)
abline(0, 1)
#RMSLE^2
mean((mybikes.day1-bikes.test.day)^2)

#Update compare.results
compare.results <- rbind(compare.results,c("day","m1: cv-tree", round(sqrt(mean((mybikes.day1-bikes.test.day)^2)), digits=5)))



######################### D.1.a. Improving the trees via random forests [day] #########################

#Check what values of mtry will be optimal
rfmeans.day <- c(2:12)
for(j in 2:12){
        rf.bikes.Temp <- randomForest(Counts~., data=bikes, subset=bikes.train.day, mtry=j, importance=TRUE)
         mybikes.Temp <- predict(rf.bikes.Temp, newdata=bikes[bikes.test.id.day, ])
    rfmeans.day[j-1] <- mean((mybikes.Temp-bikes.test.day)^2)
}

plot(c(2:12), rfmeans.day, type="b")
which.min(rfmeans.day)

#Best value of mtry chosen: 
rf.bikes.day <- randomForest(Counts~., data=bikes, subset=bikes.train.day, mtry=(1 + which.min(rfmeans.day)), 
                          importance=TRUE)

#Make predicion for Counts
mybikes.day3 <- predict(rf.bikes.day, newdata=bikes[bikes.test.id.day, ])
plot(mybikes.day3, bikes.test.day)
abline(0, 1)
#RMSLE^2
mean((mybikes.day3 - bikes.test.day)^2)

#Update compare.results
compare.results <- rbind(compare.results,c("day","m3: random forest", round(sqrt(mean((mybikes.day3 - bikes.test.day)^2)), digits=5)))

importance(rf.bikes.day)
varImpPlot(rf.bikes.day)


######################### D.2 TREES FOR BIKES.NIGHT ##################################################

###Building a tree
tree.bikes.night <- tree(Counts~., bikes, subset=bikes.train.night)
summary(tree.bikes.night)
plot(tree.bikes.night)
text(tree.bikes.night, pretty=0)

#Cross-v
cv.bikes.night <- cv.tree(tree.bikes.night, FUN=prune.tree)
plot(cv.bikes.night$size, cv.bikes.night$dev, type='b')
#check which is the best
cv.bikes.night$size[which.min(cv.bikes.night$dev)]

#PRUNING if necessary
#prune.bikes.night = prune.tree(tree.bikes.day,best=...)
#plot(prune.bikes.night)

#Make predicion for Counts
mybikes.night1 <- predict(tree.bikes.night, newdata=bikes[bikes.test.id.night, ])
bikes.test.night <- bikes[bikes.test.id.night, "Counts"]
plot(mybikes.night1, bikes.test.night)
abline(0, 1)
#RMSLE^2
mean((mybikes.night1 - bikes.test.night)^2)

#Update compare.results
compare.results <- rbind(compare.results,c("night","m1: cv-tree", round(sqrt(mean((mybikes.night1 - bikes.test.night)^2)), digits=5)))



######################### D.2.a. Improving trees via random forests [night] #########################

rfmeans.night <- c(2:12)
for(j in 2:12){
    set.seed(1)
         rf.bikes.Temp <- randomForest(Counts~., data=bikes, subset=bikes.train.night, mtry=j, importance=TRUE)
          mybikes.Temp <- predict(rf.bikes.Temp, newdata=bikes[bikes.test.id.night, ])
    rfmeans.night[j-1] <- mean((mybikes.Temp-bikes.test.night)^2)
}
plot(c(2:12), rfmeans.night, type="b")

#which value of mtry is the best one?
which.min(rfmeans.night) 
#Best value applied: 
rf.bikes.night <- randomForest(Counts~., data=bikes, subset=bikes.train.night, mtry=(1 + which.min(rfmeans.night)), 
                            importance=TRUE)

#Make predicion for Counts
mybikes.night3 <- predict(rf.bikes.night, newdata=bikes[bikes.test.id.night, ])
plot(mybikes.night3, bikes.test.night)
abline(0, 1)
#RMSLE^2
mean((mybikes.night3 - bikes.test.night)^2)

#Update compare.results
compare.results <- rbind(compare.results,c("night","m3: random forest", round(sqrt(mean((mybikes.night3 - bikes.test.night)^2)), digits=5)))

importance(rf.bikes.night)
varImpPlot(rf.bikes.night)


######################### D.2.b. Improving trees via boosting [night] #########################

set.seed(1)
#Below I use the bikes.copy, as it has $bikesDate given as a factor
boost.means.night <- c(1:8)
for(i in 1:8){
                 x.temp <- (0.04*i-0.04 + 0.001)
       boost.bikes.temp <- gbm(Counts~., data=bikes.copy[bikes.train.night, ], distribution="gaussian", 
                           n.trees=5000, interaction.depth=4, shrinkage = x.temp, 
                           verbose = F)   
    mybikes.night4.temp <- predict(boost.bikes.temp, newdata=bikes.copy[bikes.test.id.night, ], n.trees=5000)
   boost.means.night[i] <- mean((mybikes.night4.temp - bikes.test.night)^2)
}

#Choose the best shrinkage parameter and use it 
best.shrinkage.night <- 0.04*(which.min(boost.means.night)-1) + 0.001
boost.bikes.night <- gbm(Counts~., data=bikes.copy[bikes.train.night, ],distribution="gaussian", 
                  n.trees=5000, interaction.depth=4, shrinkage = best.shrinkage.night, 
                  verbose = F)   
summary(boost.bikes.night)

#Make predicion for Counts
mybikes.night4 <- predict(boost.bikes.night, newdata=bikes.copy[bikes.test.id.night, ], n.trees=5000)
plot(mybikes.night4, bikes.test.night)
abline(0, 1)
#RMSLE^2
mean((mybikes.night4 - bikes.test.night)^2)


#Update compare.results
compare.results <- rbind(compare.results,c("night","m4: boosting", round(sqrt(mean((mybikes.night4 - bikes.test.night)^2)), digits=5)))


######################### E. COMPARING METHODS ##################################################

print(compare.results)
#Best method in full-data: random forest
#Best method for day: random forest
#Best method for night: random forest

#Add bikes.day & bikes.night predictions to compare to previous predictions
#RMSLE^2
day.night <- (sum((mybikes.night3 - bikes.test.night)^2) + 
                  sum((mybikes.day3 - bikes.test.day)^2))/(length(mybikes.day3) + length(mybikes.night3))

#Update compare.results
compare.results <- rbind(compare.results,c("day & night combined","m3: random forest", round(sqrt(day.night), digits=5)))


#Now, compare bikes.day predictions with subset of (full) bike predictions - for the best method, random trees.
mybikes3.day <- predict(rf.bikes, newdata=bikes[bikes.test.id.day, ])
#RMSLE^2
mean((mybikes3.day - bikes.test.day)^2)
#Update compare.results
compare.results <- rbind(compare.results,c("from full/day only","m3: random forest", round(sqrt(mean((mybikes3.day - bikes.test.day)^2)), digits=5)))

#And compare bikes.night predictions with subset of (full) bike predictions - for the best method, random trees.
mybikes3.night <- predict(rf.bikes, newdata=bikes[bikes.test.id.night, ])
#RMSLE^2
mean((mybikes3.night - bikes.test.night)^2)

#Update compare.results
compare.results <- rbind(compare.results,c("from full/night only","m3: random forest", round(sqrt(mean((mybikes3.night - bikes.test.night)^2)), digits=5)))

######################### F. REAL TEST DATA : END OF A MONTH ##################################################

endMonth <- read.csv("test.csv")
if(sum(is.na(endMonth)) > 0){endMonth <- omit.na(endMonth)}
colnames(endMonth)

#Changing the first column into data type 
end.dates <- as.character(endMonth[, 1])
end.dates <- strptime(end.dates, "%Y-%m-%d %H:%M:%S")
#Adding columns to 'endMonth': 
endMonth$Counts <- rep(0, nrow(endMonth))  #log values
endMonth$bikesHour <- as.numeric(format(end.dates, "%H"))
endMonth$bikesDates <- as.Date(end.dates)
endMonth$bikesMonth <- as.numeric(format(end.dates, "%m"))
endMonth$bikesYear <- as.numeric(format(end.dates, "%Y"))
#Removing first column, 
endMonth <- endMonth[,-1]

## Look at the comparison table: compare.results to see which method is best

## 1) Two random forests, for data split by hour 6.5, combined predictions
endMonth.day <- subset(endMonth, bikesHour > 6.5, select = colnames(endMonth))
endMonth.night <- subset(endMonth, bikesHour < 6.5, select = colnames(endMonth))

# UPDATE THE DATAFRAME WITH PREDICTIONS
endMonth.day$Counts = predict(rf.bikes.day, newdata=endMonth.day)
endMonth.night$Counts = predict(rf.bikes.night, newdata=endMonth.night)
endMonth.daynight <- rbind(endMonth.day, endMonth.night)
head(endMonth.daynight)

# endMonth.daynight$Counts
hist(endMonth.daynight$Counts,breaks=100)


## 2). Random forest build on the full data

# UPDATE THE DATAFRAME WITH PREDICTIONS
endMonth$Counts <- predict(rf.bikes, newdata=endMonth)
hist(endMonth$Counts,breaks=100)


######################### F.1. REMOVING POSSIBLE OUTLIERS ##################################################

#I remove possible outliers that may influence our predictions: 
#Car-free day: 22nd September
# one could also remove 22nd April (Earth day)

endMonth.clear <- endMonth[-which(endMonth$bikesDates == "2011-09-22" | endMonth$bikesDates == "2012-09-22" ), ]

#t.b.c.


######################### G. Possible predictors ##################################################

# 1. Daylight (day duration): sinusoidal values, that do not require any external data.

# 2. Morning_weather: avaraging the morning weather conditions (eg. 6-8am).

# 3. Special_day: days like car-free, Earth-day; special possibly treatment required for those. 





