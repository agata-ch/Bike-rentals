# Bike-rentals

This code analyzes the historic data on bike rentals and attempts to create a predicitve model.
Data are provided in the form of 'training.csv' and 'test.csv'. The aim was to predict the total counts of bike rentals. In what follows, I present a number of methods, all of them using the tree approach. The results for RMSLE are collected in a table 'compare.results', showing the effect of different methods 'm'.

This model uses trees to predict the total number of counts; a comparison with a linear regression model would be an interesting addition to the exercise.


## Part A
This part performs basic operations related to the dataframe bikes (<-training.csv, "full data"; the test.csv is used at the end - Part F) and modifies its structure/content to extract further information:
- $bikesHour, $bikesDates, $bikesMonth, $bikesYear.

It also performs some visualization of bikes$Counts (raw data, counts by day, by year).
  
  
## Part B 
This part investigates what type of probability distribution stands behind the bikes$Counts statistics, it involves
- histograms, including Counts, casual, registered; histograms for each year
- analysis of bikes$Counts: Cullen and Frey graphs
- searching for predictors "responsible" for a bimodal behaviour: breaking the data by hour (>6.5)
- attempt to fit specific distributions
- $casual & $registered columns are removed, as they will not be used for prediction.

Further analysis of this type could enable one to discuss, possibly, the presence of outliers etc. 
  
## Part C
In this part I work with the "full data", not broken by hour (6.5).
#### C.1
This part builds a basic tree.
First, the 'bikes' data is divided into training (75%) and test (25%) part, following the structure present in the problem (sampling each month ~ 75%). The obtained training and test sets are used in the later steps.
- a basic tree is built
- a CV is performed, but no pruning is necessary: best tree is the most complex one in this case. 
- a prediction is made and compared with the test data (-> compare.results)

#### C.2
Here I use bagging to improve the performance of the model. The corresponding prediction is made and RMSLE calculated. 

#### C.3 
I use random forest method to find an improvement. I look for an optimal number or predictors (mtry) used within this method, based on the RMSLE^2 between predictions and test data. The optimal value of 'mtry' is used to construct a random forest, and make prediction for Counts. 

#### C.4
The comparison of predictors' importance is shown. 

#### C.5 
Here I use the method of boosting to check if it can improve the performance of the tree. (I work with a copied data, which has $bikesDates changed into factors, due to the method's requirements).
Within this method, I look for an optimal value of the shrinkage parameter (out of eight values). Then I use its optimal value for boosting. Corresponding Counts predictions are made and compared to the test data. A summary of boosting is made. 

## Part D
In this part, I repeat the above steps (though not all of them) for the two dataframes:
- bikes.day (includes data for $bikesHour > 6.5)
- bikes.night (includes data for $bikesHour < 6.5)

Note: I subset the previously obtained bikes.train (indices) to have bikes.train.day & bikes.train.night, and then bikes.test.id.day, bikes.test.id.night. Therefore, within eg. bikes.day set, the ratio train-test may not be 75%-25%. However, I decided to work with this subsetting in order to be able to compare new results to the previously obtained ("full data") trees & results on RMSLE. 

I find basic & cv-trees, then random-forested tree (optimal 'mtry') and boosted (for bikes.night, with optimal 'shrinkage') one.

## Part E
In this part I compare the methods used. 
At each of the above steps I have updated the compare.results table, so that I can see the resulting RMSLE.

For the purpose of a valid comparison, I combined the bikes.day & bikes.night predictions to compare them with "full data" predicions.
Moreover, I did also the opposite: I checked how the "full data grown" random forest works on the subsets day & night. 

The best method seems to be the random forest method, and "full data grown" rf gives comparable results to the "broken data grown" rf, the latter performing slightly better, giving RMSLE ~ 0.3403. [rf = random forest]

## Part F
Here, I apply the rf (the "fully grown" and "broken data" one) to the real test data, given in the dataframe endMonth(<-test.csv). Therefore, I obtain two vectors with the information on the predicted Counts:
- endMonth.daynight$Counts
- endMonth$Counts
- histograms are suggested. 
Again, the histograms show that the bimodal character of the distribution is reproduced. 

#### F.1
In this part, I suggest a special treatment for some of the entries/days in the (real) test data. This entries correspond to days like car-free day (22nd Sept) and, possibly, Earth day (22nd April). Unfortunately, due to the structure of the given data (ends of month are tested, 22nd is always outside of our training set), I was unable to train my model against these possible outliers. 
[One could, however, look for similar days that occur within first 19 days of a month].

## Part G 
Here, I briefly suggest other possible predictors that could be constructed, without referring to external data: 
- measures of daylight/duration of sunlight
- measures of morning weather - that could potentially influence the decision-making of citizens
- introducing a categorical variable for "special days", like car-free day. 







