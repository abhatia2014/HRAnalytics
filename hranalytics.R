
# Load packages -----------------------------------------------------------

library(mlr)
library(dplyr)
library(ggplot2)



# Load data ---------------------------------------------------------------

hr=read.csv("HR_comma_sep.csv",stringsAsFactors = TRUE)
str(hr)
hr$sales=factor(hr$sales)
hr$salary=factor(hr$salary)
str(hr)

#1. find the distribution of salary

table(hr$salary)

#2. find the distribution of sales column

table(hr$sales)
plot(hr$sales)
ggplot(hr,aes(sales,fill=sales))+geom_bar()+coord_flip()+theme(legend.position = "none")

#examine interaction of salary and role (sales)

#how many high medium and low belong to each category of role
hr$salary=ordered(hr$salary,c("low","medium","high"))
table(hr$sales,hr$salary)
plot(table(hr$sales,hr$salary),color=hr$salary)

#looking at the same data by percentage

prop.table(table(hr$sales,hr$salary),margin = 1)

#distribution of promotion

table(hr$promotion_last_5years)

#let's look at the intersection of promotion and salary

table(hr$salary,hr$promotion_last_5years)

prop.table(table(hr$salary,hr$promotion_last_5years),margin = 1)

#now let's loo at the intersection of promotion and role

table(hr$sales,hr$promotion_last_5years)
#as a percentage
prop.table(table(hr$sales,hr$promotion_last_5years),margin = 1)*100
#10% management people got promoted , 5% marketing people got promoted

#find the time spend in the company

table(hr$time_spend_company)
#let's see the intersection with promotion

data.frame(prop.table(table(hr$time_spend_company,hr$promotion_last_5years),margin = 1)*100)

#do analysis using tidyverse and formattable

library(formattable)
library(tidyverse)

#interaction of sales and salary features

hr%>%
  count(sales,salary)%>%
  mutate(salary=ordered(salary,c("low","medium","high")))%>%
  spread(salary,n)%>%
  formattable(align="l")

#to view this as a percent

hr%>%
  count(sales,salary)%>%
  mutate(salary=ordered(salary,c("low","medium","high")),
         n=percent(n,1),
         n=n/sum(n))%>%
  spread(salary,n)%>%
  formattable(align="l")

#view promotion by salary

hr%>%
  count(promotion_last_5years,salary)%>%
  group_by(salary)%>%
  mutate(n=n/sum(n),n=percent(n))%>%
  spread(promotion_last_5years,n)%>%
  formattable(align="l")
  
#visualize time spent

hr %>% 
  count(time_spend_company) %>% 
  ggplot(aes(time_spend_company,n))+geom_line()

#time spent related to salary

hr %>% 
  count(time_spend_company,salary) %>% 
  ggplot(aes(time_spend_company,n))+geom_line()+facet_wrap(~salary)

table(hr$left)

hr$left=factor(hr$left,labels = c("remain","left"))

#now plot the variables against left
# first the satisfaction rate

hr %>% 
  ggplot(aes(left,satisfaction_level,fill=left))+geom_boxplot()

#with last evaluation
hr %>% 
  ggplot(aes(left,last_evaluation,fill=left))+geom_boxplot()

#with salary rate

hr %>% 
  ggplot(aes(left,satisfaction_level,fill=left))+geom_boxplot()+facet_wrap(~factor(time_spend_company))+theme(legend.position = "none")
  
#data visualization using correlation plots

library(corrplot)
hr$left=as.numeric(hr$left)
corrplot(cor(hr[,1:8]),method = "circle")


# Prediction using machine learning MLR ---------------------------------------

library(mlr)
summarizeColumns(hr)
#first create train and test set
#train will be 60% of data
str(hr)
hr$left=factor(hr$left,labels = c("remain","left"))
table(hr$left)


train_hr=sample(nrow(hr),0.6*nrow(hr),replace = FALSE)
test_hr=setdiff(1:nrow(hr),train_hr)

train_hr_data=hr[train_hr,]
test_hr_data=hr[test_hr,]

#first create a classification task for the training and testing data

train_task=makeClassifTask(data = train_hr_data,target = "left",positive = 'left')
train_task
  
#similarly for testdata
test_task=makeClassifTask(data = test_hr_data,target = "left",positive = 'left')
test_task

#list all learners that can perform classification modeling on the train task

listLearners(train_task)[c("class","package")]

#we'll do the benchmarking using the following models
#1. GBM, 2. Logistic regression, 3. RPART 4. Random Forest, 5. c5.0 6. SVM

bench.learners=list(makeLearner("classif.rpart",predict.type = "prob"),makeLearner("classif.logreg",predict.type = "prob"),makeLearner("classif.gbm",predict.type = "prob"),
                    makeLearner("classif.randomForest",predict.type = "prob"),makeLearner("classif.ksvm",predict.type = "prob"),makeLearner("classif.C50",predict.type = "prob"))

#create sampling strategy- CV using 5 iterations
bench.strategy=makeResampleDesc("CV",iters=5)
#finalize performance measures
bench.measures=list(acc,auc,tpr,ppv)

#perform the benchmark experiment to find the best model

bench.expt=benchmark(learners = bench.learners,resamplings = bench.strategy,measures = bench.measures,tasks = train_task)

bench.expt
names(bench.expt)
bench.expt$results
bench.expt$measures
bench.expt$learners

#visulizing the results


#Random Forest, C50, and RPart have given the best results

#visualize performance

getBMRPerformances(bench.expt)

#aggregated performance
getBMRAggrPerformances(bench.expt)

#for plotting, better to convert this to a database

df=getBMRPerformances(bench.expt,as.df = TRUE)
plotBMRBoxplots(bench.expt,measure = auc)+aes(color=learner.id)

ggplot(df,aes(acc,auc,fill=learner.id))+geom_point(size=2)+facet_wrap(~learner.id)


# Use Random Forest to make predictions -----------------------------------

#selected as the best model

#tune hyperparameters for the best model

#first get all the parameters for random forest

getParamSet("classif.randomForest")
?randomForest

#we'll tune ntree

rf.search=makeParamSet(
  makeIntegerParam("ntree",lower=300,upper = 1000))

#find search algorithm- 200 iterations


rf.algo=makeTuneControlRandom(maxit = 10)

#define resampling strategy- 10 fold crossvalidation

rf.resam=makeResampleDesc("CV",iters=10)

#define the learner

rf.learner=makeLearner("classif.randomForest",predict.type = "prob",fix.factors.prediction = TRUE)

#finally tune parameters for selection of best hyperparameters

rf.tune=tuneParams(learner = rf.learner,task = train_task,resampling = rf.resam,
                 measures = list(auc,acc),par.set = rf.search,control = rf.algo )

#in this case, it's taking a lot of time so we use the default parameters

rf.train=train(learner = rf.learner,task = train_task)
rf.train
names(rf.train)
rf.train$time
#took 14 secs to run the training model

# now predict using the test set

rf.test=predict(rf.train,task = test_task)

names(rf.test)
rf.test
calculateConfusionMatrix(rf.test)
calculateROCMeasures(rf.test)

#calculate the performance of the model

performance(rf.test,measures = list(mmce,acc,auc,tpr,fpr))

#plotting the results

plotdata=generateThreshVsPerfData(rf.test,measures = list(fpr,tpr,mmce,acc,auc))
plotdata

plotROCCurves(plotdata)
ggplot(plotdata$data,aes(fpr,tpr,color="orange"))+geom_line()+geom_abline(slope = 1,intercept = 0,linetype="dashed",color="blue")
library(FSelector)

#find the variable contributing the maximum to the model

fv=generateFilterValuesData(task = test_task)
#this is using the default randomForestSRC package
#alternatively , the method can be information gain

fv
fv$data
#the top features resulting in people leaving are
#1. satisfaction level, number of projects (how much engaged), average monthly hours,last evaluation

#Salary, sales (Role), promotion , work accidents have the least impact on leaving 




#let's visualize the feature importance using rpart- single tree

rpart.learner=makeLearner("classif.rpart",predict.type = "prob",fix.factors.prediction = TRUE)
rpart.train=train(learner = rpart.learner,task = train_task)

#let's predict using the trained model

rpart.predict=predict(object = rpart.train,task = test_task)

calculateConfusionMatrix(rpart.predict)
calculateROCMeasures(rpart.predict)

plotdata=generateThreshVsPerfData(rpart.predict,measures = list(mmce,acc,auc,tpr,fpr))
performance(rpart.predict,measures = list(mmce,acc,auc,tpr,fpr))
generateFeatureImportanceData(task = test_task,learner = rpart.learner)

#using the tree directly w/o using the mlr package
library(rpart)
rpart.tree=rpart(left~., data=train_hr_data)

res=predict(rpart.tree,newdata = test_hr_data)

library(rpart.plot)

#let's visualize the plot
rpart.plot(rpart.tree,type=2,fallen.leaves = FALSE,cex=0.75,extra=2)

#let's try and find model importance using randomforest directly
library(randomForest)
model.rf=randomForest(as.factor(left)~.,data=train_hr_data)
importance(model.rf)

predict.rf=predict(model.rf,newdata = test_hr_data)

library(caret)
confusionMatrix(predict.rf,test_hr_data$left)

#detached caret package

library(mlr)


# Regression analysis on Satisfaction level -------------------------------

head(train_hr_data)
names(train_hr_data)
#removing left column
train_sat_data=train_hr_data[,-7]
head(train_sat_data)

#similarly for test data set

test_sat_data=test_hr_data[,-7]

#creating regression tasks for test and training
train_task_sat=makeRegrTask(data = train_sat_data,target = "satisfaction_level")
test_task_sat=makeRegrTask(data = test_sat_data,target = "satisfaction_level")

train_task_sat

#perform a benchmark experiment to find the best model for regression
#first find all models for regression for the training tasks

reg.learners=listLearners(train_task_sat)[c("class","package")]

#make a list of learners for the benchmarking experiment

bench.regr.learners=list(makeLearner("regr.gbm"),makeLearner("regr.glm"),makeLearner("regr.glmnet"),
                         makeLearner("regr.rpart"),makeLearner("regr.lm"),makeLearner("regr.randomForest"),makeLearner("regr.svm"))

# decide on the resampling strategy

bench.regr.resamp=makeResampleDesc("CV",iters=5)

listMeasures(train_task_sat)
bench.regr.measure=list(mse,rmse,medse,mae,timetrain,sse)

#perform the benchmark experiment

bench.regr.expt=benchmark(learners = bench.regr.learners,tasks = train_task_sat,resamplings = bench.regr.resamp,
                          measures = bench.regr.measure)

bench.regr.expt
bench.regr.expt$measures
#the best models for analysis are random forest, rpart, and SVM

#let's visualize the results

plotregrdata=getBMRPerformances(bench.regr.expt,as.df = TRUE)

plotBMRBoxplots(bench.regr.expt)+aes(color=learner.id)+ggtitle("Regression Model Performance")

#predicting using random forest first
#1. make learner
final.regr.learn1=makeLearner("regr.randomForest")
#2. train model
rf.regr.train=train(learner = final.regr.learn1,task = train_task_sat)

#3. predict using model
rf.regr.predict=predict(rf.regr.train,task = test_task_sat)
performance(rf.regr.predict,measures = list(mse,sse,rmse))
rf.regr.predict
head(rf.regr.predict$data,50)
ggplot(rf.regr.predict$data,aes(truth,response))+geom_point()+geom_smooth(method = "lm")

#finding variable importance
gen.fv=generateFilterValuesData(train_task_sat)
gen.fv$data
#plot filter values
plotFilterValues(gen.fv)
plotFilterValuesGGVIS(gen.fv)

#retrain and find variable importance plot using regular random forest method
rf.regr=randomForest(satisfaction_level~.,data=train_sat_data)
rf.predict=predict(rf.regr,newdata = test_sat_data)
plot(test_sat_data$satisfaction_level,rf.predict)
varImpPlot(x = rf.regr)

#let's train and predict using rpart now

rpart.regr.train=train(learner = "regr.rpart",task = train_task_sat)

#predict
rpart.regr.predict=predict(object = rpart.regr.train,task = test_task_sat)
performance(rpart.regr.predict,measures = list(mse,sse,rmse) )
ggplot(rpart.regr.predict$data,aes(truth,response))+geom_point()+geom_smooth(method = "lm")

#visualizing rpart.plot feature importance

#first a retraining is needed
rpart.regr=rpart(satisfaction_level~.,data=train_sat_data)

rpart.plot(x = rpart.regr,type = 2,fallen.leaves = FALSE,cex=0.75)

#one final model building using SVM

svm.learner=makeLearner("regr.svm")
#train model
svm.regr.train=train(learner = svm.learner,task = train_task_sat)

#predict model
svm.regr.predict=predict(svm.regr.train,task = test_task_sat)
svm.regr.predict
performance(svm.regr.predict,measures = list(mse,sse,rmse))
#plotting 

ggplot(svm.regr.predict$data,aes(truth,response))+geom_point()+geom_smooth(method="loess",se = TRUE)
