
# Load packages -----------------------------------------------------------

library(mlr)
library(dplyr)
library(ggplot2)



# Load data ---------------------------------------------------------------

hr=read.csv("HR_comma_sep.csv",stringsAsFactors = FALSE)
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

  