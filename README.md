# AUTO GM: An automated decision making application for automotive business professionals 
---
The MVP version of my data product AUTO GM, can be found here https://auto-gm.herokuapp.com/ 

## Table of Contents 

- Executive Summary 
- Application Overview
- Software Requirements
- Sources
---
## Executive Summary: 

AUTO GM has the potential to change the landscape in automotive maintenance  management by introducing automated systems of demand forecasting, KPI analysis and key stakeholder advisement 

Data was collected through an online repo of data for the auto repair chain that I analyzed. Reports can be scheduled to be sent automatically, which I will explore later when having a live up to date app running. 

The metric I used was the mean absolute error (mae) when predicting sales forecasts on a train/test split of the data. 

Of the four models I used, a SARIMA model produced the lowest mae. I would like to explore some RNN models in the future to see if I can improve performance. I also have a custom weighted model I would like to build out and test against these more proven models. 
What risks/limitations/assumptions affect these findings?
With seasonal data, major shifts in the economic climate expose the predictions to significant error. The most recent example of this is obviously the effect COVID has had on the economy, and you can see there is a significant drop in sales revenue for two months in 2020 where businesses in Fort Worth, TX  had to close or limit capacity. 

The MVP version of Auto GM was able to predict sales volume for three different time horizons: 7, 30 and 90 days. Further implementation would extend predictions with a multi variable model on several KPI’s.

The best performing SARIMA model was tuned to 12 seasonal periods, with high autocorrelation at 7 day lags.

Business metrics can be forecasted into the future, however the scale of these predictions will change the effectiveness. Larger scale, higher volume businesses will definitely have greater benefit than a local small business


--- 
## Application Overview & Future Plans 

Auto GM is built on a three pillar system of feedback for management. Predict, Identify and Inform

Predict: 
The goal of my predictions will be to outperform an industry standard mean benchmark. Modeling out KPI's in an interpretable way that saves the company money and resources is the primary goal of these predictions

Identify: 
With the numerous KPI's a business can look to the assess business health, what should they focus on first? The identification pillar of my application will be able to analyze trends in key areas of the business to understand business performance in real-time. 

Inform:
With the identification of these trends, the next step is to involve key personnel in this knowledge and provide for them a feedback mechanism on how to change the underperforming metrics and celebrate the positive trending ones.

---
## Software Requirements:
- Data analysis required the use of pandas, scikit-learn, numpy and matplotlib
- Modeling required the use of sktime's sarima model and naive forecaster as well as holt winters from statsmodels 
- I used a streamlit web app to put the model results together, deployed on the heroku platform

---
## Sources 

- Online repo for data collection: www.isicentral.com