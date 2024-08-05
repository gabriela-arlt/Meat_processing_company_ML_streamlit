# Project: Appication for ARIMA, APRIORI and kMEANS
## Premise of the project:
The focust of the procest is creating a Streamlit Application which would help a meat-processing company gain quick insights about sales forecast, relationship between their products and customer clusters.

## DATA:
provided by a meat-processing company based in Bulgaria. All names and photos in the project have been modified due to the sensitive real data. excel sheet for sales report 2022 excel sheet for slaes report 2023 excel sheet for cluent data and locations

## Part 1 (EDA):
Python libraries used: numpy, pandas, matplotlib, seaborn, scipy (stats) Steps 1: Experimental physe and exploration of the dataset (reference: EDA_1 and EDA_2)

+ checked for null and na values, filled or deleted if any. I had an issue with gaining the geographical locations for each of their customers due the company based in Bulgaria and the language of the dataset. Google MAps API could not recognize all the names so I manually corrected several in the beginning.
+ checked for duplicate-rows
+ Standardized column names
+ removed unncessesary names and did feature engineering
+ corrected formatting
+ merged all dataset for finding trends and differences
+ Experimented with different plots: Monthly Sales, Quarterly Sales, Yearly Sales, Weekend Sales, Sales by Product, Sales by Customer Type, Sales by City, Sales by Payment Type, Monthy Sales Tred by Year etc.
+ After doing that I quickly realized why the CEO needed an app for visulizing, the data really needs a specific tool for that and Excel could not provide insights for this type of data
+ saved a file for my other project which is related to the same dataset but application of machine learning models
+ tried to find out if there are any correlations just from curiousity
+ after exploting all plots I have gathered the necessary insights which would help me create the application so saved the dataset which I would need for that
## Step 2 (reference: Streamlit_ML.py)
+ exploting different optiont for Time Series, client clustering and prodcut relationship (files: ML_Apriori.ipynb, ML_ARIMA.ipynb, ML_clustering.ipynb, ML_Prophet.ipynb) 
+ creation of a CSS file for the design (this was extremely challenging because Streamlit is not very flexible with the design)
+ link to the app would be useless here because in order to see the implementation of the models, you would have to add the data (which is ownered by a third party and I am not allowed to share it) so I have added a link to the final presenation:
+ https://www.youtube.com/watch?v=UyEzFsNb0dY
