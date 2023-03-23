# Mini Project
## Technocolabs-Softwares-Data-Analyst-Mini-Project-BigMart-Outlet-Sales
### Project Description
The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim of this data science project is to build a predictive model and find out the sales of each product at a particular store.

Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales.
 The data has missing values as some stores do not report all the data due to technical glitches. Hence, it will be required to treat them accordingly.

We will handle this problem in a structured way. We will be following the table of content given below.
- 1).Problem Statement
- 2).Hypothesis Generation
- 3).Loading Packages and Data
- 4).Data Structure and Content
- 5).Exploratory Data Analysis
- 6).Univariate Analysis
- 7).Bivariate Analysis
- 8).Missing Value Treatment
- 9).Feature Engineering
- 10).Encoding Categorical Variables
- 11).Label Encoding
- 12).One Hot Encoding
- 13).PreProcessing Data
- 14).Modeling
- 15).Linear Regression
- 16).Regularized Linear Regression
- 17).RandomForest
- 18).XGBoost
- 19).Summary

### Curriculum For This Project
- The Business Problem 
- Exploring The Dataset 
- Exploratory Data Analysis (eda) - Outliers
- Exploratory Data Analysis (eda) - Graphs
- Converting Categorical To Numerical
- Seperating Training And Test Data
- Running The Models
- Hyper Parameter Tuning XGB And GBR
- Standard Scaling 06m Robust Scaling
- Final Predictions On The Test Dataset
- Saving The Final Model

## Reference

### Problem Statements
- Hypothesis Generation
  <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top: 10px; margin-bottom:10px; border-radius: 7px;">
        <a href="https://www.analyticsvidhya.com/blog/2020/09/hypothesis-generation-data-science-projects#:~:text=Hypothesis%20generation%20is%20an%20educated,generated%20based%20on%20any%20evidence."><img alt="" src="https://www.analyticsvidhya.com/wp-content/uploads/2015/02/logo_square_v2.jpg" align="left"; style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://www.analyticsvidhya.com/blog/2020/09/hypothesis-generation-data-science-projects#:~:text=Hypothesis%20generation%20is%20an%20educated,generated%20based%20on%20any%20evidence." style="padding-left: 10px; ">Hypothesis Generation for Data Science Projects</a>
</div>

   <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top:10px; margin-bottom:10px; border-radius: 7px;">
    <a href="https://vitalflux.com/data-science-how-to-formulate-hypothesis-for-hypothesis-testing" ><img alt="" src="https://vitalflux.com/wp-content/uploads/2018/10/cropped-vitalflux-192x192.png" align="left" style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://vitalflux.com/data-science-how-to-formulate-hypothesis-for-hypothesis-testing" style="padding-left: 10px; ">Hypothesis Testing Explained with Examples - Data Analytics</a>
</div>    
    
- Loading Packages and Data

  <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top: 10px; margin-bottom:10px; border-radius: 7px;">
        <a href="https://machinelearningmastery.com/load-machine-learning-data-python"><img alt="" src="https://machinelearningmastery.com/wp-content/uploads/2016/09/cropped-icon-192x192.png" align="left"; style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://machinelearningmastery.com/load-machine-learning-data-python" style="padding-left: 10px; ">How To Load Machine Learning Data in Python - Machine Learning Mastery</a>
</div>

   <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top:10px; margin-bottom:10px; border-radius: 7px;">
    <a href="https://www.kdnuggets.com/2020/08/5-different-ways-load-data-python.html" ><img alt="" src="https://www.kdnuggets.com/wp-content/themes/kdn17/images/favicon.ico" align="left" style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://www.kdnuggets.com/2020/08/5-different-ways-load-data-python.html" style="padding-left: 10px; ">5 Different Ways to Load Data in Python - KDnuggets</a>
</div> 
    
- Data Structure and Content

  <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top: 10px; margin-bottom:10px; border-radius: 7px;">
        <a href="https://www.analyticsvidhya.com/blog/2015/04/comprehensive-guide-data-exploration-sas-using-python-numpy-scipy-matplotlib-pandas"><img alt="" src="https://www.analyticsvidhya.com/wp-content/uploads/2015/02/logo_square_v2.jpg" align="left"; style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://www.analyticsvidhya.com/blog/2015/04/comprehensive-guide-data-exploration-sas-using-python-numpy-scipy-matplotlib-pandas" style="padding-left: 10px; ">Data Exploration in Python Using Pandas, Numpy, Matplotlib</a>
</div>

   <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top:10px; margin-bottom:10px; border-radius: 7px;">
    <a href="https://www.districtdatalabs.com/data-exploration-with-python-1" ><img alt="" src="https://images.squarespace-cdn.com/content/v1/55fdfa38e4b07a55be8680a4/1584138675353-BE5G7VJRAC70PPRG4MFR/favicon.ico?format=100w" align="left" style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://www.districtdatalabs.com/data-exploration-with-python-1" style="padding-left: 10px; ">Data Exploration with Python, Part 1 | District Data Labs</a>
</div> 

### Data Analysis And Preprocessing
- Exploratory Data Analysis

  <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top: 10px; margin-bottom:10px; border-radius: 7px;">
        <a href="https://www.analyticsvidhya.com/blog/2020/08/exploratory-data-analysiseda-from-scratch-in-python"><img alt="" src="https://www.analyticsvidhya.com/wp-content/uploads/2015/02/logo_square_v2.jpg" align="left"; style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://www.analyticsvidhya.com/blog/2020/08/exploratory-data-analysiseda-from-scratch-in-python" style="padding-left: 10px; ">Exploratory-Data-Analysis(EDA) from Scratch | With Python Implementation</a>
</div>

   <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top:10px; margin-bottom:10px; border-radius: 7px;">
    <a href="https://medium.com/swlh/exploratory-data-analysis-eda-from-scratch-in-python-8c12c2673aa7" ><img alt="" src="https://miro.medium.com/fit/c/152/152/1*sHhtYhaCe2Uc3IU0IgKwIQ.png" align="left" style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://medium.com/swlh/exploratory-data-analysis-eda-from-scratch-in-python-8c12c2673aa7" style="padding-left: 10px; ">Exploratory-Data-Analysis(EDA) from Scratch | With Python Implementation</a>
</div> 

   
- Univariate Analysis

  <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top: 10px; margin-bottom:10px; border-radius: 7px;">
        <a href="https://www.analyticsvidhya.com/blog/2020/07/univariate-analysis-visualization-with-illustrations-in-python"><img alt="" src="https://www.analyticsvidhya.com/wp-content/uploads/2015/02/logo_square_v2.jpg" align="left"; style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://www.analyticsvidhya.com/blog/2020/07/univariate-analysis-visualization-with-illustrations-in-python" style="padding-left: 10px; ">Univariate Data Visualization | Understand Matplotlib and Seaborn Indepth</a>
</div>

   <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top:10px; margin-bottom:10px; border-radius: 7px;">
    <a href="https://www.statology.org/univariate-analysis-in-python" ><img alt="" src="https://www.statology.org/wp-content/uploads/2019/08/cropped-StatologyFavicon-192x192.png" align="left" style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://www.statology.org/univariate-analysis-in-python" style="padding-left: 10px; ">How to Perform Univariate Analysis in Python (With Examples) - Statology</a>
</div> 

- Bivariate Analysis

  <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top: 10px; margin-bottom:10px; border-radius: 7px;">
        <a href="https://www.analyticsvidhya.com/blog/2022/03/bivariate-feature-analysis-in-python"><img alt="" src="https://www.analyticsvidhya.com/wp-content/uploads/2015/02/logo_square_v2.jpg" align="left"; style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://www.analyticsvidhya.com/blog/2022/03/bivariate-feature-analysis-in-python" style="padding-left: 10px; ">Bivariate Feature Analysis in Python - Analytics Vidhya</a>
</div>

   <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top:10px; margin-bottom:10px; border-radius: 7px;">
    <a href="https://www.statology.org/bivariate-analysis-in-python" ><img alt="" src="https://www.statology.org/wp-content/uploads/2019/08/cropped-StatologyFavicon-192x192.png" align="left" style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://www.statology.org/bivariate-analysis-in-python" style="padding-left: 10px; ">How to Perform Bivariate Analysis in Python (With Examples) - Statology</a>
</div> 

- Missing Value Treatment

  <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top: 10px; margin-bottom:10px; border-radius: 7px;">
        <a href="https://www.analyticsvidhya.com/blog/2021/05/dealing-with-missing-values-in-python-a-complete-guide"><img alt="" src="https://www.analyticsvidhya.com/wp-content/uploads/2015/02/logo_square_v2.jpg" align="left"; style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://www.analyticsvidhya.com/blog/2021/05/dealing-with-missing-values-in-python-a-complete-guide" style="padding-left: 10px; ">Dealing With Missing Values in Python - Analytics Vidhya</a>
</div>

   <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top:10px; margin-bottom:10px; border-radius: 7px;">
    <a href="https://machinelearningmastery.com/handle-missing-data-python" ><img alt="" src="https://machinelearningmastery.com/wp-content/uploads/2016/09/cropped-icon-192x192.png" align="left" style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://machinelearningmastery.com/handle-missing-data-python" style="padding-left: 10px; ">How to Handle Missing Data with Python - Machine Learning Mastery</a>
</div> 

- Feature Engineering

  <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top: 10px; margin-bottom:10px; border-radius: 7px;">
        <a href="https://www.analyticsvidhya.com/blog/2021/09/complete-guide-to-feature-engineering-zero-to-hero"><img alt="" src="https://www.analyticsvidhya.com/wp-content/uploads/2015/02/logo_square_v2.jpg" align="left"; style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://www.analyticsvidhya.com/blog/2021/09/complete-guide-to-feature-engineering-zero-to-hero" style="padding-left: 10px; ">Complete Guide to Feature Engineering: Zero to Hero</a>
</div>

   <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top:10px; margin-bottom:10px; border-radius: 7px;">
    <a href="https://www.analyticsvidhya.com/blog/2020/12/feature-engineering-using-pandas-for-beginners" ><img alt="" src="https://www.analyticsvidhya.com/wp-content/uploads/2015/02/logo_square_v2.jpg" align="left" style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://www.analyticsvidhya.com/blog/2020/12/feature-engineering-using-pandas-for-beginners" style="padding-left: 10px; ">Feature Engineering Using Pandas Library for Beginners</a>
</div> 

   <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top:10px; margin-bottom:10px; border-radius: 7px;">
    <a href="https://towardsdatascience.com/feature-engineering-in-python-part-i-the-most-powerful-way-of-dealing-with-data-8e2447e7c69e" ><img alt="" src="https://miro.medium.com/fit/c/152/152/1*sHhtYhaCe2Uc3IU0IgKwIQ.png" align="left" style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://towardsdatascience.com/feature-engineering-in-python-part-i-the-most-powerful-way-of-dealing-with-data-8e2447e7c69e" style="padding-left: 10px; ">Feature Engineering in Python Part I: The Most Powerful way of Dealing with Data.</a>
</div> 

- Encoding Categorical Variables

  <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top: 10px; margin-bottom:10px; border-radius: 7px;">
        <a href="https://www.analyticsvidhya.com/blog/2020/08/types-of-categorical-data-encoding"><img alt="" src="https://www.analyticsvidhya.com/wp-content/uploads/2015/02/logo_square_v2.jpg" align="left"; style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://www.analyticsvidhya.com/blog/2020/08/types-of-categorical-data-encoding" style="padding-left: 10px; ">What is Categorical Data | Categorical Data Encoding Methods</a>
</div>

   <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top:10px; margin-bottom:10px; border-radius: 7px;">
    <a href="https://pbpython.com/categorical-encoding.html" ><img alt="" src="https://pbpython.com/android-chrome-192x192.png" align="left" style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://pbpython.com/categorical-encoding.html" style="padding-left: 10px; ">Guide to Encoding Categorical Values in Python</a>
</div> 

- Label Encoding

  <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top: 10px; margin-bottom:10px; border-radius: 7px;">
        <a href="https://www.geeksforgeeks.org/ml-label-encoding-of-datasets-in-python"><img alt="" src="https://www.geeksforgeeks.org/wp-content/uploads/gfg_200X200.png" align="left"; style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://www.geeksforgeeks.org/ml-label-encoding-of-datasets-in-python" style="padding-left: 10px; ">ML | Label Encoding of datasets in Python - GeeksforGeeks</a>
</div>

   <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top:10px; margin-bottom:10px; border-radius: 7px;">
    <a href="https://www.mygreatlearning.com/blog/label-encoding-in-python" ><img alt="" src="https://d1m75rqqgidzqn.cloudfront.net/wp-data/2021/03/26162609/GL-Icon_16x16.png" align="left" style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://www.mygreatlearning.com/blog/label-encoding-in-python" style="padding-left: 10px; ">Label Encoding in Python Explained</a>
</div> 

- One Hot Encoding

  <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top: 10px; margin-bottom:10px; border-radius: 7px;">
        <a href="https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python"><img alt="" src="https://machinelearningmastery.com/wp-content/uploads/2016/09/cropped-icon-192x192.png" align="left"; style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python" style="padding-left: 10px; ">How to One Hot Encode Sequence Data in Python - Machine Learning Mastery</a>
</div>

   <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top:10px; margin-bottom:10px; border-radius: 7px;">
    <a href="https://stackabuse.com/one-hot-encoding-in-python-with-pandas-and-scikit-learn" ><img alt="" src="https://stackabuse.com/assets/images/favicon.ico" align="left" style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://stackabuse.com/one-hot-encoding-in-python-with-pandas-and-scikit-learn" style="padding-left: 10px; ">One-Hot Encoding in Python with Pandas and Scikit-Learn</a>
</div> 

- PreProcessing Data

  <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top: 10px; margin-bottom:10px; border-radius: 7px;">
        <a href="https://www.section.io/engineering-education/data-preprocessing-python"><img alt="" src="https://www.analyticsvidhya.com/wp-content/uploads/2015/02/logo_square_v2.jpg" align="left"; style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://www.section.io/engineering-education/data-preprocessing-python" style="padding-left: 10px; ">Getting Started with Data Preprocessing in Python</a>
</div>

   <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top:10px; margin-bottom:10px; border-radius: 7px;">
    <a href="https://www.geeksforgeeks.org/data-preprocessing-machine-learning-python" ><img alt="" src="https://www.geeksforgeeks.org/wp-content/uploads/gfg_200X200.png" align="left" style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://www.geeksforgeeks.org/data-preprocessing-machine-learning-python" style="padding-left: 10px; ">ML | Data Preprocessing in Python - GeeksforGeeks</a>
</div> 

### Modeling
- Linear Regression
  <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top: 10px; margin-bottom:10px; border-radius: 7px;">
        <a href="https://www.analyticsvidhya.com/blog/2022/02/linear-regression-with-python-implementation"><img alt="" src="https://www.analyticsvidhya.com/wp-content/uploads/2015/02/logo_square_v2.jpg" align="left"; style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://www.analyticsvidhya.com/blog/2022/02/linear-regression-with-python-implementation" style="padding-left: 10px; ">Linear Regression with Python Implementation - Analytics Vidhya</a>
</div>

   <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top:10px; margin-bottom:10px; border-radius: 7px;">
    <a href="https://realpython.com/linear-regression-in-python" ><img alt="" src="https://cdn.realpython.com/static/favicon.68cbf4197b0c.png" align="left" style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://realpython.com/linear-regression-in-python" style="padding-left: 10px; ">Linear Regression in Python – Real Python</a>
</div> 

 
- Regularized Linear Regression

  <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top: 10px; margin-bottom:10px; border-radius: 7px;">
        <a href="https://towardsdatascience.com/regularized-linear-regression-models-57bbdce90a8c"><img alt="" src="https://miro.medium.com/fit/c/152/152/1*sHhtYhaCe2Uc3IU0IgKwIQ.png" align="left"; style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://towardsdatascience.com/regularized-linear-regression-models-57bbdce90a8c" style="padding-left: 10px; ">Regularized Linear Regression Models</a>
</div>

   <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top:10px; margin-bottom:10px; border-radius: 7px;">
    <a href="https://towardsdatascience.com/ml-from-scratch-linear-polynomial-and-regularized-regression-models-725672336076" ><img alt="" src="https://miro.medium.com/fit/c/152/152/1*sHhtYhaCe2Uc3IU0IgKwIQ.png" align="left" style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://towardsdatascience.com/ml-from-scratch-linear-polynomial-and-regularized-regression-models-725672336076" style="padding-left: 10px; ">ML From Scratch: Linear, Polynomial, and Regularized Regression Models</a>
</div> 

- RandomForest

  <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top: 10px; margin-bottom:10px; border-radius: 7px;">
        <a href="https://machinelearningmastery.com/random-forest-ensemble-in-python"><img alt="" src="https://machinelearningmastery.com/wp-content/uploads/2016/09/cropped-icon-192x192.png" align="left"; style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://machinelearningmastery.com/random-forest-ensemble-in-python" style="padding-left: 10px; ">How to Develop a Random Forest Ensemble in Python - Machine Learning Mastery</a>
</div>

   <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top:10px; margin-bottom:10px; border-radius: 7px;">
    <a href="https://towardsdatascience.com/a-practical-guide-to-implementing-a-random-forest-classifier-in-python-979988d8a263" ><img alt="" src="https://miro.medium.com/fit/c/152/152/1*sHhtYhaCe2Uc3IU0IgKwIQ.png" align="left" style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://towardsdatascience.com/a-practical-guide-to-implementing-a-random-forest-classifier-in-python-979988d8a263" style="padding-left: 10px; ">A Practical Guide to Implementing a Random Forest Classifier in Python</a>
</div> 

- XGBoost

  <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top: 10px; margin-bottom:10px; border-radius: 7px;">
        <a href="https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn"><img alt="" src="https://machinelearningmastery.com/wp-content/uploads/2016/09/cropped-icon-192x192.png" align="left"; style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn" style="padding-left: 10px; ">How to Develop Your First XGBoost Model in Python - Machine Learning Mastery</a>
</div>

   <div style="color:black; font-weight:300; font-size:15px; padding: 20px; border: 1px solid #e7ebe8; margin-top:10px; margin-bottom:10px; border-radius: 7px;">
    <a href="https://towardsdatascience.com/beginners-guide-to-xgboost-for-classification-problems-50f75aac5390" ><img alt="" src="https://miro.medium.com/fit/c/152/152/1*sHhtYhaCe2Uc3IU0IgKwIQ.png" align="left" style="width: 32px; height: 32px; border-radius: 4px;"></a> <a href="https://towardsdatascience.com/beginners-guide-to-xgboost-for-classification-problems-50f75aac5390" style="padding-left: 10px; ">Beginner’s Guide to XGBoost for Classification Problems</a>
</div> 
