# ML Stock Prediction

Dataset used has been taken from https://www.kaggle.com/dgawlik/nyse

## Introduction 

<p align = "justify"> Stock Market prediction has been one of the holy grails of machine learning with extensive work being done by researchers from fields as diverse as mathematics, finance, biology, chemistry to history and literature. Almost every financial institute has been working on developing their own market prediction tool. The amount of work being done to achieve even a single digit increase in accuracy is worth it, as it will directly lead to a profit in millions of dollars.
</p>

<p align = "justify">
  <a href="https://github.com/lalithavadlamani/ML-Stock-Prediction/blob/main/SVM%20and%20Random%20Forest.ipynb"> Here </a> Machine learning algorithms - Support Vector Machine(SVM) and Random Forest and <a href="https://github.com/lalithavadlamani/ML-Stock-Prediction/blob/main/Linear%20Regression.ipynb"> Linear Regression </a> have been experimented to build a model which can predict the closing price of stocks. 
</p>
<p align = "justify">
The aim is to predict the closing price of a stock given the historical data of the stock in NYSE which has The price at the beginning of the day: Open Price, the lowest rate that stock reached that day: Low Price and the highest price that stock reached that day: High Price as the predictors and the price at the end of the day: Close Price is the output which needs to be predicted. The initial EDA of the stock data revealed a strong correlation among the variables.
 </p>
 
## Pre-Processing 
<p align = "justify"> Some initial data analysis and exploratory data analysis has been done to identify the pre-processing steps required. And after careful consideration data has been divided to train data and test data and only min-max normalization has been done. Only required columns have been then used for further steps. 
</p> 

## Methods experimented 
<p align = "justify"> 
1. <strong>Linear Regression</strong> - The parameters available in this stock exchange data allow to use regression analysis; this means that the analysis, evaluation of the bias and precision of the predicted values is possible. Regression basically allows the prediction of a required value, through the use of one or more independent predictors. The nature of regression algorithm that suits a task is dependent on the number of these independent variables and how they relate to the dependent variables. All these facts eventually made regression a popular technique for forecasting and stock market prediction. Linear Regression is regression technique that has precisely one independent variable and a simple, linear relationship defines the independent variable with respect to the dependent variable. </p>
<p align = "justify">  2. <strong> Random Forest</strong> - Random Forest consists of many decision trees which use the concept of bagging. Decision trees are a form of supervised machine learning and are generally used for classification as well as regression problems. It functions by learning basic decision rules that it can apply in the prediction of target values from given data values. A Bagging model (as a regressor model) is an ensemble estimator that fits each basic regressor on random subsets of the dataset and next accumulate their single predictions, either by voting or by averaging, to make the final prediction (Mojtaba Nabipour, 2020). It is basically doing random sampling with some replacement, ideally some small subset of data from the dataset. This leads to reduction in the variance (for algorithms suffering from high variance), usually in decision trees. Bagging runs each model separately and then combines the result, to give a result like random forest. All the trees in Random Forest are run in parallel, which means that they have no interaction during runtime. Its basic logic is that it constructs a multitude of decision trees during training and outputs a class that is the mean regression of the individual trees. While it may aggregate many decision trees, it is a meta-estimator and allows some modifications to increase randomness:

* It limits the number of features that can be split down to a small percentage of the total features. This ensures that the aggregate model does not rely on only the top few features.
* Every decision tree is picking its own random sample from the given data, adding one more layer to the randomness at play, in turn preventing overfitting.
</p>
<p align = "justify">  3. <strong> Support Vector Machine(SVM)</strong> - In simplest terms, the primary aim of SVM is to find a hyperplane in a N-dimensional space, where N is the number of features, which can classify the data values distinctly and accurately. Building blocks of SVM are Hyperplanes and Support vectors. Hyperplanes primary role is to help classify the given data coordinates. They act as decision boundary of sorts, with points falling on either side of the plane belonging to different class. The dimensionality of any hyperplane depends on the number of features: if there are 2 input features, the hyperplane is just a line; and if there are 3 input features, the hyperplane is a 2d-plane. Support vectors are basically the data points that are closest to the hyperplane and have major influence in hyperplane’s orientation and position. Support vectors can be used to maximize the margin; this gives strong reinforcement to the accuracy of future data points. And modifying these support vectors provide us with the ability to change the position of the hyperplane. Since the task here is to predict involving regression we use Support Vector Regression(SVR). It has basically the same principle as SVM, but it’s intended for regression. Other regression models are mainly concerned with minimizing the difference between the predicted value and the actual value, whereas SVR attempts to the best possible line within a threshold value (distance between boundary line and hyperspace). SVR tries to find a decision boundary which ensures that the data point closest to the hyperplane or the support vectors is within that boundary (Dubey, 2020).
</p>

## Experiments

<div align = "justify">
  
1. <strong>Linear Regression</strong> - For Linear regression there is no hyper parameter tuning involved. It's pretty straight forward in its implementation to achive optimal results. 
2. <strong>Support Vector Machine</strong> - With default parameters, Coeeficient of determination was -0.44, RSS was 3285.69 with Execution time:0.0070 seconds. While the execution time may seem very good, the Coefficient of Determination being negative, and a comparatively higher RSS means tuning parameters was required. We use the inbuilt function of sklearn, GridSearchCV, in order to conduct a comprehensive search over the parameter values specified: svm_model, grid_parameters, n_jobs, cv, refit, verbose. The resultant best parameters of SVM after tuning are: {'C': 1, 'gamma': 1, 'kernel': 'rbf'}.
Model performance was tested again with best parameters obtained after tuning. The optimal model after tuning was with parameters: {'C': 1, 'cache_size': 200, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.1, 'gamma': 1, 'kernel': 'rbf', 'max_iter': -1, 'shrinking': True, 'tol': 0.001, 'verbose': False}.
3. <strong>Random Forest</strong> - With default parameters, Coeeficient of determination was 0.84, RSS was 2328.62 with Execution time:0.5122 seconds.  Random search has been used for tuning the parameters. After tuning the best parameters came out to be {'n_estimators': 1100, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 10, 'bootstrap': False}. Model performance was tested again with the parameters obtained after tuning {'bootstrap': False, 'ccp_alpha': 0.0, 'criterion': 'mse', 'max_depth': 10, 'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 1100, 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}. But the results did not improve much even after all the experimentation. The coefficient of determination remains 0.84 and RSS has in fact increased 2328.62 to 2370.13. These results do not justify the increase in time from 0.5122 seconds to 6.74539 seconds and in turn make this an unsuccessful tuning. This might be because of tuning using Random search. But we used Random Search instead of grid search due to the time complexity . The default parameters seem to be the best parameters for a good performance in this case.
</div>


## Analysis and Observations 
<div align="justify">

* The stock data in NYSE dataset spans from 2010 to the end 2016. From the initial data analysis and EDA, it can be seen that there is not much variance in the data of individual stocks. The daily prices open, close, high and low are highly correlated linearly and are very close in value. Due to which the score of linear regression is around 99% and is the best performing model for data in the study. But it will definitely perform badly if the data is nonlinear in real time.
* SVM is widely used for classification but it can also be used for regression problems. Support Vector Regression has been tried since it gives flexibility with the error to fit the data. SVR tries to minimize the coefficients instead of considering the Least square error like in Linear Regression. For the experiment SVR has been tried using different types of Kernel and for various values of gamma - kernel coefficient for rbf, poly and sigmoid and C - regularization parameter. According to the experiment results, SVR for the stock data performed better for parameters - kernel: rbf, gamma:1, C=1 obtained from grid search rather than the default SVR parameters. A pattern has been observed in the predictions with respect to the true value of the stock so it can be inferred that at those dates the stock prices fluctuated from normal. But the prediction is not good since SVR works well for nonlinear data and the data of the study is linear.
* Random forest uses Decision trees and outputs the mean prediction of the individual decision trees for regression, so the prediction varies according to the tree parameters. Several combinations of hyperparameters have been tried. According to the experiment results, Residual sum of squares was proportional to n_estimators which is the number of trees and the performance varied accordingly. Other parameters did not affect the performance much. Since there were many combinations of parameters, random search has been used over grid search to tune and find the optimal parameters so that it's not computationally expensive time wise. Random Forest performed far better than SVR but not as good as Linear Regression. But similar patterns like in SVR were observed for the same dates where the true value is greater than the predicted price. From this it can be concluded that the stock price fluctuated from normal at those particular dates indicating some major event or change in the market.
</div>

## Conclusion 

![](https://github.com/lalithavadlamani/ML-Stock-Prediction/blob/main/Experiment%20Results.PNG)
<p align = "justify"> 
Based on the table, SVM achieves the best possible time of the 3 models tested, but we can’t declare it the best. Due to the linear nature of the dataset, it is the linear regression model that comes out on top. With the minimum Residual Sum of Squares (RSS) and the maximum Coefficient of Determination, Linear Regression is significantly better than the other 2 models.
Random Forest gave us a very curious result as its performance after hyperparameter tuning were worse than when with default parameters. This might be because of the Random search which doesn’t consider all the combinations of parameters, so the best performance parameters were the default parameters for the data.
</p>

## References 
<div align = "justify"> 
  
* Agarwal, V., 2015. Research on Data Preprocessing and Categorization Technique for Smartphone Review Analysis. International Journal of Computer Applications, 131(4).
* Bhandari, A., 2020. Feature Scaling for Machine Learning: Understanding the Difference Between Normalization vs. Standardization. [Online] Available at: https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/
* Bhaya, W. S., 2017. Review of Data Preprocessing Techniques in Data Mining. Journal of Engineering and Applied Sciences, 12(16), pp. 4102-4107.
* Chakure, A., 2019. Random Forest Regression. [Online] Available at: https://medium.com/swlh/random-forest-and-its-implementation-71824ced454f [Accessed 29 June 2019].
* Codecademy, 2019. Normalization. [Online] Available at: https://www.codecademy.com/articles/normalization
* Dubey, B., 2020. What is Support Vector Regression — SVR?. [Online] Available at: https://medium.com/@bhartendudubey/what-is-support-vector-regression-svr-760b501b6141 [Accessed 9 January 2020].
* Gandhi, R., 2018. Introduction to Machine Learning Algorithms: Linear Regression. [Online] Available at: https://towardsdatascience.com/introduction-to-machine-learning-algorithms-linear-regression-14c4e325882a [Accessed 18 May 2018].
* Gandhi, R., 2018. Support Vector Machine — Introduction to Machine Learning Algorithms. [Online] Available at: https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47 [Accessed 8 June 2018].
* Kim, K.-j., 2003. Financial time series forecasting using support vector machines. Elsevier , Neurocomputing(55).
* M. Nabipour, P. H. J. A. M. E. S. S. S., 2020. Deep Learning for Stock Market Prediction, s.l.: Entropy 2020.
* Mariette Awad, R. K., 2015. Efficient Learning Machines. Berkeley, CA: Apress.
* Mayankkumar B Patel, S. R. Y., 2014. Stock Price Prediction Using Artificial Neural Network. International Journal of Innovative Research in Science, Engineering and Technology, 3(6), p. 13755.
* Mojtaba Nabipour, P. N. H. J. A. M., 2020. Deep learning for Stock Market Prediction. [Online] Available at: https://arxiv.org/abs/2004.01497
* Molnar, C., 2016. Interpretable Machine Learning. [Online] Available at: https://christophm.github.io/interpretable-ml-book/limo.html#advantages [Accessed 16 11 2016].
* Osman Hegazy, O. S. S. M. A. S., 2013. A Machine Learning Model for Stock Market Prediction. International Journal of Computer Science and Telecommunications, 4(12).
* Scikit Learn, 2020. 3.2.4.3.2. sklearn.ensemble.RandomForestRegressor. [Online] Available at: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
* Shah, V. H., 2007. Machine Learning Techniques for Stock Prediction, New York: New York University.
* Singh, N., 2020. Advantages and Disadvantages of Linear Regression. [Online] Available at: https://iq.opengenus.org/advantages-and-disadvantages-of-linear-regression/
  
</div>
