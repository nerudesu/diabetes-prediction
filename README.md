# Machine Learning Project Report - Pradipta A. Suryadi

## Project Domain
According to ([WHO, 2023](https://www.who.int/news-room/fact-sheets/detail/diabetes))[1], diabetes is a medical condition that arises when the pancreas fails to generate sufficient insulin or when our body unable to use the insulin it produces effectively. Insulin is a hormone that responsibile for regulating blood glucose levels, it plays a crucial role in this process. Hyperglycemia, also known as elevated blood glucose or high blood sugar, is a frequent consequence of poorly managed diabetes. Over time, this condition can cause significant harm to numerous bodily systems, particularly the nerves and blood vessels.

In 2014, 8.5% of adults aged 18 years and above had diabetes. By 2019, diabetes caused 1.5 million deaths, with 48% occurring before the age of 70. Diabetes also contributed to 460,000 deaths related to kidney disease, and around 20% of cardiovascular-related deaths were linked to elevated blood glucose levels ([GBD, 2019](https://vizhub.healthdata.org/gbd-results/))[2].

Predicting diabetes based on patients' medical history and demographic information can be valuable for healthcare professionals. It helps identify individuals at risk of developing diabetes and enables the creation of personalized treatment plans for better patient care.

Researchers have employed a data-driven approach, utilizing machine learning techniques, to predict diabetes and cardiovascular disease. One study specifically utilized the National Health and Nutrition Examination Survey (NHANES) dataset to evaluate the classification performance of several machine learning models, including Logistic Regression, Support Vector Machines (SVM), Random Forest, and Gradient Boosting ([Dinh, A. et al, 2019](https://doi.org/10.1186/s12911-019-0918-5))[3]. 

## Business Understanding
Accurate diabetes prediction is crucial to ensure appropriate treatment and mitigate the risk of complications. In cases which little to no access to patient lab data, a way to predict using more general medical and demographic data is needed.

### Problem Statements
Based on the problems previously described, the problem statements for this project are as follows:
- What features have the most influence on determining patient diabetes?
- Which patients need special treatment to prevent diabetes at an early stage?

### Goals
The objectives from this project are as follows:
- Know features which highly correlated with the detection of a patient's diabetes.
- Develop a machine-learning model that can predict whether a patient has diabetes.

### Solution statements
The solution proposed to solve the problem is as follows:
- Exploring data and visualizing the correlation between existing features to find out the relationship between features
- Build machine learning models for classification using K-Nearest Neighbors (KNN), Random Forest, and AdaBoost with default parameters
- Comparing the evaluation of the resulting metrics and selecting the best Machine Learning model for model tuning to further improve the model

## Data Understanding
We will use Diabetes prediction dataset from Kaggle that can be accessed and downloaded from the following link: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset [4]

The dataset includes medical and demographic information of patients, including their diabetes status (positive or negative). It encompasses a range of features such as age, gender, body mass index (BMI), hypertension, heart disease, smoking history, HbA1c level, and blood glucose level.

### The variables in the Diabetes prediction dataset are as follows:
| **Feature**        | **Description**                                                                                                                                                                                                                    | **Range**                                                                                                                                                       |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| gender              | Gender refers to the biological sex of the individual, which can have an impact on their susceptibility to diabetes.                                                                                                               | There are three categories in it male, female and other.                                                                                                        |
| age                 | Age is an important factor as diabetes is more commonly diagnosed in older adults.                                                                                                                                                 | Age ranges from 0-80.                                                                                                                                           |
| hypertension        | Hypertension is a medical condition in which the blood pressure in the arteries is persistently elevated.                                                                                                                          | It has values a 0 or 1 where 0 indicates they don’t have hypertension and for 1 it means they have hypertension.                                                |
| heart_disease       | Heart disease is another medical condition that is associated with an increased risk of developing diabetes.                                                                                                                       | It has values a 0 or 1 where 0 indicates they don’t have heart disease and for 1 it means they have heart disease.                                              |
| smoking_history     | Smoking history is also considered a risk factor for diabetes and can exacerbate the complications associated with diabetes.                                                                                                       | There are 5 categories i.e. not current, former, No Info, current, never, and ever.                                                                             |
| bmi                 | BMI (Body Mass Index) is a measure of body fat based on weight and height. Higher BMI values are linked to a higher risk of diabetes.                                                                                              | The range of BMI in the dataset is from 10.16 to 71.55. BMI less than 18.5 is underweight, 18.5-24.9 is normal, 25-29.9 is overweight, and 30 or more is obese. |
| HbA1c_level         | HbA1c (Hemoglobin A1c) level is a measure of a person's average blood sugar level over the past 2-3 months. Higher levels indicate a greater risk of developing diabetes. Mostly more than 6.5% of HbA1c Level indicates diabetes. | The range of HbA1c_level in the dataset is from 3.5 to 9.                                                                                                       |
| blood_glucose_level | Blood glucose level refers to the amount of glucose in the bloodstream at a given time. High blood glucose levels are a key indicator of diabetes.                                                                                 | The range of blood_glucose_level in the dataset is from 80 to 300.                                                                                              |
| diabetes            | Diabetes is the target variable being predicted.                                                                                                                                                                                   | It has values of 1 indicating the presence of diabetes and 0 indicating the absence of diabetes.                                                                |

On this step we try to explore the shape of the data and it gives us following information:
1.   There are 100.000 rows of records or observation in the dataset.
2.   There are ten columns/features, as explained above.

We also try to explore the data to find correlation between features and visualize it using a heatmap. Correlation between variables indicates that as one variable changes, the other variable tends to change in a consistent direction. Recognizing this connection is valuable because it allows us to utilize the value of one variable to make predictions about the value of the other variable.

![Correlation Heatmap](https://drive.google.com/uc?export=download&id=1LfcHY6FePQYQH1BYj4E2aPrkaO1h0_R0 "Correlation Heatmap")

Upon analyzing the heatmap, we can say there's no strong correlation between features. The relationship between two variables is generally considered strong when their r value is larger than 0.7 ([Moore, D. S. et al, 2013](https://books.google.co.id/books/about/The_Basic_Practice_of_Statistics.html?id=aw61ygAACAAJ&redir_esc=y))[5].

## Data Preparation
At this stage we do following steps:
1.  **Data Cleaning**
    First let's check whether our dataset contains NA value. By ensuring the cleanliness and reliability of the data, we can reduce the bias or inaccurate analysis.
    There's no NA value on the dataset and there are two columns (gender and smoking_history) that have non-numeric values. Then we check data type for each column and make data type adjustment by casting age (discreate) type to int and blood_glucose (continous) to float to make appropriate data representation.
    We also check object type data and we found that majority of the smoking history's data were labeled with "No Info" and it's impossible if not difficult to make imputation for them, although they can represent a category of data, it may caused a bias in the decision. So we better remove it.
    |                 | Count | Percentage |
    |----------------:|------:|------------|
    | **No Info**     | 35816 |      35.8% |
    | **never**       | 35095 |      35.1% |
    | **former**      |  9352 |       9.4% |
    | **current**     |  9286 |       9.3% |
    | **not current** |  6447 |       6.4% |
    | **ever**        |  4004 |       4.0% |
    Gender with value other is in small percentage we could include them.
    |            | Count | Percentage |
    |-----------:|------:|------------|
    | **Female** | 58552 |      58.6% |
    | **Male**   | 41430 |      41.4% |
    | **Other**  |    18 |       0.0% |

2.  **Encoding Categorical Variables**
    Categorical data consists of variables with label values instead of numeric values. Some machine learning algorithms are unable to process label data directly. For it to work, these algorithms require that all input and output variables be in a numeric format, this means that categorical data must be converted to a numerical form.
    When dealing with categorical variables that lack an ordinal relationship, applying one-hot encoding is a suitable approach. This involves converting each categorical value into a new categorical column and assigning binary values of 1 or 0 to represent the presence or absence of each category. In the specific scenario mentioned, there is one remaining categorical variable (gender) that needs to be converted using the one-hot encoding (OHE) technique.

3.  **Splitting Data**
    Data splitting is a common practice in machine learning to prevent overfitting, which occurs when a model excessively tailors itself to the training data and struggles to generalize to new data. By dividing the data into separate sets, we can simulate how the model would perform with unseen data, providing a measure of its generalization capability.
    We split the data into train data and test data with ratio 80:20.

4.  **Feature Scaling**
    Feature scaling involves normalizing the range of features in a dataset. Since real-world datasets often consist of features with varying magnitudes, ranges, and units, it is necessary to perform feature scaling to ensure machine learning models can interpret these features on a consistent scale. We do feature scaling on numeric variables from our data.

## Modeling
On this stage we will build several model to train with their **default hyperparameters value** and compare **which one will give the best result**.

### K-Nearest Neighbors (KNN)
KNN is a supervised learning algorithm capable of addressing both regression and classification problems. It attempts to predict the appropriate class for test data by evaluating the distance between the test data and all the training points.

**Advantages:**
-   Easy to implement
-   No training period (data itself is a model which will be refference for future prediction)
-   New data can be added at any time (it won't affect the model)

**Disadvantages:**
-   Doesn't work well with large dataset (calculating distances between each data would be very costly)
-   Doesn't work well with high dimensionality (complicated distance calculating process)
-   Sensitive to noisy and missing data
-   Data in all dimension should be scaled (normalized and standardized) properly

### Random Forest
Random forests, also known as random decision forests, are a type of ensemble learning approach used for various tasks such as classification, regression, and more. They work by creating numerous decision trees during the training phase. In the case of classification tasks, the random forest's final output is determined by the class that is chosen by the majority of the trees.

**Advantages:**
-    Reduces overfitting in decision trees and enhances accuracy.
-    Effective with both categorical and continuous values.
-    Does not require data normalization, as it utilizes a rule-based approach.

**Disadvantages:**
-    Requires much computational power as it build numerous trees to combine their output
-    Due to the combination of numerous decision trees to determine the class, the training process of this approach demands a significant amount of time.
-    The ensemble of decision trees in this method hinders interpretability and prevents the determination of the individual significance of each variable.

### AdaBoost
Adaptive Boosting is an ensemble method used in machine learning. It combines multiple weak classifiers by adjusting the weights of the training samples based on their classification error.

**Advantages:**
- Adaboost is less susceptible to overfitting since the input parameters are not jointly optimized.
- The accuracy of weak classifiers can be enhanced through the utilization of Adaboost.

**Disadvantages:**
- Adaboost requires a high-quality dataset. It is necessary to avoid noisy data and outliers prior to implementing an Adaboost algorithm.


## Evaluation
We compare the accuracy and other metrics to determine the most effective model for predicting diabetes.

|                               | Accuracy | Precision |  Recall | F1-Score |
|------------------------------:|---------:|----------:|--------:|----------|
| **K-Nearest Neighbors (KNN)** |  0.96435 |  0.962654 | 0.96435 | 0.961790 |
| **Random Forest**             |  0.96830 |  0.967146 | 0.96830 | 0.966169 |
| **AdaBoost**                  |  0.97150 |  0.971428 | 0.97150 | 0.969222 |

First we discussed about the outcome of predicted result:

*   A person who is actually diabetes (positive) and classified as diabetes (positive). This is called TRUE POSITIVE (TP).
*   A person who is actually not diabetes (negative) and classified as not diabetes (negative). This is called TRUE NEGATIVE (TN).
*   A person who is actually not diabetes (negative) and classified as diabetes (positive). This is called FALSE POSITIVE (FP).
*   A person who is actually diabetes (positive) and classified as not diabetes (negative). This is called FALSE NEGATIVE (FN).

The Result Dataframe provides accuracy, precision, recall, and F1 values for each tested model.
1.  **Accuracy**
    It represents the ratio of accurately classified data instances to the total of data instances.
    
    $$Precision=\dfrac{TN+TP}{TN+FP+TP+FN}$$
    
    Accuracy may not be a good measure if the dataset is imbalanced (the negative and positive classes have unequal data instances).

2.  **Precision**
    It represents the ratio between the True Positive (TP) and the amount of data that is predicted to be positive.
    
    $$Precision=\dfrac{TP}{TP+FP}$$

3.  **Recall**
    Comparison between True Positive (TP) with the amount of data that is actually positive.
    
    $$Recall=\dfrac{TP}{TP+FN}$$

4.  **F1**
    The F1 score combines precision and recall into a single metric, providing an overall assessment of performance. A higher F1 score indicates better performance in terms of both precision and recall, making it a favorable criterion for evaluating models.
    
    $$F1=2\,\,\dfrac{Precision*Recall}{Precision+Recall}$$


From the models performance comparisson we can see **AdaBoost Model demonstrate the highest Accuracy and the highest F1 score**. Thus it become our preferred model. Next we will fine tune the parameters to improve the results.

## Model Tuning
We will try to improve the result from our AdaBoost Model by tuning the hyperparameters.

These are the parameters (excluding random_state) from the [scikit documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) that we could try adjust:
-   **estimator : *object, default=None***
    The base estimator from which the boosted ensemble is built. Support for sample weighting is required, as well as proper classes_ and n_classes_ attributes. If None, then the base estimator is DecisionTreeClassifier initialized with max_depth=1.

-   **n_estimators : *int, default=50***
    The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure is stopped early. Values must be in the range [1, inf).

-   **learning_rate : *float, default=1.0***
    Weight applied to each classifier at each boosting iteration. A higher learning rate increases the contribution of each classifier. There is a trade-off between the learning_rate and n_estimators parameters. Values must be in the range (0.0, inf).

-   **algorithm : *{‘SAMME’, ‘SAMME.R’}, default=’SAMME.R’***
    If ‘SAMME.R’ then use the SAMME.R real boosting algorithm. estimator must support calculation of class probabilities. If ‘SAMME’ then use the SAMME discrete boosting algorithm. The SAMME.R algorithm typically converges faster than SAMME, achieving a lower test error with fewer boosting iterations.

First we try to increase n_estimators value doubling from their default value of 50 to 100, and we half the learning rate to 0.5.

|                               | Accuracy | Precision |  Recall | F1-Score |
|-------------------------------|---------:|----------:|--------:|----------|
| **K-Nearest Neighbors (KNN)** |  0.96435 |  0.962654 | 0.96435 | 0.961790 |
| **Random Forest**             |  0.96830 |  0.967146 | 0.96830 | 0.966169 |
| **AdaBoost**                  |  0.97150 |  0.971428 | 0.97150 | 0.969222 |
| **AdaBoost-Tune**             |  0.97185 |  0.972226 | 0.97185 | 0.969420 |

We can see there's slight improvement on the model with tuned parameters. Now we try to fine tune again to further improve the model by multiplying the n_estimators to 500 and set learning rate to 0.75.

|                               | Accuracy | Precision |  Recall | F1-Score |
|------------------------------:|---------:|----------:|--------:|----------|
| **K-Nearest Neighbors (KNN)** |  0.96435 |  0.962654 | 0.96435 | 0.961790 |
| **Random Forest**             |  0.96830 |  0.967146 | 0.96830 | 0.966169 |
| **AdaBoost**                  |  0.97150 |  0.971428 | 0.97150 | 0.969222 |
| **AdaBoost-Tune**             |  0.97185 |  0.972226 | 0.97185 | 0.969420 |
| **AdaBoost-Tune_2**           |  0.97190 |  0.971992 | 0.97190 | 0.969611 |

As we can see the second iteration from fine tuning the hyperparameters further improve the result. The model can achieve 0.971790 Accuracy with F1-Score 0.969611.

## Conclusion
We have build and optimized a machine learning model using the AdaBoost classification algorithm with an accuracy level of 0.971790 and an F1-Score of 0.969611.
Based on the results of observations and exploration of the available datasets, there is no close relationship between features (low correlation).
There is suggestion to consider for the next improvement 
In this project, we removed a feature (smoking_history) because majority of the data has no info. It is necessary to improve the data collection process if health experts believe smoking history is an important predictor. Data completeness is the key to building a reliable prediction system.

## References
1.  [World Health Organization (2023, April 5). Diabetes. Retrieved May 22, 2023](https://www.who.int/news-room/fact-sheets/detail/diabetes)
2.  [Global Burden of Disease Collaborative Network. Global Burden of Disease Study 2019. Results. Institute for Health Metrics and Evaluation. 2020](https://vizhub.healthdata.org/gbd-results/)
3.  [Dinh, A., Miertschin, S., Young, A. et al. A data-driven approach to predicting diabetes and cardiovascular disease with machine learning. BMC Med Inform Decis Mak 19, 211 (2019).](https://doi.org/10.1186/s12911-019-0918-5)
4.  https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset
5.  [Moore, D. S., Notz, W. I, & Flinger, M. A. (2013). *The basic practice of statistics* (6th ed.). New York, NY: W. H. Freeman and Company.](https://books.google.co.id/books/about/The_Basic_Practice_of_Statistics.html?id=aw61ygAACAAJ&redir_esc=y)

**---This is the end of the report---**
