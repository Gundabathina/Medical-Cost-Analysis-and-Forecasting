# Medical-Cost-Analysis and Forecasting

## Acknowledgements

 I would like to extend my heartfelt gratitude to the following individuals and organizations for their invaluable support and guidance throughout the completion of this project:

Prem Teja Gundabathina: For dedicating time and effort to this project, ensuring its successful completion.
Rohit Kabul: My mentor at the Data Science Internship, Open Source Community, for providing expert guidance, insightful feedback, and unwavering support.
My Family: For their continuous encouragement, love, and support, which have been my greatest source of strength and motivation.
## Appendix

https://docs.python.org/3/

https://pandas.pydata.org/docs/

https://scikit-learn.org/stable/

https://seaborn.pydata.org/

## Authors

- [@Prem_teja] https://github.com/Gundabathina/Medical-Cost-Analysis-and-Forecasting.git


## 🚀 About Me
I am Prem Teja Gundabathina, a Data Analyst with a strong foundation in data analysis, statistical modeling, and data visualization. I have a proven track record of leveraging data to drive informed decision-making and optimize business processes. With expertise in Python, Excel, Power BI, and MySQL, I excel at transforming raw data into actionable insights.






## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://katherineoelsner.com/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/prem-teja-21856a28b)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/)


## 🛠 Skills


Programming Languages: Python, SQL
Data Visualization: Power BI, Matplotlib, Seaborn
Data Analysis: Pandas, NumPy, Scikit-learn
Database Management: MySQL
Tools: Excel, Jupyter Notebook

## 🛠 Projects

Medical-Cost-Analysis and Forecasting: An in-depth analysis and prediction of medical insurance costs using various machine learning techniques. The project involved data preprocessing, exploratory data analysis, and building predictive models to forecast medical expenses based on key factors.
Sales Data Analysis: Analyzed sales data to identify trends, patterns, and insights to drive sales strategies and improve overall performance.
Customer Segmentation: Implemented clustering algorithms to segment customers based on purchasing behavior, enhancing targeted marketing efforts.
Financial Performance Analysis: Conducted financial data analysis to assess company performance, identify growth opportunities, and provide actionable recommendations.
I am passionate about using data to solve complex problems and drive business value. As a recent graduate, I am eager to apply my skills and knowledge in a professional setting and contribute to the success of forward-thinking organizations.

## Installation

You can install these packages using the following command:

```bash
  pip install pandas openpyxl numpy matplotlib seaborn scikit-learn

```
How to Run the Project
1. Clone the Repository: Clone this repository to your local machine using the following command:
```bash
  git clone https://github.com/your-username/Medical-Cost-Analysis.git
```
2.Navigate to the Project Directory: Change your directory to the project folder:
```bash
  cd Medical-Cost-Analysis

```
 3.Install Dependencies: Install the required packages if you haven't already:
```bash
  pip install pandas openpyxl numpy matplotlib seaborn scikit-learn

```
 4.Run the Jupyter Notebook: Open the Jupyter Notebook and run the cells to execute the analysis and modeling steps:
```bash
  jupyter notebook Medical-Cost-Analysis.ipynb
```

## Running Tests

1. Data Integrity Test
This test checks if the dataset has any missing values and if the data types are correct.

```bash
  def test_data_integrity(df):
    # Check for missing values
    assert df.isnull().sum().sum() == 0, "Data contains missing values"
    
    # Check for correct data types
    assert df['age'].dtype == 'int64', "Incorrect data type for age"
    assert df['sex'].dtype == 'int32', "Incorrect data type for sex"
    assert df['bmi'].dtype == 'float64', "Incorrect data type for BMI"
    assert df['children'].dtype == 'int64', "Incorrect data type for children"
    assert df['smoker'].dtype == 'int32', "Incorrect data type for smoker"
    assert df['region'].dtype == 'int32', "Incorrect data type for region"
    assert df['charges'].dtype == 'float64', "Incorrect data type for charges"

# Load dataset
df = pd.read_excel('Medical Cost Analysis Data Set.xlsx')
# Encode categorical variables
df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])

# Run the data integrity test
test_data_integrity(df)
```

2. Model Training and Prediction Test
This test verifies if the model training and prediction process runs without errors and if the R-squared value is within a reasonable range.
```bash
def test_model_performance(X_train, X_test, y_train, y_test):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    assert r2 > 0.7, f"R-squared value is too low: {r2}"

# Split data into training and testing sets
X = df.drop(['charges'], axis=1)
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Run the model performance test
test_model_performance(X_train, X_test, y_train, y_test)

```
3. Visualization Test
This test ensures that the visualization functions generate plots without errors.
```bash
def test_visualizations(df):
    try:
        # Distribution of charges for smokers and non-smokers
        plt.figure(figsize=(12, 5))
        sns.distplot(df[df['smoker'] == 1]['charges'], color='c', label='Smokers')
        sns.distplot(df[df['smoker'] == 0]['charges'], color='b', label='Non-Smokers')
        plt.title('Distribution of Charges for Smokers and Non-Smokers')
        plt.legend()
        plt.show()
        
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), cmap='coolwarm', annot=True)
        plt.title('Correlation Heatmap')
        plt.show()
        
        print("All visualizations generated successfully.")
    except Exception as e:
        assert False, f"Visualization test failed: {e}"

# Run the visualization test
test_visualizations(df)

```
## Lessons Learned

Key Learnings
Building the "Medical-Cost-Analysis and Forecasting" project provided several valuable insights into data analysis and machine learning. Here are some of the key learnings:

# Data Preprocessing:

Handling Categorical Variables: I learned the importance of encoding categorical variables to make them suitable for machine learning models. Techniques like Label Encoding were particularly useful in transforming these variables.
Dealing with Missing Values: Ensuring data integrity by checking for and handling missing values is crucial. This step is essential to maintain the reliability of the analysis and predictions.
Exploratory Data Analysis (EDA):

# Visualization Techniques: 
EDA using visualization tools like Seaborn and Matplotlib helped uncover relationships and patterns in the data. For instance, visualizing the distribution of medical charges for smokers versus non-smokers provided clear insights into the impact of smoking on medical costs.
Correlation Analysis: Creating correlation heatmaps helped identify the strength and direction of relationships between different variables, aiding in feature selection for predictive modeling.
Predictive Modeling:

Model Selection: Experimenting with different models, such as Linear Regression and Polynomial Regression, highlighted the importance of choosing the right model for the dataset. Each model has its strengths and is suitable for different types of data and relationships.
Model Evaluation: Understanding evaluation metrics like R-squared, Mean Squared Error (MSE), and accuracy score was critical in assessing the performance of the models. These metrics provided a quantitative measure of how well the models performed.
Challenges Faced and Overcoming Them
Data Quality Issues:

# Challenge:
 1. The initial dataset contained missing values and unencoded categorical variables, which could potentially lead to inaccurate analysis and predictions.
Solution: Implementing robust data preprocessing techniques, such as filling missing values and using Label Encoding for categorical variables, ensured data quality and integrity.
Overfitting and Underfitting:

2. Balancing model complexity to avoid overfitting (model too closely fits the training data) and underfitting (model is too simple to capture the underlying trend).
Solution: This was addressed by using techniques like train-test split to validate model performance on unseen data and experimenting with Polynomial Regression to capture non-linear relationships without overfitting.
Visualization and Interpretation:

3. Creating meaningful visualizations that effectively communicate the findings can be challenging, especially with complex datasets.
Solution: Leveraging the capabilities of Seaborn and Matplotlib, I focused on creating clear and insightful visualizations. Iterative refinement and feedback were key to improving the quality and clarity of the visual representations.
Model Evaluation and Selection:

4. Evaluating different models and selecting the one that best fits the data without overfitting or underfitting was challenging.
Solution: I relied on multiple evaluation metrics and cross-validation techniques to ensure a comprehensive assessment of model performance. This approach helped in selecting the most appropriate model for the dataset.

# Conclusion
The "Medical-Cost-Analysis and Forecasting" project was a significant learning experience, enhancing my skills in data preprocessing, exploratory data analysis, and predictive modeling. It also underscored the importance of robust data handling, effective visualization, and thorough model evaluation. These lessons will undoubtedly be valuable in future data analysis and machine learning projects.


## Usage/Examples
Example 1: Loading and Preprocessing Data

This example demonstrates how to load the dataset, encode categorical variables, and check for missing values.
```javascript
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_excel('Medical Cost Analysis Data Set.xlsx')

# Encode categorical variables
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])

# Check for missing values
print(df.isnull().sum())


```
Example 2: Exploratory Data Analysis (EDA)

This example shows how to visualize the distribution of medical charges for smokers and non-smokers.

```javascript
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of charges for smokers and non-smokers
plt.figure(figsize=(12, 5))
sns.distplot(df[df['smoker'] == 1]['charges'], color='c', label='Smokers')
sns.distplot(df[df['smoker'] == 0]['charges'], color='b', label='Non-Smokers')
plt.title('Distribution of Charges for Smokers and Non-Smokers')
plt.legend()
plt.show()


```
Example 3: Correlation Analysis

This example illustrates how to create a correlation heatmap to identify relationships between variables.

```javascript
# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), cmap='coolwarm', annot=True)
plt.title('Correlation Heatmap')
plt.show()


```
Example 4: Linear Regression Model

This example demonstrates how to build and evaluate a linear regression model to predict medical costs.

```javascript
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Split data into features and target variable
X = df.drop(['charges'], axis=1)
y = df['charges']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Initialize and train the model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Evaluate the model
print(f'R-squared: {r2_score(y_test, y_pred)}')
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')


```
Example 5: Polynomial Regression Model

This example shows how to create and evaluate a polynomial regression model for predicting medical costs.
```javascript
from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Split data into training and testing sets
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y, random_state=0)

# Initialize and train the model
plr = LinearRegression()
plr.fit(X_train_poly, y_train_poly)

# Predict on the test set
y_pred_poly = plr.predict(X_test_poly)

# Evaluate the model
print(f'R-squared: {r2_score(y_test_poly, y_pred_poly)}')
print(f'Mean Squared Error: {mean_squared_error(y_test_poly, y_pred_poly)}')



```

Example 6: Visualization of Model Performance

This example illustrates how to visualize the predictions of the linear regression model compared to the actual charges.


```javascript
# Scatter plot of actual vs predicted charges
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Charges')
plt.show()


```

```javascript


```
## Used By

This project is used by the following companies:

- Healthcare Providers
- Insurance Companies
- Policy Makers
- Researchers and Data Scientists


