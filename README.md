# Student Grade Prediction System using Linear Regression

## Project Overview
The Student Grade Prediction System is a machine learning-based web application developed to predict a student’s final academic performance using Linear Regression. The project uses various academic and behavioral factors such as previous exam scores, study time, number of absences, and past failures to estimate the final grade of a student.
The application was built using Python and Streamlit to provide an interactive and user-friendly interface where users can enter student details and instantly receive a predicted final grade along with analytical visualizations.
The main objective of this project is to demonstrate how machine learning techniques can be applied in the field of education to analyze student performance and identify important factors that affect academic results.


## Objectives
- To build a predictive system for estimating student final grades.
- To understand the practical implementation of Linear Regression.
- To analyze the relationship between study habits and academic performance.
- To create an interactive dashboard using Streamlit.
- To visualize educational data using graphs and statistical plots.


## Technologies Used
-Python
-Streamlit
-Pandas
-NumPy
-Matplotlib
-Seaborn
-Scikit-learn


## Dataset Used
The project uses the UCI Student Performance Dataset. The dataset contains information related to student academic records, study patterns, social behavior, and personal attributes.
Some important features used in this project are:
- G1 – First exam grade
- G2 – Second exam grade
- Study Time
- Number of Failures
- Absences
- The target variable for prediction is:
- G3 – Final Grade


## Working of the System
The system first loads the dataset and selects important features related to student performance. The dataset is then divided into training and testing data using the train-test split method.
A Linear Regression model is trained multiple times, and the model with the best accuracy score is selected for prediction. After training, the user can provide input values through the Streamlit sidebar interface.
Based on the entered values, the system predicts the final grade of the student and displays:
- Predicted score out of 20
- Grade category (A, B, C, D, or F)
- Percentage score
- Performance metrics of the model
- Visual analytics and charts


## Features
- Interactive User Interface
- Grade Prediction
- Grade Classification
- Performance Metrics
- Data Visualization

## Machine Learning Algorithm Used
### Linear Regression
Linear Regression is a supervised machine learning algorithm used to predict continuous numerical values. In this project, it is used to estimate the final grade of students based on multiple input features.
The model learns the relationship between independent variables and the final grade by fitting a linear equation to the data.
Why Linear Regression was chosen:
- Simple and easy to understand
- Efficient for numerical prediction
- Suitable for educational datasets
- Provides interpretable coefficients for feature analysis


## Results and Observations
The project successfully predicts student final grades with reasonable accuracy. The analysis shows that previous exam grades (G1 and G2) have the strongest impact on the final result, while factors like absences and failures negatively affect performance.
The visualizations also help in understanding how different student behaviors influence academic outcomes.


## Conclusion
This project demonstrates how machine learning can be used in the education sector to predict student performance and analyze important academic factors. The system combines predictive modeling with data visualization to create an informative and interactive application.
The project also helped in understanding concepts such as data preprocessing, model training, evaluation metrics, and dashboard development using Streamlit.
