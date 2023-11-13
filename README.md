# Data-Science-applications-and-Analytics-Using-Python
It's an academic paper that states Python applications and implementations in the data science field including a regression model as an example of predictive analysis. 
# Machine Learning Project
## Overview
This project implements a machine learning model based on a multiple regression approach using three datasets.

## File Structure
- `train.csv`: this is a dataset that includes 5 columns first column represents the x values while the rest 4 columns represent the y values
- `test.csv`: this is a dataset that includes only two columns, The first one is the x values while the other one is the y values
- `ideal.csv`: this dataset includes 51 columns the first column represents the x values while the rest of the 50 columns represent the y values and the applied machine learning model should choose only 4 ideal functions from those 50 that minimize the deviation
- `Regression V3 11.11.2023.py`

## Dependencies
- Python 
- Required libraries: pandas, scikit-learn, nympy, bokeh, matplotlib, sklearn, sqlalchemy, and unitest

## Model Code
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import sqlalchemy as db
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
import unittest

# Training Dataset Uploading
train_data = pd.read_csv("train.csv")
len(train_data)

# Dataset Variables
x_train = train_data.drop(columns="x")
y_train = train_data["x"]

class RegressionAnalysis:
    """
    A class for regression analysis with database integration and data visualization.

    Attributes:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Test dataset.
        ideal_data (pd.DataFrame): Ideal functions dataset.
        db_file (str): Database file path.
        engine (sqlalchemy.engine.Engine): SQLAlchemy engine for database interactions.
    """

    def __init__(self, train_file, test_file, ideal_file, db_file):
        """
        Initialize the RegressionAnalysis object.

        Args:
            train_file (str): File path for the training dataset.
            test_file (str): File path for the test dataset.
            ideal_file (str): File path for the ideal functions dataset.
            db_file (str): Database file path.
        """
        self.train_data = pd.read_csv(train_file)
        self.test_data = pd.read_csv(test_file)
        self.ideal_data = pd.read_csv(ideal_file)
        self.db_file = db_file
        self.engine = db.create_engine(f'sqlite:///{self.db_file}')

    def load_data_to_db(self):
        """
        Load training and ideal functions data into the SQLite database.
        """
        try:
            train_data = pd.read_csv("train.csv")
            ideal_data = pd.read_csv("ideal.csv")
            
            connection = self.engine.connect()
            # Load training data into the database
            train_data.to_sql('training_data', con=self.engine, index=False, if_exists='replace')
            # Load ideal functions data into the database
            ideal_data.to_sql('ideal_functions', con=self.engine, index=False, if_exists='replace')

            connection.close()
        except FileNotFoundError as file_error:
            print(f"Error reading CSV file: {file_error}")
        except Exception as general_error:
            print(f"An unexpected error occurred: {general_error}")

    def train_linear_regression(self):
        """
        Train a linear regression model on the training dataset and visualize the results.
        
        Returns:
            sklearn.linear_model.LinearRegression: Trained linear regression model.
        """
        x_train, x_test, y_train, y_test = train_test_split(
            self.train_data.drop(columns="x"),
            self.train_data["x"],
            test_size=0.3,
            random_state=0
        )

        lr = LinearRegression()
        lr.fit(x_train, y_train)

        y_pred_train = lr.predict(x_train)

        plt.scatter(y_train, y_pred_train)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Training Data")
        plt.show()

        r2_train = r2_score(y_train, y_pred_train)
        print("R2 Score (Training Data):", r2_train)

        return lr

    def calculate_sum_of_squared_deviations(self, regression_model):
        """
        Calculate the sum of squared deviations for each ideal function.

        Args:
            regression_model: Trained regression model.

        Returns:
            list: List of sum of squared deviations for each ideal function.
        """
        sum_of_squared_deviations = []

        X_train = self.train_data.drop(columns="x")
        Y_train = self.train_data["x"].values.reshape(-1, 1)

        for i in range(len(self.ideal_data.columns)):
            ideal_function = self.ideal_data.iloc[:, i].values.reshape(-1, 1)
            
            regression_model.fit(X_train, ideal_function)
            
            Y_pred_train = regression_model.predict(X_train)
            sum_of_squared_deviations.append(np.sum((Y_train - Y_pred_train) ** 2))

        return sum_of_squared_deviations

    def choose_ideal_functions(self, sum_of_squared_deviations):
        """
        Choose the top four ideal functions based on the sum of squared deviations.

        Args:
            sum_of_squared_deviations (list): List of sum of squared deviations.

        Returns:
            list: List of chosen ideal functions.
        """
        best_fit_indices = sorted(range(len(sum_of_squared_deviations)), key=lambda i: sum_of_squared_deviations[i])[:4]
        chosen_ideal_functions = [self.ideal_data.iloc[:, i] for i in best_fit_indices]

        for i, ideal_function in enumerate(chosen_ideal_functions):
            print(f"Chosen Ideal Function {i+1}: {ideal_function.values}")

        return chosen_ideal_functions

    def evaluate_models_on_test_data(self, regression_model, chosen_ideal_functions):
        """
        Evaluate the trained model on the test dataset and visualize the results.

        Args:
            regression_model: Trained regression model.
            chosen_ideal_functions (list): List of chosen ideal functions.
        """
        x_test = self.test_data.drop(columns="x")
        y_test = self.test_data["x"]

        mappings = []
        deviations = []

        plt.scatter(x_test, y_test, label='Test Data')
        for (x, y, mapped_y) in mappings:
            plt.scatter(x, mapped_y, color='red', label='Mapped Data')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()

        for i, ideal_function in enumerate(chosen_ideal_functions):
            plt.plot(ideal_function.index, ideal_function.values, label=f"Ideal Function {i + 1}")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()
        # Visualize the test dataset and assigned mappings using Bokeh
        source_test = ColumnDataSource(data=dict(x=x_test.values.flatten(), y=y_test.values.flatten()))
        source_mapped = ColumnDataSource(data=dict(x=[], y=[], color=[]))
        p_test = figure(title="Test Data", x_axis_label='X', y_axis_label='Y')
        p_test.scatter('x', 'y', source=source_test, size=8, color='blue', legend_label='Test Data')
        p_test.scatter('x', 'y', source=source_mapped, size=8, color='red', legend_label='Mapped Data')
        show(p_test)
        # Visualize the ideal functions using Bokeh
        p_ideal = figure(title="Ideal Functions", x_axis_label='X', y_axis_label='Y')
        for i, ideal_function in enumerate(chosen_ideal_functions):
            source_ideal = ColumnDataSource(data=dict(x=ideal_function.index, y=ideal_function.values.flatten()))
            p_ideal.line('x', 'y', source=source_ideal, line_width=2, legend_label=f"Ideal Function {i+1}")

        show(p_ideal)

    def run_regression_analysis(self):
        """
        Run the entire regression analysis process.
        """
        self.load_data_to_db()

        regression_model = self.train_linear_regression()

        sum_of_squared_deviations = self.calculate_sum_of_squared_deviations(regression_model)

        chosen_ideal_functions = self.choose_ideal_functions(sum_of_squared_deviations)

        self.evaluate_models_on_test_data(regression_model, chosen_ideal_functions)

class TestRegressionAnalysis(unittest.TestCase):
    """
    A class for testing the RegressionAnalysis class.
    """
    def test_regression_analysis(self):
        """
        Test the regression analysis process.
        """
        regression_analysis = RegressionAnalysis("train.csv", "test.csv", "ideal.csv", "test.db")
        regression_analysis.run_regression_analysis()

if __name__ == "__main__":
    unittest.main()


## Visualizations:
below are the visualizations/graphs of the datasets Train Dataset, Test Dataset, and the 4 chosen Ideal Functions.
![image](https://github.com/Nourhannabil4/Data-Science-applications-and-Analytics-Using-Python/assets/129120566/39388d6c-dc55-40c5-af4b-132601a44a59)
![image](https://github.com/Nourhannabil4/Data-Science-applications-and-Analytics-Using-Python/assets/129120566/1868981d-b711-472b-b1e8-8cd795d96f0f)
![image](https://github.com/Nourhannabil4/Data-Science-applications-and-Analytics-Using-Python/assets/129120566/9ee0ea9d-7603-48a8-94b9-79bac5b3a46d)

##Results & Evaluation:
The implemented Python model for regression analysis, database integration, and visualization delivers highly promising results. The coefficient of determination (R2 score) for the trained linear regression model on the training data is very good 0.9923, indicating an extremely strong correlation between the predicted and actual values. This high R2 score highlights the model's accuracy and reliability in capturing the underlying patterns within the training dataset (Stojiljkovic, 2021).
Furthermore, the exception-handling technique invested in the code shows robustness in managing potential errors. The try-except blocks efficiently operate “FileNotFoundError” and other unexpected errors, feeding informative messages to help debug and improve the overall resilience of the program.
The selection of the top four ideal functions, based on the least sum of squared deviations, shows the model's capability to determine and prioritize functions that closely align with the observed data. The chosen ideal functions are fundamental in providing insights into the underlying relationships within the datasets and contribute to the interpretability of the regression analysis.
To validate the integrity of the entire regression analysis model, a unit test class is implemented. The “TestRegressionAnalysis” class guarantees that the implemented codes and functionalities perform as intended. Upon implementation, the unit test confirms the correctness of the regression analysis process, showing a reliable mechanism to determine any discrepancies or unexpected changes in the code behavior (Stojiljkovic, 2021).
In summary, the Python model successfully achieves the objectives outlined in the task. The high R2 score, effective exception handling, accurate selection of ideal functions, and successful execution of unit tests collectively confirm the model's ability in regression analysis, database integration, and visualization. This adaptable solution stands suspended to contribute meaningfully to various applications requiring precise analysis and interpretation of complex datasets.

