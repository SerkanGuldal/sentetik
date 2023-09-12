import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from tabulate import tabulate

def calculate_lasso_and_write_output(input_file):
    """
    Feature Selection Using L1 Regularization (Lasso)
    
    This script performs feature selection using L1 Regularization (Lasso), a method that adds a penalty term to the 
    linear regression or logistic regression cost function. It encourages the model to select a sparse set of features.
    
    Dataset Format:
    - Features are in columns preceding the label column.
    - The label column is the last column.
    - There are no ordinal numbers in the dataset.
    
    Parameters:
    - input_file (str): The name of the input CSV file containing your dataset.
    """
    def calculate_lasso(input_file):
        path = os.path.abspath(os.path.dirname(__file__) + '/../datasets/' + input_file)
        df = pd.read_csv(path)
        NoV = df.shape[1] - 1  # Calculate the number of columns excluding the target (label) column
        X = df.iloc[:, :-1]  # Select all columns except the last one as features
        y = df.iloc[:, -1]   # Select the last column as the target variable
        model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
        model.fit(X, y)
        feature_coefs = model.coef_[0]
        return feature_coefs, NoV, df

    feature_coefs, NoV, df = calculate_lasso(input_file)

    # Sort the features by their coefficients (Lasso regularization)
    sorted_features = sorted(range(NoV), key=lambda i: abs(feature_coefs[i]), reverse=True)

    # Create a directory with a fixed name
    output_directory = os.path.abspath(os.path.dirname(__file__)) + f'/../datasets/'

    os.makedirs(output_directory, exist_ok=True)

    # Create a DataFrame for feature coefficients
    coefs_df = pd.DataFrame({'Feature': df.columns[:-1], 'Coefficient': feature_coefs})

    # Write the feature coefficients to a visually appealing CSV file
    coefs_file = os.path.join(output_directory, f'{input_file}_lasso_coefficients.csv')

    # Convert the DataFrame to a nicely formatted table
    coefs_table = tabulate(coefs_df, headers='keys', tablefmt='pretty', showindex=False)

    # Write the table to the CSV file
    with open(coefs_file, 'w') as f:
        f.write(coefs_table)

    # Append the Lasso coefficients to a new CSV file
    scores_file = os.path.join(output_directory, f'{input_file.split(".csv")[0]}_lasso_scores.csv')
    coefs_df.to_csv(scores_file, mode='a', header=False, index=False)

    for n in range(1, NoV + 1):
        # Select the top N features based on absolute coefficients
        selected_features = sorted_features[:n]

        # Create a DataFrame with the selected features
        selected_df = df.iloc[:, selected_features]

        # Write the selected features to a CSV file
        output_file = os.path.join(output_directory, f'{input_file}_lasso_{n}_features.csv')
        selected_df.to_csv(output_file, index=False)
        print(f'Wrote {n} selected features to {output_file}')

if __name__ == '__main__':
    rawFile = 'yeast3_label_class.csv'  # Update with your CSV file name
    calculate_lasso_and_write_output(rawFile)
