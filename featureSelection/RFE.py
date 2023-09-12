import os
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from tabulate import tabulate

def calculate_rfe_and_write_output(input_file):
    """
    Feature Selection Using Recursive Feature Elimination (RFE)
    
    This script performs feature selection using Recursive Feature Elimination (RFE), a method that iteratively 
    selects features by training a model and ranking features based on their importance. It is particularly useful 
    when you have a model in mind (e.g., Logistic Regression) and want to select the most important features for that 
    specific model.
    
    Dataset Format:
    - Features are in columns preceding the label column.
    - The label column is the last column.
    - There are no ordinal numbers in the dataset.
    
    Parameters:
    - input_file (str): The name of the input CSV file containing your dataset.
    """
    def calculate_rfe(input_file):
        path = os.path.abspath(os.path.dirname(__file__) + '/../datasets/' + input_file)
        df = pd.read_csv(path)
        NoV = df.shape[1] - 1  # Calculate the number of columns excluding the target (label) column
        X = df.iloc[:, :-1]  # Select all columns except the last one as features
        y = df.iloc[:, -1]   # Select the last column as the target variable
        model = LogisticRegression()  # You can replace this with your preferred model
        rfe = RFE(model, n_features_to_select=1)
        rfe.fit(X, y)
        rankings = rfe.ranking_
        return rankings, NoV, df

    feature_rankings, NoV, df = calculate_rfe(input_file)

    # Sort the features by their rankings (lower is better)
    sorted_features = sorted(range(NoV), key=lambda i: feature_rankings[i])

    # Create a directory with a fixed name
    output_directory = os.path.abspath(os.path.dirname(__file__)) + f'/../datasets/{input_file}_rfe/'

    os.makedirs(output_directory, exist_ok=True)

    # Create a DataFrame for feature rankings
    rankings_df = pd.DataFrame({'Feature': df.columns[:-1], 'Ranking': feature_rankings})

    # Write the feature rankings to a visually appealing CSV file
    rankings_file = os.path.join(output_directory, f'{input_file}_rfe_rankings.csv')

    # Convert the DataFrame to a nicely formatted table
    rankings_table = tabulate(rankings_df, headers='keys', tablefmt='pretty', showindex=False)

    # Write the table to the CSV file
    with open(rankings_file, 'w') as f:
        f.write(rankings_table)

    for n in range(1, NoV + 1):
        # Select the top N features based on rankings
        selected_features = sorted_features[:n]

        # Create a DataFrame with the selected features
        selected_df = df.iloc[:, selected_features]

        # Write the selected features to a CSV file
        output_file = os.path.join(output_directory, f'{input_file}_rfe_{n}_features.csv')
        selected_df.to_csv(output_file, index=False)
        print(f'Wrote {n} selected features to {output_file}')

if __name__ == '__main__':
    rawFile = 'yeast3_label_class.csv'  # Update with your CSV file name
    calculate_rfe_and_write_output(rawFile)
