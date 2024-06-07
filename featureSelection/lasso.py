import os
import pandas as pd
import matplotlib.pyplot as plt
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
        # Construct the absolute path to the input file
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', input_file))
        # Read the CSV file into a DataFrame
        df = pd.read_csv(path)
        # Calculate the number of columns excluding the target column
        NoV = df.shape[1] - 1
        # Separate the features (X) and the target variable (y)
        X = df.iloc[:, :-1]  # Select all columns except the last one as features
        y = df.iloc[:, -1]   # Select the last column as the target variable
        # Define the Lasso model with specified parameters
        model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
        model.fit(X, y)
        # Get feature coefficients from the Lasso model
        feature_coefs = model.coef_[0]
        
        # Normalize the feature coefficients using Min-Max normalization
        normalized_coefs = (feature_coefs - min(feature_coefs)) / (max(feature_coefs) - min(feature_coefs))
        
        return normalized_coefs, NoV, df

    # Call the calculate_lasso function to get feature coefficients, the number of features, and the dataset
    feature_coefs, NoV, df = calculate_lasso(input_file)

    # Sort the features by their absolute coefficients (Lasso regularization)
    sorted_features = sorted(range(NoV), key=lambda i: feature_coefs[i], reverse=True)

    # Create necessary output directories if they don't exist
    datasets_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
    os.makedirs(datasets_folder, exist_ok=True)
    output_data_directory = os.path.join(datasets_folder, f'{input_file}_lasso')
    os.makedirs(output_data_directory, exist_ok=True)
    output_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
    os.makedirs(output_directory, exist_ok=True)

    # Create a DataFrame for feature coefficients
    coefs_df = pd.DataFrame({'Feature': df.columns[:-1], 'Lasso': feature_coefs})

    # Write the feature coefficients to a visually appealing CSV file in the "datasets" folder
    coefs_file = os.path.join(output_data_directory, f'{input_file}_lasso_coefficients.csv')

    # Convert the DataFrame to a nicely formatted table with headers
    coefs_table = tabulate(coefs_df, headers='keys', tablefmt='pretty', showindex=False)

    # Write the table to the CSV file
    with open(coefs_file, 'w') as f:
        f.write(coefs_table)

    # Define the path to the Lasso scores CSV file
    lasso_scores_csv_file = os.path.join(output_directory, f'{input_file}_fe.csv')

    if os.path.exists(lasso_scores_csv_file):
        # If the file already exists, update the data for the respective keys
        existing_scores_df = pd.read_csv(lasso_scores_csv_file, index_col='Feature', dtype={'Lasso': str})
        for feature, score in zip(coefs_df['Feature'], coefs_df['Lasso']):
            if feature in existing_scores_df.index:
                existing_scores_df.loc[feature, 'Lasso'] = score
            else:
                existing_scores_df.loc[feature] = [''] * NoV + [score]
        existing_scores_df.to_csv(lasso_scores_csv_file)
    else:
        # If the file doesn't exist, create a new one
        coefs_df.to_csv(lasso_scores_csv_file, index=False)

    for n in range(1, NoV + 1):
        # Select the top N features based on absolute coefficients
        selected_features = sorted_features[:n]

        # Create a DataFrame with the selected features
        selected_df = df.iloc[:, selected_features]

        # Write the selected features to a CSV file
        output_file = os.path.join(output_data_directory, f'{input_file}_lasso_{n}_features.csv')
        selected_df.to_csv(output_file, index=False)
        print(f'Wrote {n} selected features to {output_file}')

    # Normalize Lasso scores to a range of 0 to 1
    normalized_scores = ((feature_coefs - min(feature_coefs)) / (max(feature_coefs) - min(feature_coefs))) * 100

    # Create a DataFrame to store normalized Lasso scores
    normalized_scores_df = pd.DataFrame({'Feature': df.columns[:-1], 'Lasso': normalized_scores})

    # Check if the file already exists
    normalized_scores_csv_file = os.path.join(output_directory, f'{input_file}_fe_normalized.csv')

    if os.path.exists(normalized_scores_csv_file):
        # If the file exists, read its contents
        existing_normalized_scores_df = pd.read_csv(normalized_scores_csv_file)

        # Merge the existing features with the new features based on the 'Feature' column
        merged_normalized_scores_df = existing_normalized_scores_df.merge(normalized_scores_df, on='Feature', how='outer')

        # Fill missing values in the 'Lasso' column with 0
        merged_normalized_scores_df['Lasso'].fillna(0, inplace=True)

        # Write the updated DataFrame back to the same file
        merged_normalized_scores_df.to_csv(normalized_scores_csv_file, index=False)
        print(f'Appended normalized Lasso scores to {normalized_scores_csv_file}')
    else:
        # If the file doesn't exist, create a new one and write the normalized Lasso scores to it
        normalized_scores_df.to_csv(normalized_scores_csv_file, index=False)
        print(f'Wrote normalized Lasso scores to {normalized_scores_csv_file}')



if __name__ == '__main__':
    rawFile = 'yeast3_label_class.csv'  # Update with your CSV file name
    calculate_lasso_and_write_output(rawFile)