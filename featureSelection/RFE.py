import os
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from tabulate import tabulate

def calculate_rfe_and_write_output(input_file):
    """
    Feature Selection Using Recursive Feature Elimination (RFE)
    
    This script performs feature selection using Recursive Feature Elimination (RFE), a method that iteratively 
    selects features by training a model and RFE features based on their importance. It is particularly useful 
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
        # Construct the absolute path to the input file
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', input_file))
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(path)
        
        # Calculate the number of columns excluding the target column
        NoV = df.shape[1] - 1
        
        # Separate the features (X) and the target variable (y)
        X = df.iloc[:, :-1]  # Select all columns except the last one as features
        y = df.iloc[:, -1]   # Select the last column as the target variable
        
        # Define the model for RFE (you can replace this with your preferred model)
        model = LogisticRegression()
        
        # Initialize RFE with the model and specify the number of features to select (1 in each iteration)
        rfe = RFE(model, n_features_to_select=1)
        
        # Fit RFE to the data
        rfe.fit(X, y)
        
        # Get feature rankings from RFE
        rankings = rfe.ranking_
        
        return rankings, NoV, df

    # Call the calculate_rfe function to get feature rankings, the number of features, and the dataset
    feature_rankings, NoV, df = calculate_rfe(input_file)

    # Sort the features by their rankings (lower is better)
    sorted_features = sorted(range(NoV), key=lambda i: feature_rankings[i], reverse=True)

    # Create necessary output directories if they don't exist
    datasets_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
    os.makedirs(datasets_folder, exist_ok=True)

    output_data_directory = os.path.join(datasets_folder, f'{input_file}_rfe')
    os.makedirs(output_data_directory, exist_ok=True)

    output_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
    os.makedirs(output_directory, exist_ok=True)

    # Create a DataFrame for feature rankings
    rankings_df = pd.DataFrame({'Feature': df.columns[:-1], 'RFE': feature_rankings})

    # Write the feature rankings to a visually appealing CSV file in the "datasets" folder
    rankings_file = os.path.join(output_data_directory, f'{input_file}_rfe_rankings.out')

    # Convert the DataFrame to a nicely formatted table
    rankings_table = tabulate(rankings_df, headers='keys', tablefmt='pretty', showindex=False)

    # Write the table to the CSV file
    with open(rankings_file, 'w') as f:
        f.write(rankings_table)

    # Define the path to the RFE scores CSV file
    mi_scores_csv_file = os.path.join(output_directory, f'{input_file}_fe.csv')

    if os.path.exists(mi_scores_csv_file):
        # If the file already exists, update the data for the respective keys
        existing_scores_df = pd.read_csv(mi_scores_csv_file, index_col='Feature', dtype={'RFE': str})
        for feature, score in zip(rankings_df['Feature'], rankings_df['RFE']):
            if feature in existing_scores_df.index:
                existing_scores_df.loc[feature, 'RFE'] = score
            else:
                existing_scores_df.loc[feature] = [''] * NoV + [score]
        existing_scores_df.to_csv(mi_scores_csv_file)
    else:
        # If the file doesn't exist, create a new one
        rankings_df.to_csv(mi_scores_csv_file, index=False)

    for n in range(1, NoV + 1):
        # Select the top N features based on rankings
        selected_features = sorted_features[:n]

        # Create a DataFrame with the selected features
        selected_df = df.iloc[:, selected_features]

        # Write the selected features to a CSV file
        output_file = os.path.join(output_data_directory, f'{input_file}_rfe_{n}_features.csv')
        selected_df.to_csv(output_file, index=False)
        print(f'Wrote {n} selected features to {output_file}')

    # Normalize RFE scores to a range of 0 to 100 with three decimal places
    normalized_scores = ((feature_rankings - min(feature_rankings)) / (max(feature_rankings) - min(feature_rankings))) * 100
    normalized_scores = normalized_scores.round(3)

    # Create a DataFrame to store normalized RFE scores
    normalized_scores_df = pd.DataFrame({'Feature': df.columns[:-1], 'RFE': normalized_scores})

    # Check if the file already exists
    normalized_scores_csv_file = os.path.join(output_directory, f'{input_file}_fe_normalized.csv')
    if os.path.exists(normalized_scores_csv_file):
        # If the file exists, read its contents
        existing_normalized_scores_df = pd.read_csv(normalized_scores_csv_file)

        # Merge the existing features with the new features based on the 'Feature' column
        merged_normalized_scores_df = existing_normalized_scores_df.merge(normalized_scores_df, on='Feature', how='outer')

        # Fill missing values in the 'RFE' column with 0
        merged_normalized_scores_df['RFE'].fillna(0, inplace=True)

        # Write the updated DataFrame back to the same file
        merged_normalized_scores_df.to_csv(normalized_scores_csv_file, index=False)
        print(f'Appended RFE scores to {normalized_scores_csv_file}')
    else:
        # If the file doesn't exist, create a new one and write the RFE scores to it
        normalized_scores_df.to_csv(normalized_scores_csv_file, index=False)
        print(f'Wrote RFE scores to {normalized_scores_csv_file}')

if __name__ == '__main__':
    rawFile = 'yeast3_label_class.csv'  # Update with your CSV file name
    calculate_rfe_and_write_output(rawFile)
