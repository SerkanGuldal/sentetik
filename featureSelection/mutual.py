import os
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from tabulate import tabulate


def calculate_mutual_info_and_write_output(input_file):
    """
    Feature Selection Using Mutual Information
    
    This script performs feature selection using Mutual Information, a statistical measure that quantifies the dependency 
    between features and the target variable. It is particularly useful for selecting informative features for 
    classification tasks. Mutual Information scores rank the features based on their information gain with respect to 
    the target variable. Features with higher Mutual Information scores are considered more relevant.
    
    Dataset Format:
    - Features are in columns preceding the label column.
    - The label column is the last column.
    - There are no ordinal numbers in the dataset.
    
    Parameters:
    - input_file (str): The name of the input CSV file containing your dataset.
    """
    
    def calculate_mutual_info(input_file):
        # Construct the absolute path to the input file
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', input_file))
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(path)
        
        # Calculate the number of columns excluding the target column
        NoV = df.shape[1] - 1
        
        # Separate the features (X) and the target variable (y)
        X = df.iloc[:, :NoV]
        y = df.iloc[:, NoV]
        
        # Compute Mutual Information scores for feature selection
        scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
        return scores, NoV, df

    # Call the calculate_mutual_info function to get Mutual Information scores, the number of features, and the dataset
    mi_scores, NoV, df = calculate_mutual_info(input_file)

    # Sort the columns by their Mutual Information scores in descending order
    sorted_columns = sorted(range(NoV), key=lambda i: mi_scores[i], reverse=True)

    # Create a directory for the Mutual Information results specific to this input file
    datasets_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
    os.makedirs(datasets_folder, exist_ok=True)

    output_data_directory = os.path.join(datasets_folder, f'{input_file}_mi')
    os.makedirs(output_data_directory, exist_ok=True)

    output_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
    os.makedirs(output_directory, exist_ok=True)

    # Create a DataFrame to store Mutual Information scores
    mi_scores_df = pd.DataFrame({'Feature': df.columns[:-1], 'Mutual': mi_scores})

    # Write the mututal info scores to a visually appealing CSV file in the "datasets" folder
    mi_scores_file = os.path.join(output_data_directory, f'{input_file}_mi_scores.out')

    # Convert the DataFrame to a nicely formatted table
    chi2_scores_table = tabulate(mi_scores_df, headers='keys', tablefmt='pretty', showindex=False)

    # Write the table to the CSV file
    with open(mi_scores_file, 'w') as f:
        f.write(chi2_scores_table)

    # Define the path to the Mutual Information scores CSV file
    mi_scores_csv_file = os.path.join(output_directory, f'{input_file}_fe.csv')

    if os.path.exists(mi_scores_csv_file):
        # If the file already exists, update the data for the respective keys
        existing_scores_df = pd.read_csv(mi_scores_csv_file, index_col='Feature', dtype={'Mutual': str})
        for feature, score in zip(mi_scores_df['Feature'], mi_scores_df['Mutual']):
            if feature in existing_scores_df.index:
                existing_scores_df.loc[feature, 'Mutual'] = score
            else:
                existing_scores_df.loc[feature] = [''] * NoV + [score]
        existing_scores_df.to_csv(mi_scores_csv_file)
    else:
        # If the file doesn't exist, create a new one
        mi_scores_df.to_csv(mi_scores_csv_file, index=False)

    for n in range(1, NoV + 1):
        # Select the top N columns based on Mutual Information scores
        selected_columns = sorted_columns[:n]

        # Create a DataFrame with the selected columns
        selected_df = df.iloc[:, selected_columns]

        # Write the selected columns to a CSV file
        output_file = os.path.join(output_data_directory, f'{input_file}_mutual_info_{n}.csv')
        selected_df.to_csv(output_file, index=False)
        print(f'Wrote {n} highest-scoring columns to {output_file}')


    # Normalize Mutual Information scores to a range of 0 to 100 with three decimal places
    normalized_scores = ((mi_scores - min(mi_scores)) / (max(mi_scores) - min(mi_scores))) * 100
    normalized_scores = normalized_scores.round(3)

    # Create a DataFrame to store normalized Mutual Information scores
    normalized_scores_df = pd.DataFrame({'Feature': df.columns[:-1], 'Mutual Information': normalized_scores})

    # Check if the file already exists
    normalized_scores_csv_file = os.path.join(output_directory, f'{input_file}_fe_normalized.csv')
    if os.path.exists(normalized_scores_csv_file):
        # If the file exists, read its contents
        existing_normalized_scores_df = pd.read_csv(normalized_scores_csv_file)

        # Merge the existing features with the new features based on the 'Feature' column
        merged_normalized_scores_df = existing_normalized_scores_df.merge(normalized_scores_df, on='Feature', how='outer')

        # Fill missing values in the 'Mutual Information' column with 0
        merged_normalized_scores_df['Mutual Information'].fillna(0, inplace=True)

        # Write the updated DataFrame back to the same file
        merged_normalized_scores_df.to_csv(normalized_scores_csv_file, index=False)
        print(f'Appended Mutual Information scores to {normalized_scores_csv_file}')
    else:
        # If the file doesn't exist, create a new one and write the Mutual Information scores to it
        normalized_scores_df.to_csv(normalized_scores_csv_file, index=False)
        print(f'Wrote Mutual Information scores to {normalized_scores_csv_file}')



if __name__ == '__main__':
    rawFile = 'yeast3_label_class.csv'  # Update with your CSV file name
    calculate_mutual_info_and_write_output(rawFile)
