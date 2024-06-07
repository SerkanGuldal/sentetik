import os
import pandas as pd
from sklearn.feature_selection import chi2
from tabulate import tabulate
from sklearn.preprocessing import MinMaxScaler

def calculate_chi2_and_write_output(input_file):
    """
    Feature Selection Using Chi-squared (chi2) Test
    
    This script performs feature selection using the chi-squared (chi2) test, a statistical method that measures the 
    dependency between features and the target variable. It is particularly useful for selecting relevant features 
    for classification tasks. Features are ranked based on their chi2 scores, with higher scores indicating higher 
    relevance.
    
    Dataset Format:
    - Features are in columns preceding the label column.
    - The label column is the last column.
    - There are no ordinal numbers in the dataset.
    
    Parameters:
    - input_file (str): The name of the input CSV file containing your dataset.
    """
    
    def calculate_chi2(input_file):
        # Construct the absolute path to the input file
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', input_file))
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(path)
        
        # Calculate the number of columns excluding the target column
        NoV = df.shape[1] - 1
        
        # Separate the features (X) and the target variable (y)
        X = df.iloc[:, :NoV]
        y = df.iloc[:, NoV]
        
        # Compute chi2 scores for feature selection
        score, _ = chi2(X, y)
        return score, NoV, df

    # Call the calculate_chi2 function to get chi2 scores, number of features, and the dataset
    chi2_scores, NoV, df = calculate_chi2(input_file)

    # Sort the columns by their chi2 scores in descending order
    sorted_columns = sorted(range(NoV), key=lambda i: chi2_scores[i], reverse=True)

    # Create necessary output directories if they don't exist
    datasets_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
    os.makedirs(datasets_folder, exist_ok=True)

    output_data_directory = os.path.join(datasets_folder, f'{input_file}_chi2')
    os.makedirs(output_data_directory, exist_ok=True)

    output_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
    os.makedirs(output_directory, exist_ok=True)

    # Create a DataFrame to store chi2 scores
    chi2_scores_df = pd.DataFrame({'Feature': df.columns[:-1], 'Chi2 Score': chi2_scores})

    # Write the chi2 scores to a visually appealing CSV file in the "datasets" folder
    chi2_scores_file = os.path.join(output_data_directory, f'{input_file}_chi2_scores.out')

    # Convert the DataFrame to a nicely formatted table
    chi2_scores_table = tabulate(chi2_scores_df, headers='keys', tablefmt='pretty', showindex=False)

    # Write the table to the CSV file
    with open(chi2_scores_file, 'w') as f:
        f.write(chi2_scores_table)

    # Write the chi2 scores to a simple CSV file in the "datasets" folder
    chi2_scores_csv_file = os.path.join(output_directory, f'{input_file}_fe.csv')
    if os.path.exists(chi2_scores_csv_file):
        # If the file already exists, update the data for the representing keys
        existing_scores_df = pd.read_csv(chi2_scores_csv_file, index_col='Feature', dtype={'Chi2 Score': str})
        for feature, score in zip(chi2_scores_df['Feature'], chi2_scores_df['Chi2 Score']):
            if feature in existing_scores_df.index:
                existing_scores_df.loc[feature, f'Chi2 Score'] = score
            else:
                existing_scores_df.loc[feature] = [''] * NoV + [score]
        existing_scores_df.to_csv(chi2_scores_csv_file)
    else:
        chi2_scores_df.to_csv(chi2_scores_csv_file, index=False)

    for n in range(1, NoV + 1):
        # Select the top N columns based on chi2 scores
        selected_columns = sorted_columns[:n]

        # Create a DataFrame with the selected columns
        selected_df = df.iloc[:, selected_columns]

        # Write the selected columns to a CSV file under the specific input file's directory
        output_file = os.path.join(output_data_directory, f'{input_file}_chi2_{n}.csv')
        selected_df.to_csv(output_file, index=False)
        print(f'Wrote {n} highest-scoring columns to {output_file}')


    # Normalize chi2 scores to a range of 0 to 100 with three decimal places
    normalized_scores = ((chi2_scores - min(chi2_scores)) / (max(chi2_scores) - min(chi2_scores))) * 100
    normalized_scores = normalized_scores.round(3)

    # Create a DataFrame to store normalized chi2 scores
    normalized_scores_df = pd.DataFrame({'Feature': df.columns[:-1], 'Chi2': normalized_scores})

    # Check if the file already exists
    normalized_scores_csv_file = os.path.join(output_directory, f'{input_file}_fe_normalized.csv')
    if os.path.exists(normalized_scores_csv_file):
        # If the file exists, read its contents
        existing_normalized_scores_df = pd.read_csv(normalized_scores_csv_file)

        # Merge the existing features with the new features based on the 'Feature' column
        merged_normalized_scores_df = existing_normalized_scores_df.merge(normalized_scores_df, on='Feature', how='outer')

        # Fill missing values in the 'Chi2' column with 0
        merged_normalized_scores_df['Chi2'].fillna(0, inplace=True)

        # Write the updated DataFrame back to the same file
        merged_normalized_scores_df.to_csv(normalized_scores_csv_file, index=False)
        print(f'Appended chi2 scores to {normalized_scores_csv_file}')
    else:
        # If the file doesn't exist, create a new one and write the chi2 scores to it
        normalized_scores_df.to_csv(normalized_scores_csv_file, index=False)
        print(f'Wrote chi2 scores to {normalized_scores_csv_file}')


if __name__ == '__main__':
    rawFile = 'yeast3_label_class.csv'  # Update with your CSV file name
    calculate_chi2_and_write_output(rawFile)
