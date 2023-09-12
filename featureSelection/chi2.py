import os
import pandas as pd
from sklearn.feature_selection import chi2
from tabulate import tabulate

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
        path = os.path.abspath(os.path.dirname(__file__) + '/../datasets/' + input_file)
        df = pd.read_csv(path)
        NoV = df.shape[1] - 1  # Calculate the number of columns excluding the target column
        X = df.iloc[:, :NoV]
        y = df.iloc[:, NoV]
        score, _ = chi2(X, y)
        return score, NoV, df

    chi2_scores, NoV, df = calculate_chi2(input_file)

    # Sort the columns by their chi2 scores in descending order
    sorted_columns = sorted(range(NoV), key=lambda i: chi2_scores[i], reverse=True)

    # Create the "datasets" folder if it doesn't exist
    datasets_folder = os.path.abspath(os.path.dirname(__file__)) + '/../datasets/'
    os.makedirs(datasets_folder, exist_ok=True)

    # Create a directory for the chi2 results specific to this input file
    output_directory = os.path.join(datasets_folder, f'{input_file}_chi2')
    os.makedirs(output_directory, exist_ok=True)

    # Create a DataFrame for chi2 scores
    chi2_scores_df = pd.DataFrame({'Feature': df.columns[:-1], 'Chi2 Score': chi2_scores})

    # Write the chi2 scores to a visually appealing CSV file in the "datasets" folder
    chi2_scores_file = os.path.join(output_directory, f'{input_file}_chi2_scores.csv')

    # Convert the DataFrame to a nicely formatted table
    chi2_scores_table = tabulate(chi2_scores_df, headers='keys', tablefmt='pretty', showindex=False)

    # Write the table to the CSV file
    with open(chi2_scores_file, 'w') as f:
        f.write(chi2_scores_table)

    # Write the chi2 scores to a simple CSV file in the "datasets" folder
    chi2_scores_csv_file = os.path.join(datasets_folder, f'{input_file}_feature_extraction_scores.csv')
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
        output_file = os.path.join(output_directory, f'{input_file}_chi2_{n}.csv')
        selected_df.to_csv(output_file, index=False)
        print(f'Wrote {n} highest-scoring columns to {output_file}')

if __name__ == '__main__':
    rawFile = 'yeast3_label_class.csv'  # Update with your CSV file name
    calculate_chi2_and_write_output(rawFile)
