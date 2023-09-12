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
        path = os.path.abspath(os.path.dirname(__file__) + '/../datasets/' + input_file)
        df = pd.read_csv(path)
        NoV = df.shape[1] - 1  # Calculate the number of columns excluding the target column
        X = df.iloc[:, :NoV]
        y = df.iloc[:, NoV]
        scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
        return scores, NoV, df

    mi_scores, NoV, df = calculate_mutual_info(input_file)

    # Sort the columns by their Mutual Information scores in descending order
    sorted_columns = sorted(range(NoV), key=lambda i: mi_scores[i], reverse=True)

    # Create a directory with a fixed name
    output_directory = os.path.abspath(os.path.dirname(__file__)) + f'/../datasets/{input_file}_mutual_info/'

    os.makedirs(output_directory, exist_ok=True)

    # Create a DataFrame for Mutual Information scores
    mi_scores_df = pd.DataFrame({'Feature': df.columns[:-1], 'Mutual Information Score': mi_scores})

    # Write the Mutual Information scores to a visually appealing CSV file
    mi_scores_file = os.path.join(output_directory, f'{input_file}_mutual_info_scores.csv')

    # Convert the DataFrame to a nicely formatted table
    mi_scores_table = tabulate(mi_scores_df, headers='keys', tablefmt='pretty', showindex=False)

    # Write the table to the CSV file
    with open(mi_scores_file, 'w') as f:
        f.write(mi_scores_table)

    for n in range(1, NoV + 1):
        # Select the top N columns based on Mutual Information scores
        selected_columns = sorted_columns[:n]

        # Create a DataFrame with the selected columns
        selected_df = df.iloc[:, selected_columns]

        # Write the selected columns to a CSV file
        output_file = os.path.join(output_directory, f'{input_file}_mutual_info_{n}.csv')
        selected_df.to_csv(output_file, index=False)
        print(f'Wrote {n} highest scoring columns to {output_file}')

if __name__ == '__main__':
    rawFile = 'yeast3_label_class.csv'  # Update with your CSV file name
    calculate_mutual_info_and_write_output(rawFile)
