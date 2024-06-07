# Author Serkan Güldal 2024.03.20
# Author Serkan Güldal 2024.06.01
import random
import pandas as pd
import numpy as np
import collections
import os
import sys
from collections import Counter
import random
import colorama
from colorama import Fore, Style
from scipy.spatial.distance import euclidean
import csv

np.set_printoptions(suppress=True)

def generate_random_numbers(num_vars):
    """
    Generate a list of normalized random numbers.

    Parameters:
    num_vars (int): The number of random variables to generate.

    Returns:
    list: A list of normalized random numbers.

    Examples:
    >>> generate_random_numbers(5)
    [0.1805, 0.2352, 0.1231, 0.3069, 0.1543]
    >>> generate_random_numbers(0)
    []
    """
    random_numbers = []
    
    # Generate random numbers
    for i in range(num_vars):
        random_var = random.random()  # Generate a random number between 0 and 1
        random_numbers.append(random_var)

    # Calculate the sum of random numbers
    random_numbers_sum = sum(random_numbers)

    # Normalize the random numbers
    normalized_random_numbers = [x/random_numbers_sum for x in random_numbers]

    return normalized_random_numbers

def load_dataset_with_header_check(dataname):
    def is_header(row):
        try:
            # Attempt to convert each element to a float; if it fails, it is likely a header
            [float(item) for item in row]
            return False
        except ValueError:
            return True

    # Construct the file path
    filepath = 'datasets/' + dataname

    # Read the first row separately
    with open(filepath, 'r') as file:
        first_line = file.readline().strip().split(',')

    # Check if the first row is a header
    header_exists = is_header(first_line)

    # Load the dataset accordingly
    if header_exists:
        print("There is a header")
        data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    else:
        print("There is no header")
        data = np.genfromtxt(filepath, delimiter=',', skip_header=0)
    
    return data


num_sample = 2 # Number of samples to be used for synthetic data generation
dataname = 'page-blocks0.csv'

file = load_dataset_with_header_check(dataname)

# Display the shape of the file
print("\nFile shape:", file.shape)

label, counts = np.unique(file[:,-1], return_counts=True) # label: class, counts: number of each class

# Display the original data profile
print("\nOriginal data profile", label, counts, "\n")

index_of_the_majority = np.argmax(counts)
index_of_the_minority = np.argmin(counts)

minority_index = np.where(file[:,-1] == label[index_of_the_minority])
majority_index = np.where(file[:,-1] == label[index_of_the_majority])

minority = file[minority_index,:][0]
majority = file[majority_index,:][0]

N_majority = majority.shape[0]
N_minority = minority.shape[0]

print('Majority class is ' + str(label[0]) + '. Number of majority is ' + str(N_majority))
print('Minority class is ' + str(label[1]) + '. Number of minority is ' + str(N_minority))
print("\n")

needed =int(N_majority - N_minority)
print(str(needed) + ' datapoints need to be generated.\n')

random_sample_list = []
for i in range(needed):
    random_samples = np.random.randint(0, N_minority, num_sample)
    random_sample_list.append(np.ndarray.tolist(random_samples))

print(str(len(random_sample_list)) + ' sample group are selected to be proceed.')
print("Each group has", num_sample, "samples.\n")




import numpy as np

pairs = {}
for i in range(10):
    key = tuple(minority[i])
    pairs[key] = []
    for j in range(10):  # Avoid comparing the same pairs twice
        if not np.array_equal(minority[i], minority[j]):
            dist = euclidean(minority[i], minority[j])
            pairs[key].append((tuple(minority[i]), tuple(minority[j]), dist))

    pairs[key].sort(key=lambda x: x[2])
    pairs[key] = pairs[key][:5]



# The file path
file_path = 'datasets/' + dataname + '_pairs.txt'

# Write the pairs dictionary to a text file
with open(file_path, "w") as file_pair:
    for key, values in pairs.items():
        file_pair.write(f"{key}:\n")
        for value in values:
            file_pair.write(f"  {value[1]} - Distance: {value[2]}\n")

print(f"pairs have been written to {file_path}")








synthetics = []
printed_new = False
printed_sample = False
printed_alpha = False

for i in random_sample_list:
    samples = minority[i,:]

    if not printed_sample:
        colorama.init()
        print(f"{Fore.YELLOW}{Style.BRIGHT}First selected random sample group.{Style.RESET_ALL}")
        colorama.deinit()
        print(samples, "\n")
        printed_sample = True

    new = [0]*(file.shape[1])
    random_nums = generate_random_numbers(num_sample)
    
    if not printed_alpha:
        colorama.init()
        print(f"{Fore.YELLOW}{Style.BRIGHT}First selected random alpha.{Style.RESET_ALL}")
        colorama.deinit()
        print(random_nums, "\n")
        printed_alpha = True

    for j in range(len(random_nums)):        
        product = random_nums[j]*samples[j] # product: each sample multiplied by its corresponding alpha.
        new += product                      # new: generated synthetic data.

    if not printed_new:
        colorama.init()
        print(f"{Fore.YELLOW}{Style.BRIGHT}First generated synthetic data.{Style.RESET_ALL}")
        colorama.deinit()
        print(new)
        print("\n")
        printed_new = True

    new = np.ndarray.tolist(new)
    synthetics.append(new)

synthetics = np.asarray(synthetics)
synthetics[:, -1] = label[index_of_the_minority]


size = np.shape(synthetics)
print('Synthetically generated data size is ', size[0])

minority_increased = np.concatenate((synthetics, minority))

size = np.shape(minority_increased)
print('New total number of minority class ', size[0])

# Output file.
script_dir = os.path.abspath(os.path.dirname(sys.argv[0]) or '.')
filename = '../datasets/' + dataname + '_' + str(len(random_nums)) + '_SMOTE.csv'
datasets_path = os.path.join(script_dir, filename)

new_dataset = np.concatenate((minority_increased, majority))
label_updated, counts_updated = np.unique(new_dataset[:,-1], return_counts=True)

# Display the original data profile
print("\nOriginal data profile", label, counts)
print("Updated data profile", label_updated, counts_updated)


np.savetxt(datasets_path, new_dataset, delimiter=',', fmt='%.3f')

# This output is for information about the number of resamples

with open('datasets/' + dataname + '_' + str(len(random_nums)) + '_SMOTE' + '_info.txt', 'w') as filehandle:
    filehandle.write("Original distribution is " + str(label) + str(counts))
    filehandle.write("\nResampled distribution is " + str(label_updated) + str(counts_updated))

