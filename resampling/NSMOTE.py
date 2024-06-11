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


num_sample = 10 # Number of samples to be used for synthetic data generation
num_sample_considered = 2*num_sample # Number of samples to be considered for selecting samples from
dataname = 'pima.csv'

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

print('Majority class is ' + str(label[index_of_the_majority]) + '. Number of majority is ' + str(N_majority))
print('Minority class is ' + str(label[index_of_the_minority]) + '. Number of minority is ' + str(N_minority))
print("\n")

needed =int(N_majority - N_minority)
print(str(needed) + ' datapoints need to be generated.\n')

# Calculates distances between the samples
distances = {}
num_same_sample = 0
# Write the header row to the CSV file
with open('datasets/' + dataname + '_distances.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Point 1', 'Point 2', 'Distance'])

for i in range(len(minority)):
    key = i
    distances[key] = []
    for j in range(len(minority)):
        if not np.array_equal(minority[i], minority[j]):
            dist = euclidean(minority[i], minority[j])
            distances[key].append((list(minority[i]), list(minority[j]), dist))
        else:
            if i!= j:   # If there are identical pairs, prints the index of the pair.
                print("Same pair")
                print("index 1 " + str(i))
                print(list(minority[i]))
                print("index 2 " + str(j))
                print(list(minority[j]), "\n")
                num_same_sample += 1

    distances[key].sort(key=lambda x: x[2])

    # Write the distances to a CSV file
    with open('datasets/' + dataname + '_distances.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        distance_list = distances[key]
        for point1, point2, dist in distance_list:
            writer.writerow([str(point1), str(point2), dist])
    
    distances[key] = distances[key][:num_sample_considered]  # Keep the closest num_sample_considered samples to key sample.

print("Same pairs flag is raised " + str(num_same_sample) + " times.") # Make sure it is right! Test is needed.




# Groups the samples to be used for synthetic data generation
used_samples_list = [] # all the sample groups to be used to generate sythetic data
for i in range(needed):
    for key in range(len(distances)):
        reference_sample_key = key # Reference sample
        random_reference_group = np.random.choice(range(num_sample_considered), size=num_sample-1, replace=False)
        used_samples = [] # Creates data groups to be used
        used_samples.append(list(distances[reference_sample_key][0][0])) # Key sample to start

        for j in random_reference_group: # Samples to generate synthetic data
            used_samples.append(list(distances[reference_sample_key][j][1]))
        used_samples_list.append(used_samples)

        if len(used_samples_list) == needed:
                break

    if len(used_samples_list) == needed:
        break



print(str(len(used_samples_list)) + ' sample group are selected to be proceed.')
print("Each group has", num_sample, "samples.\n")


synthetics = []
printed_new = False
printed_sample = False
printed_alpha = False

for i in range(len(used_samples_list)):
    samples = used_samples_list[i]


    if not printed_sample:
        colorama.init()
        print(f"{Fore.YELLOW}{Style.BRIGHT}First selected sample group.{Style.RESET_ALL}")
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
        product = [element * random_nums[j] for element in samples[j]] # product: each sample multiplied by its corresponding alpha.
        new = [new[i] + product[i] for i in range(len(new))] # new: generated synthetic data.

    if not printed_new:
        colorama.init()
        print(f"{Fore.YELLOW}{Style.BRIGHT}First generated synthetic data.{Style.RESET_ALL}")
        colorama.deinit()
        print(new)
        print("\n")
        printed_new = True

    # new = np.ndarray.tolist(new)
    synthetics.append(new)

synthetics = np.asarray(synthetics)
synthetics[:, -1] = label[index_of_the_minority] # Update the label of synthetic data becuase synthetic generation generates different labels.	


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

