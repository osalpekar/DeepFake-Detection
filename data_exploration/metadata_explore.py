import os
import json
from glob import glob

data_folder = "/home/ubuntu/cs230_data"
# bucket_dirs = glob("/home/ubuntu/cs230_data/*/")
metadata_files = glob("/home/ubuntu/cs230_data/*/metadata.json")

# Explore the Metadata Files Thoroughly
def main():
    # Metadata will be the concatenation of all the metadata.json files
    metadata = {}
    for metadata_file in metadata_files:
        with open(metadata_file) as f:
            data = json.load(f)
        metadata.update(data)

    print("Total # of Samples in this dataset: " + str(len(metadata)))

    class_distributions(metadata)
    fakes_per_real(metadata)

# This function will analyze the class distributions (real/fake)
def class_distributions(metadata):
    reals = 0
    fakes = 0
    missing = 0
    for sample in metadata:
        if metadata[sample]['label'] == 'FAKE':
            fakes += 1
        elif metadata[sample]['label'] == 'REAL':
            reals += 1
        else:
            missing += 1

    # Compute the class distributions
    print("Number of REAL Samples: " + str(reals))
    print("Number of FAKE Samples: " + str(fakes))
    print("Number of NO LABEL Samples: " + str(missing))

    print("Percentage of REAL Samples: " + str(reals/len(metadata)))
    print("Percentage of FAKE Samples: " + str(fakes/len(metadata)))

def fakes_per_real(metadata):
    # Compute the number of fakes without an associated real image
    fakes = [sample for sample in metadata if metadata[sample]['label'] == 'FAKE']
    num_fakes_without_real = len([fake for fake in fakes if
                                    metadata[fake]['original'] is None])

    print("Number of deepfakes without a corresponding real image: " + str(num_fakes_without_real))

    # Compute the number of fakes associated with each real image
    reals = {sample : 0 for sample in metadata if metadata[sample]['label'] == 'REAL'}
    for sample in metadata:
        if metadata[sample]['label'] == 'FAKE':
            original = metadata[sample]['original']
            reals[original] += 1

    maximum = max(reals, key=reals.get)
    minimum = min(reals, key=reals.get)
    avg = sum(reals.values()) / len(reals)
    print("Number of FAKES per REAL image: ")
    print("min: " + str(reals[minimum]))
    print("max: " + str(reals[maximum]))
    print("avg: " + str(avg))

main()
