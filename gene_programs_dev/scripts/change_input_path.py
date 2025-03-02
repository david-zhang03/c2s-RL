# upon changes in the *.h5ad file paths, this script updates the path in the training_input.pickle
import argparse
import pickle
import os

parser = argparse.ArgumentParser(description="Update the 'input_h5ad_file_path' attribute in training_input.pickle files.")
parser.add_argument('--training_input_path', type=str, default="/home/ddz5/scratch/Cell2GSEA_QA_dataset_models/",
                    help='Path to the directory containing training_input.pickle files')
parser.add_argument('--new_path', type=str, required=True, help="Directory containing updated .h5ad files")

args = parser.parse_args()

# Get absolute paths
training_input_path = os.path.abspath(args.training_input_path)
new_path = os.path.abspath(args.new_path)

h5ad_suffixes = [os.path.basename(file) for file in os.listdir(new_path) if file.endswith('.h5ad')]

print(training_input_path, new_path)

for dirpath, _, files in os.walk(training_input_path):
    for file in files:
        if file == 'training_inputs.pickle':
            pickle_path = os.path.join(dirpath, file)
            
            # Load the pickle file
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            
            # Check if 'input_h5ad_file_path' needs updating
            if 'input_h5ad_file_path' in data:
                original_path = data['input_h5ad_file_path']
                original_suffix = os.path.basename(original_path)

                print(original_suffix)

                # original_suffix = original_suffix.split('/')[1]

                # If the suffix matches one in h5ad_suffixes, update the path
                if original_suffix in h5ad_suffixes:
                    new_h5ad_path = os.path.join(new_path, original_suffix)
                    print(f"Updating '{original_path}' to '{new_h5ad_path}'")
                    data['input_h5ad_file_path'] = new_h5ad_path
                    
                    # Re-pickle the updated data
                    with open(pickle_path, 'wb') as f:
                        pickle.dump(data, f)
