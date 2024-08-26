import os
import pickle
import random

def write_list_to_txt(file_path, data):
    with open(file_path, 'w') as f:
        for item in data:
            f.write("%s\n" % item)

def convert_torch_to_txt(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
    # List all files in the directory
    files = []
    for f in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, f)):
            if os.path.isfile(os.path.join(directory, f + ".txt")):
                continue
            if f.endswith(".txt"):
                continue
            files.append(f)
    # files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and (not os.path.isfile(os.path.join(directory, f + ".txt")) and f.endswith(".txt"))]
    print(f"Found {len(files)} un-converted files in {directory}")
    random.shuffle(files)
    # Process each file
    for file in files:
        if file.endswith(".txt"):
            print(f"Skipping file {file} because it already has a .txt extension.")
            continue
        file_path = os.path.join(directory, file)
        new_file_path = os.path.join(directory, f"{os.path.splitext(file)[0]}.txt")

        if os.path.exists(new_file_path):
            print(f"Skipping file {file} because {new_file_path} already exists.")
            continue
        
        # Load the file using torch.load
        try:
            print(f"Loading {file_path} using pickle.load")
            data = pickle.load(open(file_path, "rb"))
        except Exception as e:
            print(f"Failed to load {file_path} using pickle.load: {e}")
            continue
        
        # Convert the data to a dictionary if it's not already one
        if not isinstance(data, dict):
            print(f"Data in {file_path} is not a dictionary and cannot be converted to TXT (type = {type(data)}).")
            continue

        values = [
            value for key, value in sorted(data.items())
        ]
        values = [
            value[0] if isinstance(value, list) else value for value in values
        ]
        
        try:
            write_list_to_txt(new_file_path, values)
            print(f"Successfully saved {new_file_path}")
        except Exception as e:
            print(f"Failed to save data to {new_file_path}: {e}")


if __name__ == "__main__":
    convert_torch_to_txt("data/cluster")