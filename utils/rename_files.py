import os

def change_file_name(directory):
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Check if 'EP' is in the filename
        if 'EP' in filename:
            # Replace 'EP' with 'CP'
            new_filename = filename.replace('EP', 'CP')
            # Get full paths
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(old_file, new_file)
            print(f"Renamed: {filename} -> {new_filename}")

def add_piece_to_file_name(directory, pathway):
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Check if the pathway is missing in the filename
        if pathway not in filename:
            # Split the filename into parts to insert the pathway
            parts = filename.split('_')
            if len(parts) > 2:  # Ensure the filename has enough parts to modify
                new_filename = f"{parts[0]}_{parts[1]}_{pathway}_{'_'.join(parts[2:])}"
                # Get full paths
                old_file = os.path.join(directory, filename)
                new_file = os.path.join(directory, new_filename)
                # Rename the file
                os.rename(old_file, new_file)
                print(f"Renamed: {filename} -> {new_filename}")

"""
# Example usage - specify the directory containing the files
# directory_path = 'path/to/your/data/directory/CP_pathway_both_confounders'
# change_file_name(directory_path)
"""

# Example usage for EP pathway
# directory_path = 'path/to/your/data/directory/EP_pathway_both_confounders'
# add_piece_to_file_name(directory_path, 'EP')