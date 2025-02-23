import os


def replace_text_in_filename(file_path, old_str, new_str):
    # Get the directory and the filename separately
    folder_path, filename = os.path.split(file_path)

    # Check if the filename contains the old string
    if old_str in filename:
        # Replace old_str with new_str in the filename
        new_filename = filename.replace(old_str, new_str)

        # Get the new file path
        new_file_path = os.path.join(folder_path, new_filename)

        # Rename the file
        os.rename(file_path, new_file_path)
        print(f'Renamed: {file_path} -> {new_file_path}')


def search_and_rename_in_directory(target_directory, old_str, new_str):
    # Walk through all files and subdirectories in the specified path
    for foldername, subfolders, filenames in os.walk(target_directory):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            # Call function to rename the file if it contains the old_str
            replace_text_in_filename(file_path, old_str, new_str)


def replace_text_in_py_files_in_first_layer(directory, search_string, replace_string):
    # Look only one directory deep
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):  # If it's a directory, go one level deep
            for filename in os.listdir(item_path):
                if filename.endswith(".py"):  # Only process .py files
                    file_path = os.path.join(item_path, filename)
                    replace_text_in_file(file_path, search_string, replace_string)


def replace_text_in_file(file_path, search_string, replace_string):
    # Open the file and read its content
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Replace occurrences of the search_string with replace_string
    updated_content = content.replace(search_string, replace_string)

    # Write the updated content back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(updated_content)

    print(f'Updated file content in: {file_path}')


# Specify the directory and the text to search and replace in filenames and .py files
base_directory = r"C:\Users\baciu\Desktop\Neo Training\Transcendence\Trancendence_Versions\Current Version_4"
search_str = "v3"
replace_str = "v4"

# Execute the search and rename for filenames
search_and_rename_in_directory(base_directory, search_str, replace_str)

# Search and replace in .py files in the first layer of subfolders
search_string_in_py = "v3_step"
replace_string_in_py = "v4_step"
replace_text_in_py_files_in_first_layer(base_directory, search_string_in_py, replace_string_in_py)
