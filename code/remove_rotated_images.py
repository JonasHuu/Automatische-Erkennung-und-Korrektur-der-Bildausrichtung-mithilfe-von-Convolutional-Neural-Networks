import os

def list_files_to_delete(file_path):
    try:
        with open(file_path, 'r') as file:
            filenames = file.read().splitlines()
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return []

    files_to_delete = []
    for filename in filenames:
        if os.path.isfile(filename):
            files_to_delete.append(filename)
        else:
            print(f"File '{filename}' not found.")

    return files_to_delete

def delete_files(files_to_delete):
    for filename in files_to_delete:
        try:
            os.remove(filename)
            print(f"Deleted '{filename}'.")
        except Exception as e:
            print(f"Error deleting '{filename}': {str(e)}")

def main():
    input_file_path = input("Enter the path to the file containing filenames: ")
    files_to_delete = list_files_to_delete(input_file_path)

    print("Files to delete:")
    for file in files_to_delete:
        print(file)

    confirmation = input("Do you want to proceed with the deletion? (yes/no): ")
    if confirmation.lower() == 'yes':
        delete_files(files_to_delete)
    else:
        print("Deletion canceled.")

if __name__ == "__main__":
    main()
