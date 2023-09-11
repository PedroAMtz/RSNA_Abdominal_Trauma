import os 

"""
This script will run only once, the objective is to rename the files
from the segmentations directory adding the .nii extension in order 
for them to be opened by the nibabel library
"""

directory = "D:/Downloads/rsna-2023-abdominal-trauma-detection/segmentations"
niftii_files = os.listdir(directory)

def rename_files(directory_path, files, extension=".nii"):

	print(f"Initializing rename of files adding extension {extension}")

	for file in files:
		if not file.endswith(extension):
			new_file_name = file + extension
			old_name = os.path.join(directory_path, file)
			new_name = os.path.join(directory_path, new_file_name)

			try:
				os.rename(old_name, new_name)
				print(f"Renamed: {file} -> {new_file_name}")
			except Exception as e:
            	print(f"Error renaming {file}: {str(e)}")
            	continue

if __name__ == "__main__":

	rename_files(directory, niftii_files)


