import os
import shutil

def merge_folders(src_dirs, dest_dir):
    """Merge image and label folders from multiple sources into a single destination."""
    for subfolder in ['images', 'labels']:
        dest_subfolder = os.path.join(dest_dir, subfolder)
        os.makedirs(dest_subfolder, exist_ok=True)
        
        for src in src_dirs:
            src_subfolder = os.path.join(src, subfolder)
            if os.path.exists(src_subfolder):
                for file_name in os.listdir(src_subfolder):
                    src_file = os.path.join(src_subfolder, file_name)
                    dest_file = os.path.join(dest_subfolder, file_name)
                    
                    # Ensure no overwrites by renaming if needed
                    if os.path.exists(dest_file):
                        base, ext = os.path.splitext(file_name)
                        counter = 1
                        new_dest_file = os.path.join(dest_subfolder, f"{base}_{counter}{ext}")
                        while os.path.exists(new_dest_file):
                            counter += 1
                            new_dest_file = os.path.join(dest_subfolder, f"{base}_{counter}{ext}")
                        dest_file = new_dest_file
                    
                    shutil.copy2(src_file, dest_file)
                    print(f"Copied {src_file} -> {dest_file}")

if __name__ == "__main__":
    source_dirs = ['/mnt/data/PythonProject/Pig_counting/dataset/pigs_detection/valid', '/mnt/data/PythonProject/Pig_counting/dataset/pigs_detection/test']
    destination_dir = '/mnt/data/PythonProject/Pig_counting/dataset/pigs_detection/val_test'
    merge_folders(source_dirs, destination_dir)
    print("Merging completed successfully!")