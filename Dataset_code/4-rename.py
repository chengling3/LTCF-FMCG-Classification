import os

ROOT_DIR = "/...../RP_203/image/train/"
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')


def is_image_file(filename):
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def rename_txt_to_res_txt():
    if not os.path.exists(ROOT_DIR):
        print(f"Error: Root directory does not exist! Path: {ROOT_DIR}")
        return
    if not os.path.isdir(ROOT_DIR):
        print(f"Error: Specified path is not a directory! Path: {ROOT_DIR}")
        return

    print(f"Starting processing root directory: {ROOT_DIR}\n")

    for current_dir, subdirs, files in os.walk(ROOT_DIR):
        image_files = [f for f in files if is_image_file(f)]
        image_basenames = [os.path.splitext(img)[0] for img in image_files]

        if image_basenames:
            print(f"\nImage files found in {current_dir}, matching subfolders will be processed")

        for subdir_name in subdirs:
            if subdir_name in image_basenames:
                subdir_path = os.path.join(current_dir, subdir_name)
                print(f"\nProcessing image-named subfolder: {subdir_path}")

                for filename in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, filename)

                    if (os.path.isfile(file_path)
                            and filename.endswith(".txt")
                            and "_res.txt" not in filename):
                        name_without_ext = os.path.splitext(filename)[0]
                        new_filename = f"{name_without_ext}_res.txt"
                        old_path = os.path.join(subdir_path, filename)
                        new_path = os.path.join(subdir_path, new_filename)

                        os.rename(old_path, new_path)
                        print(f"Modified: {filename} → {new_filename}")

    print("\nAll files processed successfully!")


if __name__ == "__main__":
    rename_txt_to_res_txt()