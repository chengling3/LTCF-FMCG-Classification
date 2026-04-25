import os

# ====================== 核心配置：在此处修改你的根目录路径 ======================
ROOT_DIR = "/...../RP_203/image/train/"  # 替换为你的目标根目录
# 图片文件扩展名列表，可根据需要添加或修改
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')


# ============================================================================


def is_image_file(filename):
    """判断文件是否为图片文件"""
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def rename_txt_to_res_txt():
    """
    遍历ROOT_DIR下的所有文件夹，仅将与图片同名的子文件夹中的txt文件添加_res后缀
    例如：1.jpg对应的1文件夹中的1.txt → 1_res.txt（仅处理未添加过_res的文件）
    """
    # 检查根目录是否存在
    if not os.path.exists(ROOT_DIR):
        print(f"错误：根目录不存在！路径：{ROOT_DIR}")
        return
    if not os.path.isdir(ROOT_DIR):
        print(f"错误：指定路径不是文件夹！路径：{ROOT_DIR}")
        return

    print(f"开始处理根目录：{ROOT_DIR}\n")

    # 遍历根目录下的所有目录
    for current_dir, subdirs, files in os.walk(ROOT_DIR):
        # 找出当前目录下的所有图片文件
        image_files = [f for f in files if is_image_file(f)]

        # 提取图片文件名（不含扩展名）作为需要匹配的子文件夹名称
        image_basenames = [os.path.splitext(img)[0] for img in image_files]

        if image_basenames:
            print(f"\n在目录 {current_dir} 中发现图片文件，对应的同名子文件夹将被处理")

        # 遍历当前目录下的所有子文件夹
        for subdir_name in subdirs:
            # 只处理与图片同名的子文件夹
            if subdir_name in image_basenames:
                subdir_path = os.path.join(current_dir, subdir_name)
                print(f"\n处理与图片同名的子文件夹：{subdir_path}")

                # 遍历该子文件夹内的所有文件
                for filename in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, filename)

                    # 筛选条件：是文件 + 以.txt结尾 + 未包含_res（避免重复修改）
                    if (os.path.isfile(file_path)
                            and filename.endswith(".txt")
                            and "_res.txt" not in filename):
                        # 构造新文件名
                        name_without_ext = os.path.splitext(filename)[0]
                        new_filename = f"{name_without_ext}_res.txt"
                        old_path = os.path.join(subdir_path, filename)
                        new_path = os.path.join(subdir_path, new_filename)

                        # 执行重命名
                        os.rename(old_path, new_path)
                        print(f"已修改：{filename} → {new_filename}")

    print("\n所有文件处理完成！")


if __name__ == "__main__":
    rename_txt_to_res_txt()
