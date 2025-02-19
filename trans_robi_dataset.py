import os
import shutil
from pathlib import Path
from tqdm import tqdm

def reorganize_dataset(root_dir):
    """
    Reorganize dataset from:
        robi/obj_000001/rgb_0.png, mask_0.png, *.npy
    to:
        robi/obj_000001/images/0.png
        robi/obj_000001/masks/0.png
    And remove all .npy files
    """
    root_path = Path(root_dir)
    
    # 遍历所有对象文件夹
    obj_dirs = [d for d in root_path.iterdir() if d.is_dir() and d.name.startswith('obj_')]
    
    for obj_dir in tqdm(obj_dirs, desc="Processing objects"):
        # 创建新的文件夹结构
        images_dir = obj_dir / 'images'
        masks_dir = obj_dir / 'masks'
        
        images_dir.mkdir(exist_ok=True)
        masks_dir.mkdir(exist_ok=True)
        
        # 删除所有.npy文件
        npy_files = list(obj_dir.glob('*.npy'))
        for npy_file in npy_files:
            npy_file.unlink()  # 删除文件
        
        # 获取所有视角的文件
        files = list(obj_dir.glob('*_*.png'))
        
        for file_path in files:
            # 解析文件名获取视角编号
            if file_path.name.startswith('rgb_'):
                view_num = file_path.stem.split('_')[1]
                new_name = f"{view_num}.png"
                shutil.move(str(file_path), str(images_dir / new_name))
            elif file_path.name.startswith('mask_'):
                view_num = file_path.stem.split('_')[1]
                new_name = f"{view_num}.png"
                shutil.move(str(file_path), str(masks_dir / new_name))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Reorganize dataset structure')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of the dataset')
    
    args = parser.parse_args()
    
    reorganize_dataset(args.root_dir)
    print("Dataset reorganization completed!")