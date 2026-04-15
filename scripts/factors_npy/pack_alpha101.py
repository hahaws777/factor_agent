#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将alpha101目录下的101个文件打包成8个zip文件，每组13个文件
"""
import os
import zipfile
from pathlib import Path
from tqdm import tqdm

def pack_alpha101_files():
    """打包alpha101文件"""
    import os
    _root = Path(__file__).resolve().parents[2]
    os.chdir(_root)
    # 源目录
    source_dir = Path("factors_by_type_npy/factors_by_type/alpha101")
    
    # 输出目录
    output_dir = Path("alpha101_zips")
    output_dir.mkdir(exist_ok=True)
    
    # 获取所有npz文件并排序
    files = sorted([f for f in source_dir.glob("*.npz")])
    total_files = len(files)
    
    print(f"找到 {total_files} 个文件")
    
    # 每组13个文件
    files_per_group = 13
    num_groups = 8
    
    # 打包文件
    for group_idx in range(num_groups):
        start_idx = group_idx * files_per_group
        end_idx = min(start_idx + files_per_group, total_files)
        
        if start_idx >= total_files:
            break
        
        group_files = files[start_idx:end_idx]
        zip_filename = output_dir / f"alpha101_part{group_idx + 1:02d}.zip"
        
        print(f"\n正在打包第 {group_idx + 1} 组 ({len(group_files)} 个文件) -> {zip_filename.name}")
        
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in tqdm(group_files, desc=f"Part {group_idx + 1}"):
                # 保持相对路径结构
                arcname = file_path.name
                zipf.write(file_path, arcname)
        
        # 显示文件大小
        zip_size = zip_filename.stat().st_size / (1024 * 1024)  # MB
        print(f"完成: {zip_filename.name} ({zip_size:.2f} MB)")
    
    print(f"\n所有文件已打包完成！共生成 {num_groups} 个zip文件在 {output_dir} 目录下")

if __name__ == "__main__":
    pack_alpha101_files()

