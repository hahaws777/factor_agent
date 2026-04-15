#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量打包所有因子目录下的文件，支持并行处理
"""
import os
import zipfile
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

def pack_directory_files(source_dir, output_dir, num_parts=8, max_workers=4):
    """
    打包一个目录下的所有文件，分成指定数量的zip文件
    
    Args:
        source_dir: 源目录路径
        output_dir: 输出目录路径
        num_parts: 分成多少个zip文件
        max_workers: 并行处理的最大线程数
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # 获取所有npz文件并排序
    files = sorted([f for f in source_path.glob("*.npz")])
    total_files = len(files)
    
    if total_files == 0:
        print(f"{source_dir}: 没有找到文件")
        return 0, 0
    
    dir_name = source_path.name
    
    # 计算每个zip文件应该包含多少个文件
    files_per_part = math.ceil(total_files / num_parts)
    
    print(f"\n{dir_name}: 找到 {total_files} 个文件，将分成 {num_parts} 个zip文件（每个约 {files_per_part} 个文件）")
    
    def create_zip(group_idx, group_files, zip_filename):
        """创建单个zip文件"""
        try:
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in group_files:
                    arcname = file_path.name
                    zipf.write(file_path, arcname)
            
            zip_size = zip_filename.stat().st_size / (1024 * 1024)  # MB
            return group_idx, zip_filename, zip_size, True, None
        except Exception as e:
            return group_idx, zip_filename, 0, False, str(e)
    
    # 准备所有任务
    tasks = []
    for part_idx in range(num_parts):
        start_idx = part_idx * files_per_part
        end_idx = min(start_idx + files_per_part, total_files)
        
        if start_idx >= total_files:
            break
        
        group_files = files[start_idx:end_idx]
        zip_filename = output_path / f"{dir_name}_part{part_idx + 1:02d}.zip"
        tasks.append((part_idx + 1, group_files, zip_filename))
    
    # 并行处理
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(create_zip, idx, files, zip_path): (idx, zip_path.name)
            for idx, files, zip_path in tasks
        }
        
        # 使用tqdm显示进度
        with tqdm(total=len(tasks), desc=f"{dir_name}") as pbar:
            for future in as_completed(future_to_task):
                idx, zip_name = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    group_idx, zip_filename, zip_size, success, error = result
                    if success:
                        pbar.set_postfix_str(f"Part {group_idx}: {zip_size:.1f}MB")
                    else:
                        print(f"错误: {zip_name} - {error}")
                except Exception as e:
                    print(f"错误: {zip_name} - {str(e)}")
                finally:
                    pbar.update(1)
    
    # 显示结果
    successful = [r for r in results if r[3]]
    total_size = sum(r[2] for r in successful)
    print(f"{dir_name}: 完成 {len(successful)}/{len(tasks)} 个zip文件，总大小 {total_size:.2f} MB")
    
    return len(successful), total_size

def pack_all_factors(base_dir="factors_by_type", output_base="factors_zips", num_parts=8, max_workers=4):
    """
    打包所有因子目录，每个目录分成指定数量的zip文件
    
    Args:
        base_dir: 基础目录
        output_base: 输出基础目录
        num_parts: 每个目录分成多少个zip文件
        max_workers: 并行处理的最大线程数
    """
    base_path = Path(base_dir)
    
    # 获取所有子目录
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    print(f"找到 {len(subdirs)} 个目录需要打包")
    print(f"每个目录将分成 {num_parts} 个zip文件")
    
    total_zips = 0
    total_size = 0
    
    # 逐个目录处理（目录之间串行，但每个目录内的zip文件并行）
    for subdir in subdirs:
        output_dir = Path(output_base) / subdir.name
        zips, size = pack_directory_files(subdir, output_dir, num_parts, max_workers)
        total_zips += zips
        total_size += size
    
    print(f"\n所有目录打包完成！")
    print(f"总共生成 {total_zips} 个zip文件")
    print(f"总大小: {total_size / 1024:.2f} GB")

if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[2])
    # 每个目录分成8个zip文件；因子目录在 factors_by_type_npy/factors_by_type
    pack_all_factors(
        base_dir=str(Path("factors_by_type_npy/factors_by_type")),
        output_base="factors_zips",
        num_parts=8,
        max_workers=4,
    )

