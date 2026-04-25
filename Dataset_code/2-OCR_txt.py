import json
import os
from pathlib import Path

def extract_texts_from_json(json_path, output_txt_path=None):
    """
    从JSON文件中提取rec_texts内容并保存为txt文件
    
    Args:
        json_file_path: JSON文件路径
        output_txt_path: 输出txt文件路径，如果为None则使用与JSON文件同名的txt文件
    """
    try:
        json_path = Path(json_path)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 修复：使用Path对象的with_suffix方法
        output_path = json_path.with_suffix('.txt')
        
        # 筛选置信度≥70%的文本
        filtered_texts = [
            text for text, score in zip(data['rec_texts'], data['rec_scores'])
            if score >= 0.0  # 置信度转换为小数形式
        ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(filtered_texts))
            
        #print(f'成功生成：{output_path}')
        return True
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {json_path}")
    except json.JSONDecodeError:
        print(f"错误：{json_path} 不是有效的JSON文件")
    except Exception as e:
        print(f"处理文件 {json_path} 时出错：{e}")

def batch_extract_texts(folder_path):
    """
    批量处理文件夹中的所有JSON文件
    
    Args:
        folder_path: 包含JSON文件的文件夹路径
    """
    folder = Path(folder_path)
    
    # 递归查找所有JSON文件
    json_files = list(folder.rglob('*.json'))
    
    if not json_files:
        print(f"在 {folder_path} 中没有找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    for i, json_file in enumerate(json_files, 1):
        print(f"\n正在处理第 {i}个文件: {json_file.name}")
        extract_texts_from_json(json_file)
    
    print(f"\n批量处理完成！共处理了 {len(json_files)} 个JSON文件")

if __name__ == "__main__":
    

    folder_path = "/...../RP_203/image/train/"  # 修改为你的JSON文件所在目录
    batch_extract_texts(folder_path)
    
  