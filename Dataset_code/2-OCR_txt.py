import json
import os
from pathlib import Path

def extract_texts_from_json(json_path, output_txt_path=None):
    try:
        json_path = Path(json_path)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        output_path = json_path.with_suffix('.txt')
        
        filtered_texts = [
            text for text, score in zip(data['rec_texts'], data['rec_scores'])
            if score >= 0.0
        ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(filtered_texts))
            
        return True
        
    except FileNotFoundError:
        print(f"Error: File not found {json_path}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file {json_path}")
    except Exception as e:
        print(f"Error processing {json_path}: {e}")

def batch_extract_texts(folder_path):
    folder = Path(folder_path)
    
    json_files = list(folder.rglob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    
    for i, json_file in enumerate(json_files, 1):
        print(f"\nProcessing file {i}: {json_file.name}")
        extract_texts_from_json(json_file)
    
    print(f"\nBatch processing completed! Processed {len(json_files)} JSON files")

if __name__ == "__main__":
    folder_path = "/...../RP_203/image/train/"
    batch_extract_texts(folder_path)