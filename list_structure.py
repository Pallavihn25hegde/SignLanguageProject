# list_structure.py
import os

def list_structure(path=".", level=0):
    if level > 3:  # Limit depth
        return
    
    indent = "  " * level
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            print(f"{indent}📁 {item}")
            if level < 2:  # Only go 2 levels deep
                list_structure(item_path, level + 1)
        else:
            print(f"{indent}📄 {item}")

if __name__ == "__main__":
    print(f"Directory structure of: {os.getcwd()}")
    print("=" * 50)
    list_structure()
    