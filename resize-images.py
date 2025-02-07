import os
from PIL import Image

FOLDER_PATH = "data/ORIGINAL_IMAGES"
OUTPUT_PATH = "data/FULL_IMAGES"

def resize_images(folder_path, size=(500, 500)):
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if os.path.isfile(file_path) and filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            try:
                with Image.open(file_path) as img:
                    img_resized = img.resize(size)
                    output_path = os.path.join(OUTPUT_PATH, filename)
                    img_resized.save(output_path, format=img.format)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    resize_images(FOLDER_PATH)
