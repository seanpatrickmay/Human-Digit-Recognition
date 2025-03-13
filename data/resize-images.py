import os
from PIL import Image

FOLDER_PATH = "data/ORIGINAL_IMAGES"
OUTPUT_PATH = "data/FULL_IMAGES"

def orient_image(image):
    try:
        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(274)
            if orientation is not None:
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                if orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
    except Exception as e:
        print("Couldn't apply orientation")
    return image

def resize_images(folder_path, size=(512, 512)):
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if os.path.isfile(file_path) and filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            try:
                with Image.open(file_path) as img:
                    img = orient_image(img)
                    img_resized = img.resize(size)
                    output_path = os.path.join(OUTPUT_PATH, filename)
                    img_resized.save(output_path, format=img.format)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    resize_images(FOLDER_PATH)
