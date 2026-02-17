# Skrypt do przygotowania zbioru danych COCO - przycinanie i filtrowanie obrazów.

import os
import pathlib
import cv2
import numpy as np
from tqdm import tqdm

# Konfiguracja
SOURCE_IMAGES_DIR = "./datasets/coco/" # Ścieżka do obrazów COCO
TARGET_DIR = "./datasets/coco_prepared/" # Katalog na przetworzone obrazy
IMG_WIDTH = 256
IMG_HEIGHT = 256
EDGE_THRESHOLD = 1500 # Minimalna liczba pikseli krawędzi, aby obraz został zakwalifikowany.

os.makedirs(TARGET_DIR, exist_ok=True)
# Przetwarzanie pojedynczego obrazu
def process_and_save_image(image_path):

    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False

        # Wycinanie fragmentu 256x256 z lewego górnego rogu
        cropped_img = img[0:IMG_HEIGHT, 0:IMG_WIDTH]

        
        if cropped_img.shape[0] != IMG_HEIGHT or cropped_img.shape[1] != IMG_WIDTH:
            return False

        # Detekcja krawędzi do filtrowania
        gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_img, 100, 200)

        #Odrzucenie obrazów z niewystarczającą liczbą krawędzi 
        edge_pixel_count = np.count_nonzero(edges)
        if edge_pixel_count < EDGE_THRESHOLD:
            return False

        #Zapis obrazu
        base_name = os.path.basename(image_path)
        output_path = os.path.join(TARGET_DIR, base_name)
        cv2.imwrite(output_path, cropped_img)

        return True
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return False

def main():
   
    source_path = pathlib.Path(SOURCE_IMAGES_DIR)
    image_paths = list(source_path.glob('*.jpg')) + list(source_path.glob('*.png'))

    print(f"Found {len(image_paths)} images in {SOURCE_IMAGES_DIR}.")
    
    processed_count = 0
    for path in tqdm(image_paths, desc="Processing images"):
        if process_and_save_image(path):
            processed_count += 1

    print(f"\nProcessing complete.")
    print(f"Successfully processed and saved {processed_count} images.")
    print(f"Output images saved to: {TARGET_DIR}")


if __name__ == "__main__":
    main()