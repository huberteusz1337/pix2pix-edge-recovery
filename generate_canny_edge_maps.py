# Skrypt do generowania map krawędzi metodą Canny dla zbioru obrazów.
import os
import pathlib
import cv2
import numpy as np
from tqdm import tqdm


# Katalog z oryginalnymi obrazami
SOURCE_IMAGES_DIR = "/mnt/home/datasets/coco256/coco256_gt_in/train_30000_gt_in" 
# Katalog na obrazy OUT (szkice z krawędziami)
TARGET_OUT_DIR = "/mnt/home/datasets/coco256/coco256_gt_out/train_30000_gt_out_canny_less_edges_200_255" 

# Parametry algorytmu Canny
CANNY_THRESHOLD1 = 200  # Dolny próg
CANNY_THRESHOLD2 = 255  # Górny próg

# --- Tworzenie katalogu docelowego ---
os.makedirs(TARGET_OUT_DIR, exist_ok=True)
# Detekcja krawędzi
def detect_edges_canny(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Wynik: BIAŁE krawędzie na CZARNYM tle
    edges = cv2.Canny(gray_img, CANNY_THRESHOLD1, CANNY_THRESHOLD2,L2gradient=True)
    sketch = cv2.bitwise_not(edges)
    
    return sketch
# Przetwarzanie pojedynczego obrazu
def process_image(image_path, output_dir):
    try:
        img = cv2.imread(str(image_path))
        
        if img is None:
            print(f"Nie udało się wczytać: {image_path.name}")
            return False
        
        sketch = detect_edges_canny(img)
        
        output_path = os.path.join(output_dir, image_path.name)
        cv2.imwrite(output_path, sketch)
        
        return True

    except Exception as e:
        print(f"Błąd podczas przetwarzania {image_path.name}: {e}")
        return False

def main():
    source_path = pathlib.Path(SOURCE_IMAGES_DIR)
    
    # Wyszukaj wszystkie pliki obrazów
    allowed_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_paths = [f for f in source_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in allowed_extensions]

    if not image_paths:
        print(f"Nie znaleziono obrazów w folderze '{SOURCE_IMAGES_DIR}'")
        return

    print(f"Rozpoczynam przetwarzanie obrazów z folderu: '{SOURCE_IMAGES_DIR}'")
    print(f"Znaleziono {len(image_paths)} obraz(ów)")
    print(f"Parametry Canny: threshold1={CANNY_THRESHOLD1}, threshold2={CANNY_THRESHOLD2}")
    
    processed_count = 0
    
    for path in tqdm(image_paths, desc="Przetwarzanie", unit="obraz"):
        if process_image(path, TARGET_OUT_DIR):
            processed_count += 1

    # Podsumowanie
    print(f"\n{'='*60}")
    print(f"Pomyślnie przetworzono: {processed_count}/{len(image_paths)} obraz(ów)")
    print(f"Szkice zapisano w: '{TARGET_OUT_DIR}'")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()