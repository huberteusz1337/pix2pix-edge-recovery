#Skrypt do ewaluacji modelu U-Net dla zadania detekcji krawędzi.
#Oblicza metryki: PSNR, SSIM, ERC oraz F1 Score (ODS/OIS).


import os
import re
import pathlib
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# =============================================================================
# Konfiguracja ścieżek na podstawie nazwy skryptu
# =============================================================================
script_name = os.path.splitext(os.path.basename(__file__))[0]

match = re.match(r'(s3a_)(x\d+_)(\d+)_(\d+)_([a-z]+)', script_name)
if not match:
    raise ValueError(f"Nazwa pliku '{script_name}' nie pasuje do wzorca 's3a_xQP_NNN_S_ttt'")

base_prefix, qp_part, num_str, step_str, script_type = match.groups()
prefix = base_prefix + qp_part
prev_step = f"{int(step_str) - 1}"

exp_dir = f"exp_{prefix}{num_str}_{prev_step}_tgd"
test_dir = os.path.join(exp_dir, f"test_{script_name}")
train_dir = os.path.join(exp_dir, f"train_{prefix}{num_str}_{prev_step}_tgd")
weights_src_dir = os.path.join(train_dir, "weights")
psnr_dir = os.path.join(test_dir, "PSNR_db")
results_dir = os.path.join(test_dir, "results")
erc_dir = os.path.join(test_dir, "EdgeRecoverCoefficient")
ssim_dir = os.path.join(test_dir, "SSIM")
f1_dir = os.path.join(test_dir, "F1_Score")

os.makedirs(test_dir, exist_ok=True)
os.makedirs(psnr_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(erc_dir, exist_ok=True)
os.makedirs(ssim_dir, exist_ok=True)
os.makedirs(f1_dir, exist_ok=True)

# =============================================================================
# Architektura generatora U-Net
# =============================================================================
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(
        filters, size, strides=2, padding='same',
        kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(
        filters, size, strides=2, padding='same',
        kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result


def Generator():
    inputs = tf.keras.layers.Input(shape=[None, None, 3])
    
    down_stack = [
        downsample(32, 4, apply_batchnorm=False),
        downsample(64, 4),
        downsample(128, 4),
        downsample(256, 4),
        downsample(256, 4),
        downsample(256, 4),
        downsample(256, 4),
    ]
    
    up_stack = [
        upsample(256, 4, apply_dropout=True),
        upsample(256, 4, apply_dropout=True),
        upsample(256, 4, apply_dropout=True),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]
    
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        3, 4, strides=2, padding='same',
        kernel_initializer=initializer, activation='tanh')
    
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

# =============================================================================
# Funkcje przetwarzania obrazów
# =============================================================================
def load_image(image_path: str) -> tf.Tensor:
    #Wczytuje obraz PNG jako tensor float32 w zakresie [0, 255].
    image = tf.io.read_file(image_path)
    image = tf.io.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32)
    return image


def normalize_for_generator(image: tf.Tensor) -> tf.Tensor:
    #Normalizuje obraz z zakresu [0, 255] do [-1, 1]."""
    return (image / 127.5) - 1.0


def crop_both_same_window(input_image: tf.Tensor, 
                          org_image: tf.Tensor,
                          crop_size: tuple = (128, 128),
                          edge_threshold: int = 80,
                          grid_size: int = 3) -> tuple:
   
    #Wycina fragment obrazu wejściowego i referencyjnego w tym samym oknie.
    #Wybór okna bazuje na maksymalizacji liczby krawędzi w obrazie referencyjnym.
    
    org_np = org_image.numpy().astype(np.uint8)
    h, w = org_np.shape[:2]
    crop_h, crop_w = crop_size
    
    if h < crop_h or w < crop_w:
        offset_h = max(0, (h - crop_h) // 2)
        offset_w = max(0, (w - crop_w) // 2)
        return (input_image[offset_h:offset_h+crop_h, offset_w:offset_w+crop_w, :],
                org_image[offset_h:offset_h+crop_h, offset_w:offset_w+crop_w, :])
    
    if len(org_np.shape) == 3 and org_np.shape[2] == 3:
        gray_img = cv2.cvtColor(org_np, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = org_np
    
    edge_mask = (gray_img <= edge_threshold)
    best_count = 0
    best_offset_h = (h - crop_h) // 2
    best_offset_w = (w - crop_w) // 2
    
    step_h = (h - crop_h) // (grid_size - 1) if grid_size > 1 else 0
    step_w = (w - crop_w) // (grid_size - 1) if grid_size > 1 else 0
    
    for i in range(grid_size):
        for j in range(grid_size):
            y = i * step_h if i < grid_size - 1 else h - crop_h
            x = j * step_w if j < grid_size - 1 else w - crop_w
            window_edges = np.sum(edge_mask[y:y+crop_h, x:x+crop_w])
            if window_edges > best_count:
                best_count = window_edges
                best_offset_h = y
                best_offset_w = x
    
    input_cropped = input_image[best_offset_h:best_offset_h+crop_h, 
                                best_offset_w:best_offset_w+crop_w, :]
    org_cropped = org_image[best_offset_h:best_offset_h+crop_h, 
                            best_offset_w:best_offset_w+crop_w, :]
    return input_cropped, org_cropped


def edge_prog_image(input_image: tf.Tensor) -> tf.Tensor:
    #Generuje mapę krawędzi Canny dla obrazu wejściowego.
    img_np = input_image.numpy().astype(np.uint8)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 200, 255, L2gradient=True)
    edges = 255 - edges  # Inwersja: białe tło, czarne krawędzie
    edges_rgb = np.stack([edges]*3, axis=-1)
    return tf.convert_to_tensor(edges_rgb, dtype=tf.float32)

# =============================================================================
# Metryki jakości obrazu
# =============================================================================
def psnr_numpy_per_channel(img_ref: np.ndarray, img_test: np.ndarray, 
                           data_range: float = 255.0) -> float:
    #Oblicza PSNR uśredniony po kanałach RGB.
    if img_ref.shape != img_test.shape:
        raise ValueError("Obrazy muszą mieć ten sam kształt")
    if img_ref.shape[-1] != 3:
        raise ValueError("Obrazy muszą mieć 3 kanały RGB")
    
    psnr_values = []
    for c in range(3):
        x = img_ref[..., c].astype(np.float64)
        y = img_test[..., c].astype(np.float64)
        mse = np.mean((x - y) ** 2)
        if mse == 0:
            psnr_values.append(100.0)
        else:
            psnr = 10.0 * np.log10((data_range ** 2) / mse)
            psnr_values.append(psnr)
    return np.mean(psnr_values)


def ssim_numpy(img_ref: np.ndarray, img_test: np.ndarray, 
               data_range: float = 255.0) -> float:
    #Oblicza SSIM uśredniony po kanałach RGB.
    if img_ref.shape != img_test.shape:
        raise ValueError("Obrazy muszą mieć ten sam kształt")
    if img_ref.shape[-1] != 3:
        raise ValueError("Obrazy muszą mieć 3 kanały RGB")
    
    ssim_values = []
    for c in range(3):
        ssim_val = ssim(img_ref[..., c], img_test[..., c], data_range=data_range)
        ssim_values.append(ssim_val)
    return float(np.mean(ssim_values))


def calculate_edge_percentage(image: tf.Tensor, edge_threshold: int = 80) -> float:
    #Oblicza procent pikseli krawędziowych w obrazie.
    img_np = image.numpy().astype(np.uint8)
    
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = img_np
    
    edge_mask = (gray_img <= edge_threshold)
    num_edges = np.sum(edge_mask)
    total_pixels = gray_img.shape[0] * gray_img.shape[1]
    
    return (num_edges / total_pixels) * 100 if total_pixels > 0 else 0

# =============================================================================
# Metryki F1 Score dla detekcji krawędzi (ODS/OIS)
# =============================================================================
def calculate_f1_all_thresholds(pred_image_rgb: np.ndarray, 
                                gt_image_rgb: np.ndarray, 
                                thresholds: np.ndarray,
                                gt_edge_thresh: int = 128) -> dict:
    
    #Oblicza F1, Precision, Recall dla wszystkich progów binaryzacji.
    
    EPS = 1e-12
    
    if pred_image_rgb.shape != gt_image_rgb.shape:
        raise ValueError("Predykcja i GT muszą mieć ten sam rozmiar")
    if pred_image_rgb.shape[2] != 3:
        raise ValueError("Obrazy muszą mieć 3 kanały RGB")
    
    results = {}
    
    for t in thresholds:
        total_tp, total_fp, total_fn = 0, 0, 0
        
        for c in range(3):
            pred_channel = pred_image_rgb[..., c].astype(np.float32)
            gt_channel = gt_image_rgb[..., c].astype(np.uint8)
            
            p_map = 1.0 - (pred_channel / 255.0)
            gt_binary = (gt_channel <= gt_edge_thresh).astype(np.uint8)
            pred_binary = (p_map >= t).astype(np.uint8)
            
            total_tp += int(np.sum((pred_binary == 1) & (gt_binary == 1)))
            total_fp += int(np.sum((pred_binary == 1) & (gt_binary == 0)))
            total_fn += int(np.sum((pred_binary == 0) & (gt_binary == 1)))
        
        precision = total_tp / (total_tp + total_fp + EPS)
        recall = total_tp / (total_tp + total_fn + EPS)
        f1 = 2.0 * precision * recall / (precision + recall + EPS)
        
        results[t] = {
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'tp': total_tp, 'fp': total_fp, 'fn': total_fn
        }
    
    return results


def calculate_ods_ois(all_images_results: list, thresholds: np.ndarray) -> dict:
    
    #Oblicza ODS i OIS na podstawie wyników ze wszystkich obrazów.
    
    EPS = 1e-12
    num_images = len(all_images_results)
    
    if num_images == 0:
        raise ValueError("Lista wyników jest pusta")
    
    # ODS: znajdź najlepszy globalny próg
    ods_best_f1 = -1.0
    ods_best_threshold = 0.0
    ods_best_precision = 0.0
    ods_best_recall = 0.0
    
    for t in thresholds:
        f1_values = [img_results[t]['f1'] for img_results in all_images_results]
        mean_f1 = np.mean(f1_values)
        
        if mean_f1 > ods_best_f1:
            ods_best_f1 = mean_f1
            ods_best_threshold = float(t)
            precision_values = [img_results[t]['precision'] for img_results in all_images_results]
            recall_values = [img_results[t]['recall'] for img_results in all_images_results]
            ods_best_precision = float(np.mean(precision_values))
            ods_best_recall = float(np.mean(recall_values))
    
    # OIS: dla każdego obrazu znajdź najlepszy próg
    best_f1_per_image = []
    best_precision_per_image = []
    best_recall_per_image = []
    
    for img_results in all_images_results:
        best_f1 = -1.0
        best_precision = 0.0
        best_recall = 0.0
        
        for t in thresholds:
            if img_results[t]['f1'] > best_f1:
                best_f1 = img_results[t]['f1']
                best_precision = img_results[t]['precision']
                best_recall = img_results[t]['recall']
        
        best_f1_per_image.append(best_f1)
        best_precision_per_image.append(best_precision)
        best_recall_per_image.append(best_recall)
    
    return {
        'ods': {
            'f1': float(ods_best_f1),
            'threshold': ods_best_threshold,
            'precision': ods_best_precision,
            'recall': ods_best_recall
        },
        'ois': {
            'f1': float(np.mean(best_f1_per_image)),
            'precision': float(np.mean(best_precision_per_image)),
            'recall': float(np.mean(best_recall_per_image))
        }
    }


def get_best_f1_for_image(image_results: dict, thresholds: np.ndarray) -> dict:
    #Znajduje najlepszy próg i metryki F1 dla pojedynczego obrazu.
    best_f1 = -1.0
    best_threshold = 0.0
    best_precision = 0.0
    best_recall = 0.0
    
    for t in thresholds:
        if image_results[t]['f1'] > best_f1:
            best_f1 = image_results[t]['f1']
            best_threshold = float(t)
            best_precision = image_results[t]['precision']
            best_recall = image_results[t]['recall']
    
    return {
        'f1': float(best_f1),
        'threshold': best_threshold,
        'precision': float(best_precision),
        'recall': float(best_recall)
    }

# =============================================================================
# Funkcje zapisu wyników
# =============================================================================
def write_formatted_table(file_path, headers, data, title):
    #Zapisuje dane w formie sformatowanej tabeli do pliku TXT i CSV.
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"=== {title} ===\n\n")
        
        col_widths = [len(str(header)) for header in headers]
        for row in data:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        header_row = " | ".join(str(headers[i]).ljust(col_widths[i]) 
                                for i in range(len(headers)))
        f.write(header_row + "\n")
        
        separator = "-|-".join("-" * col_widths[i] for i in range(len(headers)))
        f.write(separator + "\n")
        
        for row in data:
            data_row = " | ".join(str(row[i]).ljust(col_widths[i]) 
                                  for i in range(min(len(row), len(headers))))
            f.write(data_row + "\n")
    
    csv_path = file_path.replace('.txt', '.csv')
    write_csv_file(csv_path, headers, data)


def write_csv_file(file_path, headers, data):
    #Zapisuje dane w formacie CSV.
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(','.join(str(h) for h in headers) + '\n')
        for row in data:
            f.write(','.join(str(cell) for cell in row[:len(headers)]) + '\n')


def write_or_append_summary(file_path, headers, new_data, title):
    #Zapisuje lub dodaje dane podsumowujące do pliku.
    csv_path = file_path.replace('.txt', '.csv')
    
    if not os.path.exists(file_path):
        write_formatted_table(file_path, headers, new_data, title)
    else:
        col_widths = [len(str(header)) for header in headers]
        for row in new_data:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        with open(file_path, 'a', encoding='utf-8') as f:
            for row in new_data:
                data_row = " | ".join(str(row[i]).ljust(col_widths[i]) 
                                      for i in range(min(len(row), len(col_widths))))
                f.write(data_row + "\n")
        
        with open(csv_path, 'a', encoding='utf-8') as f:
            for row in new_data:
                f.write(','.join(str(cell) for cell in row[:len(headers)]) + '\n')
    
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(','.join(str(h) for h in headers) + '\n')
            for row in new_data:
                f.write(','.join(str(cell) for cell in row[:len(headers)]) + '\n')

# =============================================================================
# Wczytanie wag modelu
# =============================================================================
if not os.path.exists(weights_src_dir):
    raise ValueError(f"Katalog z wagami nie istnieje: {weights_src_dir}")

def extract_step_num(filename):
    match = re.search(r'step_(\d+)', filename)
    return int(match.group(1)) if match else -1

weights_files = sorted(
    [f for f in os.listdir(weights_src_dir) if f.endswith('.h5')],
    key=extract_step_num
)

if not weights_files:
    raise ValueError(f"Nie znaleziono plików wag .h5 w katalogu {weights_src_dir}")

print(f"Znaleziono {len(weights_files)} plików wag")

# =============================================================================
# Konfiguracja zbiorów testowych
# =============================================================================
possible_input_dirs = [
    pathlib.Path("/mnt/home/datasets/coco256/coco256_gt_in/test_3000_gt_in_pngs_qp52/"),
]

possible_output_dirs = [
    pathlib.Path("/mnt/home/datasets/coco256/coco256_gt_out/test_3000_gt_out_canny_less_edges_200_255/"),
]

input_test_dir = None
for dir_path in possible_input_dirs:
    if dir_path.exists():
        input_test_dir = dir_path
        break

output_test_dir = None
for dir_path in possible_output_dirs:
    if dir_path.exists():
        output_test_dir = dir_path
        break

if input_test_dir is None:
    input_test_dir = possible_input_dirs[0]
if output_test_dir is None:
    output_test_dir = possible_output_dirs[0]

input_images = sorted(list(input_test_dir.glob('*.png')))
output_images = sorted(list(output_test_dir.glob('*.png')))

if not input_images:
    raise ValueError(f"Nie znaleziono obrazów wejściowych w katalogu {input_test_dir}")
if len(input_images) != len(output_images):
    raise ValueError("Liczba obrazów wejściowych i wyjściowych się nie zgadza")

print(f"Znaleziono {len(input_images)} par obrazów testowych")

# =============================================================================
# Inicjalizacja modelu
# =============================================================================
generator = Generator()
example_input = tf.random.uniform(shape=(1, 128, 128, 3), minval=-1.0, maxval=1.0)
_ = generator(example_input, training=False)

# =============================================================================
# Pętla testowania po wagach
# =============================================================================
THRESHOLDS = np.linspace(0.001, 0.95, 25)

for i, weight_file in enumerate(weights_files):
    weight_path = os.path.join(weights_src_dir, weight_file)
    print(f"Wczytywanie wag: {weight_file}")
    generator.load_weights(weight_path)
    
    iter_match = re.search(r'step_(\d+)', weight_file)
    iter_num = iter_match.group(1) if iter_match else f"{i+1}"
    
    iter_results_dir = os.path.join(results_dir, f"iter_{iter_num}")
    os.makedirs(iter_results_dir, exist_ok=True)
    
    results_file = os.path.join(psnr_dir, f"iter_{iter_num}.txt")
    erc_results_file = os.path.join(erc_dir, f"iter_{iter_num}.txt")
    ssim_results_file = os.path.join(ssim_dir, f"iter_{iter_num}.txt")
    f1_results_file = os.path.join(f1_dir, f"iter_{iter_num}.txt")
    
    psnr_headers = ["n_img", "prog_org_psnr", "gen_org_psnr", "psnr_improv"]
    erc_headers = ["n_img", "erc_prog", "erc_gen", "erc_improv"]
    ssim_headers = ["n_img", "prog_org_ssim", "gen_org_ssim", "ssim_improv"]
    f1_headers = ["n_img", "f1_prog(ODS)", "f1_gen(ODS)", "f1_improv(ODS)", 
                  "thresh_prog", "thresh_gen"]
    
    summary_file_path = os.path.join(psnr_dir, f"mean_PSNR_{script_name}.txt")
    erc_summary_file_path = os.path.join(erc_dir, f"mean_ERC_{script_name}.txt")
    ssim_summary_file_path = os.path.join(ssim_dir, f"mean_SSIM_{script_name}.txt")
    f1_summary_file_path = os.path.join(f1_dir, f"mean_F1_{script_name}.txt")
    
    # Listy do zbierania metryk
    prog_org_psnrs, gen_org_psnrs, psnr_improvs = [], [], []
    ercs_prog, ercs_gen, ercs_improv = [], [], []
    prog_org_ssims, gen_org_ssims, ssim_improvs = [], [], []
    
    psnr_data, erc_data, ssim_data, f1_data = [], [], [], []
    all_prog_f1_results, all_gen_f1_results = [], []
    images_data, input_images_data = [], []
    
    print(f"  Przetwarzanie {len(input_images)} obrazów...")
    
    for j, (input_path, output_path) in enumerate(zip(input_images, output_images)):
        # Wczytanie obrazów
        input_image = load_image(str(input_path))
        org_image = load_image(str(output_path))
        
        # Przycinanie do 128x128
        input_image, org_image = crop_both_same_window(
            input_image, org_image, crop_size=(128, 128), grid_size=3)
        
        # Generacja mapy krawędzi Canny
        prog_image = edge_prog_image(input_image)[tf.newaxis, ...]
        
        # Zapis obrazów
        input_image_np = input_image.numpy().astype(np.uint8)
        org_image_save_np = org_image.numpy().astype(np.uint8)
        prog_image_save_np = tf.squeeze(prog_image, axis=0).numpy().astype(np.uint8)
        
        img_name = input_path.stem
        cv2.imwrite(os.path.join(iter_results_dir, f"{img_name}_input.png"),
                    cv2.cvtColor(input_image_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(iter_results_dir, f"{img_name}_gt.png"),
                    cv2.cvtColor(org_image_save_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(iter_results_dir, f"{img_name}_canny.png"),
                    cv2.cvtColor(prog_image_save_np, cv2.COLOR_RGB2BGR))
        
        # Predykcja generatora
        normalized_input = normalize_for_generator(input_image)
        normalized_input_batch = normalized_input[tf.newaxis, ...]
        gen_normalized = generator(normalized_input_batch, training=False)
        
        gen_image = (gen_normalized + 1.0) * 127.5
        gen_image = tf.clip_by_value(gen_image, 0.0, 255.0)
        
        # Konwersja do NumPy
        prog_image_np = tf.squeeze(prog_image, axis=0).numpy().astype(np.uint8)
        gen_image_np = tf.squeeze(gen_image, axis=0).numpy().astype(np.uint8)
        org_image_np = org_image.numpy().astype(np.uint8)
        
        # Obliczenie metryk PSNR
        prog_org_psnr = psnr_numpy_per_channel(prog_image_np, org_image_np)
        gen_org_psnr = psnr_numpy_per_channel(gen_image_np, org_image_np)
        psnr_improv = gen_org_psnr - prog_org_psnr
        prog_org_psnrs.append(prog_org_psnr)
        gen_org_psnrs.append(gen_org_psnr)
        psnr_improvs.append(psnr_improv)
        
        # Obliczenie metryk ERC
        erc_prog = calculate_edge_percentage(tf.squeeze(prog_image, axis=0))
        erc_gen = calculate_edge_percentage(tf.squeeze(gen_image, axis=0))
        erc_improv = erc_gen - erc_prog
        ercs_prog.append(erc_prog)
        ercs_gen.append(erc_gen)
        ercs_improv.append(erc_improv)
        
        # Obliczenie metryk SSIM
        prog_org_ssim = ssim_numpy(prog_image_np, org_image_np)
        gen_org_ssim = ssim_numpy(gen_image_np, org_image_np)
        ssim_improv = gen_org_ssim - prog_org_ssim
        prog_org_ssims.append(prog_org_ssim)
        gen_org_ssims.append(gen_org_ssim)
        ssim_improvs.append(ssim_improv)
        
        # Obliczenie F1 Score dla wszystkich progów
        prog_f1_results = calculate_f1_all_thresholds(prog_image_np, org_image_np, THRESHOLDS)
        all_prog_f1_results.append(prog_f1_results)
        
        gen_f1_results = calculate_f1_all_thresholds(gen_image_np, org_image_np, THRESHOLDS)
        all_gen_f1_results.append(gen_f1_results)
        
        prog_best = get_best_f1_for_image(prog_f1_results, THRESHOLDS)
        gen_best = get_best_f1_for_image(gen_f1_results, THRESHOLDS)
        f1_improv = gen_best['f1'] - prog_best['f1']
        
        # Zapis danych
        psnr_data.append([j+1, f"{prog_org_psnr:.4f}", f"{gen_org_psnr:.4f}", 
                         f"{psnr_improv:.4f}"])
        erc_data.append([j+1, f"{erc_prog:.6f}", f"{erc_gen:.6f}", 
                        f"{erc_improv:.6f}"])
        ssim_data.append([j+1, f"{prog_org_ssim:.6f}", f"{gen_org_ssim:.6f}", 
                         f"{ssim_improv:.6f}"])
        
        if j < 5:
            input_images_data.append({
                'input_image': input_image.numpy().astype(np.uint8),
                'org_image': org_image_np,
                'prog_image': prog_image_np,
                'gen_image': gen_image_np,
            })
        
        if (j+1) % 100 == 0:
            print(f"    Przetworzono {j+1}/{len(input_images)} obrazów...")
    
    # Obliczenie ODS i OIS
    print("  Obliczanie ODS i OIS...")
    prog_ods_ois = calculate_ods_ois(all_prog_f1_results, THRESHOLDS)
    gen_ods_ois = calculate_ods_ois(all_gen_f1_results, THRESHOLDS)
    
    # Aktualizacja danych F1 z progami ODS
    f1_data = []
    prog_ods_threshold = prog_ods_ois['ods']['threshold']
    gen_ods_threshold = gen_ods_ois['ods']['threshold']
    
    for j in range(len(all_prog_f1_results)):
        prog_f1_ods = all_prog_f1_results[j][prog_ods_threshold]['f1']
        gen_f1_ods = all_gen_f1_results[j][gen_ods_threshold]['f1']
        f1_improv_ods = gen_f1_ods - prog_f1_ods
        
        f1_data.append([j+1, f"{prog_f1_ods:.6f}", f"{gen_f1_ods:.6f}", 
                       f"{f1_improv_ods:.6f}", f"{prog_ods_threshold:.4f}", 
                       f"{gen_ods_threshold:.4f}"])
        
        if j < 5:
            images_data.append({
                'input_image': input_images_data[j]['input_image'],
                'org_image': input_images_data[j]['org_image'],
                'prog_image': input_images_data[j]['prog_image'],
                'gen_image': input_images_data[j]['gen_image'],
                'f1_prog': prog_f1_ods,
                'f1_gen': gen_f1_ods,
            })
    
    ois_improv = gen_ods_ois['ois']['f1'] - prog_ods_ois['ois']['f1']
    ods_improv = gen_ods_ois['ods']['f1'] - prog_ods_ois['ods']['f1']
    
    # Generowanie wizualizacji
    print("  Generowanie wizualizacji...")
    for j, img_data in enumerate(images_data[:5]):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(img_data['input_image'])
        axes[0].set_title('Input (skompresowany)')
        axes[0].axis('off')
        
        axes[1].imshow(img_data['org_image'])
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(img_data['prog_image'])
        axes[2].set_title(f'Canny (F1: {img_data["f1_prog"]:.3f})')
        axes[2].axis('off')
        
        axes[3].imshow(img_data['gen_image'])
        axes[3].set_title(f'Generator (F1: {img_data["f1_gen"]:.3f})')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(iter_results_dir, f"comparison_{j+1}.png"))
        plt.close()
    
    # Obliczenie średnich
    avg_prog_org_psnr = float(np.mean(prog_org_psnrs))
    avg_gen_org_psnr = float(np.mean(gen_org_psnrs))
    avg_psnr_improv = float(np.mean(psnr_improvs))
    
    avg_erc_prog = float(np.mean(ercs_prog))
    avg_erc_gen = float(np.mean(ercs_gen))
    avg_erc_improv = float(np.mean(ercs_improv))
    
    avg_prog_org_ssim = float(np.mean(prog_org_ssims))
    avg_gen_org_ssim = float(np.mean(gen_org_ssims))
    avg_ssim_improv = float(np.mean(ssim_improvs))
    
    # Dodanie podsumowań
    psnr_data.append(["ŚREDNIA", f"{avg_prog_org_psnr:.4f}", 
                     f"{avg_gen_org_psnr:.4f}", f"{avg_psnr_improv:.4f}"])
    erc_data.append(["ŚREDNIA", f"{avg_erc_prog:.6f}", 
                    f"{avg_erc_gen:.6f}", f"{avg_erc_improv:.6f}"])
    ssim_data.append(["ŚREDNIA", f"{avg_prog_org_ssim:.6f}", 
                     f"{avg_gen_org_ssim:.6f}", f"{avg_ssim_improv:.6f}"])
    
    f1_data.append(["---", "---", "---", "---", "---", "---"])
    f1_data.append(["OIS_PROG", f"{prog_ods_ois['ois']['f1']:.6f}", "-", "-",
                   f"P={prog_ods_ois['ois']['precision']:.4f}", 
                   f"R={prog_ods_ois['ois']['recall']:.4f}"])
    f1_data.append(["OIS_GEN", "-", f"{gen_ods_ois['ois']['f1']:.6f}", 
                   f"{ois_improv:+.6f}",
                   f"P={gen_ods_ois['ois']['precision']:.4f}", 
                   f"R={gen_ods_ois['ois']['recall']:.4f}"])
    f1_data.append(["ODS_PROG", f"{prog_ods_ois['ods']['f1']:.6f}", "-", "-",
                   f"t={prog_ods_ois['ods']['threshold']:.4f}", 
                   f"P={prog_ods_ois['ods']['precision']:.4f}"])
    f1_data.append(["ODS_GEN", "-", f"{gen_ods_ois['ods']['f1']:.6f}", 
                   f"{ods_improv:+.6f}",
                   f"t={gen_ods_ois['ods']['threshold']:.4f}", 
                   f"P={gen_ods_ois['ods']['precision']:.4f}"])
    
    # Zapis wyników
    write_formatted_table(results_file, psnr_headers, psnr_data, 
                         f"WYNIKI PSNR - ITERACJA {iter_num}")
    write_formatted_table(erc_results_file, erc_headers, erc_data, 
                         f"WYNIKI ERC - ITERACJA {iter_num}")
    write_formatted_table(ssim_results_file, ssim_headers, ssim_data, 
                         f"WYNIKI SSIM - ITERACJA {iter_num}")
    write_formatted_table(f1_results_file, f1_headers, f1_data, 
                         f"WYNIKI F1 SCORE - ITERACJA {iter_num}")
    
    # Zapis podsumowań
    summary_headers = ["iter_num", "avg_prog_org_psnr", "avg_gen_org_psnr", 
                      "avg_psnr_improv"]
    erc_summary_headers = ["iter_num", "avg_erc_prog", "avg_erc_gen", 
                          "avg_erc_improv"]
    ssim_summary_headers = ["iter_num", "avg_prog_org_ssim", "avg_gen_org_ssim", 
                           "avg_ssim_improv"]
    f1_summary_headers = ["iter_num", "f1_prog(ODS)", "f1_gen(ODS)", 
                         "f1_improv(ODS)", "f1_gen(OIS)", "threshold(ODS_gen)", 
                         "precision(ODS)", "recall(ODS)"]
    
    summary_psnr_data = [[iter_num, f"{avg_prog_org_psnr:.4f}", 
                         f"{avg_gen_org_psnr:.4f}", f"{avg_psnr_improv:.4f}"]]
    summary_erc_data = [[iter_num, f"{avg_erc_prog:.6f}", 
                        f"{avg_erc_gen:.6f}", f"{avg_erc_improv:.6f}"]]
    summary_ssim_data = [[iter_num, f"{avg_prog_org_ssim:.6f}", 
                         f"{avg_gen_org_ssim:.6f}", f"{avg_ssim_improv:.6f}"]]
    summary_f1_data = [[iter_num, 
                       f"{prog_ods_ois['ods']['f1']:.6f}",
                       f"{gen_ods_ois['ods']['f1']:.6f}",
                       f"{ods_improv:+.6f}",
                       f"{gen_ods_ois['ois']['f1']:.6f}",
                       f"{gen_ods_ois['ods']['threshold']:.4f}",
                       f"{gen_ods_ois['ods']['precision']:.6f}",
                       f"{gen_ods_ois['ods']['recall']:.6f}"]]
    
    write_or_append_summary(summary_file_path, summary_headers, 
                           summary_psnr_data, f"PODSUMOWANIE PSNR - {script_name}")
    write_or_append_summary(erc_summary_file_path, erc_summary_headers, 
                           summary_erc_data, f"PODSUMOWANIE ERC - {script_name}")
    write_or_append_summary(ssim_summary_file_path, ssim_summary_headers, 
                           summary_ssim_data, f"PODSUMOWANIE SSIM - {script_name}")
    write_or_append_summary(f1_summary_file_path, f1_summary_headers, 
                           summary_f1_data, f"PODSUMOWANIE F1 SCORE - {script_name}")
    
    print(f"\n  === PODSUMOWANIE ITERACJI {iter_num} ===")
    print(f"  PSNR: prog={avg_prog_org_psnr:.4f}dB, gen={avg_gen_org_psnr:.4f}dB")
    print(f"  ERC:  prog={avg_erc_prog:.4f}%, gen={avg_erc_gen:.4f}%")
    print(f"  SSIM: prog={avg_prog_org_ssim:.6f}, gen={avg_gen_org_ssim:.6f}")
    print(f"  F1 ODS: prog={prog_ods_ois['ods']['f1']:.6f}, gen={gen_ods_ois['ods']['f1']:.6f}")
    print(f"  F1 OIS: prog={prog_ods_ois['ois']['f1']:.6f}, gen={gen_ods_ois['ois']['f1']:.6f}")

print("\nTestowanie zakończone!")