import ffmpeg 
import os
import subprocess
import glob
from multiprocessing import Pool, cpu_count
from functools import partial

def process_single_image(in_png, out_yuv_dir, out_encoder_dir, out_pngs_qp_dir):
    """Process a single image"""
    file_name = os.path.splitext(os.path.basename(in_png))[0]
    
    print(f"Processing: {file_name}")
    
    out_ffmpeg_yuv = os.path.join(out_yuv_dir, file_name + "_256x256_yuv420p.yuv")
    out_encoder_vvc_file = os.path.join(out_encoder_dir, file_name + "_256x256_yuv420p10le.vvc") 
    out_encoder_yuv_file = os.path.join(out_encoder_dir, file_name + "_256x256_yuv420p10le.yuv")
    out_png_qp_file = os.path.join(out_pngs_qp_dir, file_name + ".png")

    try:
        # Convert PNG to YUV
        ffmpeg.input(in_png)\
        .filter('pad', 'ceil(iw/2)*2', 'ceil(ih/2)*2')\
        .output(out_ffmpeg_yuv,
                f='rawvideo',
                pix_fmt='yuv420p',
                dst_range=1,
                y=None)\
        .run(quiet=True)
        
        # VVC encoding
        encoder_app = "/home/VTM/bin/EncoderAppStatic"
        config_file = "/home/VTM/cfg/encoder_intra_vtm.cfg"
        
        args = [
            encoder_app,
            "-c", config_file,
            "-i", out_ffmpeg_yuv,
            "-o", out_encoder_yuv_file,
            "-b", out_encoder_vvc_file,
            "-wdt", "256",
            "-hgt", "256",
            "-f", "1",
            "-fr", "1",
            "--InternalBitDepth=10",
            "--ConformanceWindowMode=1",
            "--IntraPeriod=1",
            "--FrameSkip=0",
            "--PrintHexPSNR=1",
            "-q", "27"
        ]
        
        subprocess.run(args, check=True, capture_output=True, text=True)

        # Convert back to PNG
        ffmpeg.input(out_encoder_yuv_file,
                    format='rawvideo', 
                    pix_fmt='yuv420p10le', 
                    s='256x256', 
                    src_range='1')\
        .output(out_png_qp_file,
                    vframes=1,
                    update=1,
                    pix_fmt='rgb24',
                    y=None)\
        .run(quiet=True)
        
        print(f"✓ Completed: {file_name}")
        return True
    except Exception as e:
        print(f"✗ Failed: {file_name} - {str(e)}")
        return False

if __name__ == '__main__':
    input_dir = "./datasets/coco256/coco256_gt_in/train_30000_gt_in/"
    base_output_name = "train_30000_gt"

    out_yuv_dir = f"./datasets/{base_output_name}_out_ffmpeg_yuvs"
    out_encoder_dir = f"./datasets/{base_output_name}_out_encoder_qp27"
    out_pngs_qp_dir = f"./datasets/coco256/coco256_gt_in/{base_output_name}_in_pngs_qp27"

    os.makedirs(out_yuv_dir, exist_ok=True)
    os.makedirs(out_encoder_dir, exist_ok=True)
    os.makedirs(out_pngs_qp_dir, exist_ok=True)

    png_files = glob.glob(os.path.join(input_dir, "*.png"))
    png_files.sort()
    
    # Use 75% of CPU cores
    num_processes = max(1, int(cpu_count() * 0.75))
    print(f"Using {num_processes} parallel processes")
    
    process_func = partial(process_single_image, 
                          out_yuv_dir=out_yuv_dir,
                          out_encoder_dir=out_encoder_dir, 
                          out_pngs_qp_dir=out_pngs_qp_dir)
    
    with Pool(num_processes) as pool:
        results = pool.map(process_func, png_files)
    
    successful = sum(results)
    print(f"\nCompleted: {successful}/{len(png_files)} images")