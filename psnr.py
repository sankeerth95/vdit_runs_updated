import argparse
import os
import csv
import torch
import torchvision.io
# Import the LPIPS metric along with the existing ones
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

def calculate_video_metrics(gt_video_path, gen_video_path, device):
    """
    Calculates PSNR, SSIM, and LPIPS for each frame between two videos.

    Args:
        gt_video_path (str): Path to the ground truth video.
        gen_video_path (str): Path to the generated video.
        device (torch.device): The device to run computations on (e.g., 'cuda' or 'cpu').

    Returns:
        list: A list of dictionaries, where each dictionary contains the metrics for a single frame.
        dict: A dictionary containing the average metrics over the entire video.
    """
    try:
        # Read video frames. read_video returns a tensor of shape (T, H, W, C)
        # and dtype uint8, where T is the number of frames.
        gt_frames, _, _ = torchvision.io.read_video(gt_video_path, output_format="TCHW", pts_unit="sec")
        gen_frames, _, _ = torchvision.io.read_video(gen_video_path, output_format="TCHW", pts_unit="sec")

    except Exception as e:
        print(f"Error reading video files: {e}")
        return None, None

    # --- Basic Validation ---
    if gt_frames.shape != gen_frames.shape:
        print(f"Warning: Video dimensions or frame counts do not match for {os.path.basename(gt_video_path)}.")
        print(f"  Ground Truth: {gt_frames.shape}")
        print(f"  Generated: {gen_frames.shape}")
        # Truncate to the minimum number of frames as a simple fix
        min_frames = min(gt_frames.shape[0], gen_frames.shape[0])
        gt_frames = gt_frames[:min_frames]
        gen_frames = gen_frames[:min_frames]
        print(f"  Processing the first {min_frames} frames.")


    # --- Preprocessing ---
    # Convert frames from uint8 [0, 255] to float32 [0, 1]
    gt_frames = gt_frames.to(torch.float32) / 255.0
    gen_frames = gen_frames.to(torch.float32) / 255.0

    # Move tensors to the selected device
    gt_frames = gt_frames.to(device)
    gen_frames = gen_frames.to(device)

    # --- Initialize Metrics ---
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    # Initialize LPIPS. normalize=True ensures input in [0, 1] is correctly handled for the network.
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)

    frame_results = []
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0 # New accumulator for LPIPS
    num_frames = gt_frames.shape[0]

    if num_frames == 0:
        print(f"Warning: No frames found in {os.path.basename(gt_video_path)}")
        return [], {}

    # --- Frame-by-Frame Calculation ---
    for i in range(num_frames):
        # Get a single frame for both videos.
        # The metric expects a batch dimension, so we add one with unsqueeze(0).
        # Shape changes from (C, H, W) to (1, C, H, W).
        gt_frame = gt_frames[i].unsqueeze(0)
        gen_frame = gen_frames[i].unsqueeze(0)

        # Calculate metrics
        current_psnr = psnr(gen_frame, gt_frame).item()
        current_ssim = ssim(gen_frame, gt_frame).item()
        current_lpips = lpips(gen_frame, gt_frame).item() # Calculate LPIPS

        total_psnr += current_psnr
        total_ssim += current_ssim
        total_lpips += current_lpips # Add to total

        frame_results.append({
            'video_name': os.path.basename(gt_video_path),
            'frame_index': i,
            'psnr': current_psnr,
            'ssim': current_ssim,
            'lpips': current_lpips # Add LPIPS to frame results
        })

    # --- Calculate Averages ---
    avg_metrics = {
        'video_name': os.path.basename(gt_video_path),
        'average_psnr': total_psnr / num_frames,
        'average_ssim': total_ssim / num_frames,
        'average_lpips': total_lpips / num_frames # Add average LPIPS
    }

    return frame_results, avg_metrics


def main():
    """
    Main function to parse arguments and process video directories.
    """
    # NOTE: You might need to change these paths back to your own
    gt_dir = "/home/sdurvasula/minimal/zeroseed_vids_checkpoint/wan21_1.3b_720x1280x81_baseline"
    gen_dir = "/home/sdurvasula/minimal/zeroseed_vids_checkpoint/wan21_1.3b_720x1280x81_bitmaskcached"

    # gt_dir = "/home/sdurvasula/minimal/zeroseed_vids_checkpoint/wan21_14b_480x832x81_bitmaskcached"
    # gen_dir = "/home/sdurvasula/relatedwork/Sparse-VideoGen/zeroseed_vids/Wan-AI/Wan2.1-T2V-14B-Diffusers"

    gt_dir = "/home/sdurvasula/minimal/zeroseed_vids_checkpoint/wan21_14b_480x832x81_baseline"
    gen_dir = "/home/sdurvasula/minimal/zeroseed_vids_checkpoint/wan21_14b_480x832x81_bitmaskcached"


    # gt_dir = "/home/sdurvasula/minimal/zeroseed_vids_checkpoint/wan21_14b_720x1280x81_baseline"
    # gen_dir = "/home/sdurvasula/relatedwork/Sparse-VideoGen/zeroseed_vids/Wan-AI/Wan2.1-T2V-14B-Diffusers"

    # gt_dir = "/home/sdurvasula/minimal/zeroseed_vids_checkpoint/hunyuan_720x1280x81_baseline"
    # gen_dir = "/home/sdurvasula/minimal/zeroseed_vids_checkpoint/hunyuan_720x1280x81_bitmaskcached"



    output_csv = "video_metrics.csv"


    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        all_files = set()
        gt_files = []
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')

        for root, _, files in os.walk(gt_dir):
            for file in files:
                if file.lower().endswith(video_extensions):
                    # We need the full path, not just the filename
                    if file in all_files:
                        continue
                    all_files.add(file)
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, start=gt_dir)
                    gt_files.append(relative_path)

        gt_files.sort()

    except FileNotFoundError:
        print(f"Error: Ground truth directory not found at {gt_dir}")
        return

    if not gt_files:
        print(f"No video files found in {gt_dir}")
        return

    all_frame_results = []
    all_avg_results = []

    # --- Main Processing Loop ---
    print(f"Found {len(gt_files)} videos to process.")
    for video_name in tqdm(gt_files, desc="Processing Videos"):
        gt_video_path = os.path.join(gt_dir, video_name)
        gen_video_path = os.path.join(gen_dir, video_name)

        if not os.path.exists(gen_video_path):
            print(f"Warning: Corresponding video for {video_name} not found in {gen_dir}. Skipping.")
            continue

        print(f"Processing {video_name}...")
        print(f"  Ground Truth: {gt_video_path}")
        print(f"  Generated: {gen_video_path}")
        frame_results, avg_metrics = calculate_video_metrics(gt_video_path, gen_video_path, device)

        if frame_results and avg_metrics:
            all_frame_results.extend(frame_results)
            all_avg_results.append(avg_metrics)
        print(avg_metrics)



    # --- Save Results to CSV ---
    if not all_frame_results:
        print("No videos were processed successfully. Exiting.")
        return

    try:
        with open(output_csv, 'w', newline='') as csvfile:
            # Add 'lpips' to the CSV header
            fieldnames = ['video_name', 'frame_index', 'psnr', 'ssim', 'lpips']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_frame_results)
        print(f"\nDetailed frame-by-frame results saved to {output_csv}")
    except IOError as e:
        print(f"Error writing to CSV file: {e}")

    # --- Print Summary of Averages ---
    print("\n--- Average Metrics Per Video ---")
    avg_psnr_total = 0
    avg_ssim_total = 0
    avg_lpips_total = 0 # New accumulator for overall LPIPS average
    for res in all_avg_results:
        # Update print statement to include LPIPS
        print(f"  {res['video_name']}: PSNR = {res['average_psnr']:.2f}, SSIM = {res['average_ssim']:.4f}, LPIPS = {res['average_lpips']:.4f}")
        avg_psnr_total += res['average_psnr']
        avg_ssim_total += res['average_ssim']
        avg_lpips_total += res['average_lpips'] # Add to overall total

    # --- Print Overall Averages ---
    if all_avg_results:
        overall_avg_psnr = avg_psnr_total / len(all_avg_results)
        overall_avg_ssim = avg_ssim_total / len(all_avg_results)
        overall_avg_lpips = avg_lpips_total / len(all_avg_results) # Calculate overall LPIPS
        print("\n--- Overall Average Metrics ---")
        print(f"  Average PSNR across all videos: {overall_avg_psnr:.2f}")
        print(f"  Average SSIM across all videos: {overall_avg_ssim:.4f}")
        print(f"  Average LPIPS across all videos: {overall_avg_lpips:.4f}") # Print overall LPIPS


if __name__ == "__main__":
    main()