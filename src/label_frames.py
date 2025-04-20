# Extract and label frames from video for training your own classifier

import cv2
import os
import sys
import glob

#List of Classes
classes = ["reading", "coding", "piano_practice", "eating", "1on1"]

video_path = "./data/training_videos/webcamtalking.mp4"
output_dir = "./output/training_output"
label = "1on1"  # change per run

# Check if video file exists
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
    print("Current working directory:", os.getcwd())
    sys.exit(1)

# Create output directory
try:
    os.makedirs(os.path.join(output_dir, label), exist_ok=True)
    print(f"Created output directory: {os.path.join(output_dir, label)}")
except Exception as e:
    print(f"Error creating output directory: {e}")
    sys.exit(1)

# Find the highest existing frame number
existing_frames = glob.glob(os.path.join(output_dir, label, "frame_*.jpg"))
if existing_frames:
    # Extract numbers from filenames and find the maximum
    frame_numbers = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in existing_frames]
    start_count = max(frame_numbers) + 1
    print(f"Found {len(existing_frames)} existing frames. Starting from frame_{start_count}")
else:
    start_count = 0
    print("No existing frames found. Starting from frame_0")

# Open video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    sys.exit(1)

# Get video properties
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video properties:")
print(f"- Total frames: {total_frames}")
print(f"- FPS: {fps}")
print(f"- Duration: {total_frames/fps:.2f} seconds")

frame_interval = 10
frame_count = 0
saved_count = start_count

print("\nStarting frame extraction...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        filename = f"{output_dir}/{label}/frame_{saved_count}.jpg"
        try:
            success = cv2.imwrite(filename, frame)
            if success:
                saved_count += 1
                if saved_count % 10 == 0:  # Print progress every 10 frames
                    print(f"Saved {saved_count} frames...")
            else:
                print(f"Warning: Failed to save frame {saved_count}")
        except Exception as e:
            print(f"Error saving frame {saved_count}: {e}")

    frame_count += 1

cap.release()
print(f"\nExtraction complete!")
print(f"Processed {frame_count} frames")
print(f"Added {saved_count - start_count} new frames to {output_dir}/{label}")
print(f"Total frames in directory: {saved_count}")