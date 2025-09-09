'''
This code is a simple Python script that uses OpenCV to split a video file into individual frames 
and save them as JPG images.
It takes two command-line arguments: 
the input video file 
and 
the output directory where the frames will be saved. 
The script reads each frame from the video and saves it as a JPG file with a 
sequential filename format (e.g., frame_0000.jpg, frame_0001.jpg, etc.). 
It also prints the filename of each saved frame and the total number of frames processed at the end.

example:
python3 movieTOframes.py <input_video_file> <output_directory>

'''

import cv2
import os
import argparse


def split_movie(input_file, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    video = cv2.VideoCapture(input_file)
    if not video.isOpened():
        print(f"Error: Unable to open video file {input_file}")
        return

    count = 0
    ok, frame = video.read()
    while ok:
        # Save each frame as a JPG file
        frame_filename = os.path.join(output_dir, f"frame_{count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"Written frame: {frame_filename}")
        count += 1
        ok, frame = video.read()

    video.release()
    print(f"Finished splitting video. Total frames: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a video into individual JPG frames.")
    parser.add_argument("input_file", type=str, help="Path to the input video file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the extracted frames.")
    args = parser.parse_args()

    split_movie(args.input_file, args.output_dir)