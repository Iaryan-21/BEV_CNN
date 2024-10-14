import cv2
import os
import numpy as np

def create_video_from_images(image_folder, output_file, fps=30):
    try:
        # Get the list of images from the folder and filter for valid image files
        images = [img for img in os.listdir(image_folder) if img.endswith((".jpg", ".png", ".jpeg"))]
        if not images:
            raise ValueError(f"No images found in {image_folder}")
        
        # Sort images to maintain order
        images.sort()
        print(f"Found {len(images)} images")
        
        # Read the first image to get dimensions and ensure it's not black
        first_image_path = os.path.join(image_folder, images[0])
        frame = cv2.imread(first_image_path)
        if frame is None:
            raise ValueError(f"Unable to read image: {first_image_path}")
        
        # Convert frame to 3-channel if it has an alpha channel (PNG with transparency)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        height, width, layers = frame.shape
        print(f"Image dimensions: {width}x{height}")
        
        # Display pixel values of the first image to check if it's black
        print(f"First pixel values of the first image: {frame[0,0]}")
        
        # Define the codec and create VideoWriter object (try MJPG for better compatibility)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        if not video.isOpened():
            raise IOError("VideoWriter failed to open")
        
        # Iterate through all images
        for i, image in enumerate(images):
            img_path = os.path.join(image_folder, image)
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Warning: Unable to read image {img_path}")
                continue
            
            # Check if the image has an alpha channel and convert if needed
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Check if the frame is all black
            if np.sum(frame) == 0:
                print(f"Warning: Image {img_path} appears to be all black")
            
            # Display pixel values for every 10th image
            if i % 10 == 0:
                print(f"First pixel values of image {i}: {frame[0,0]}")
            
            # Ensure that all images are of the same size
            if frame.shape[0] != height or frame.shape[1] != width:
                print(f"Warning: Image {img_path} has different dimensions ({frame.shape[1]}x{frame.shape[0]}), resizing it to {width}x{height}")
                frame = cv2.resize(frame, (width, height))

            # Write the frame to the video
            video.write(frame)

        video.release()
        print(f"Video created successfully: {output_file}")
        
        # Check the size of the output file
        file_size = os.path.getsize(output_file)
        print(f"Output file size: {file_size} bytes")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Usage
image_folder = '/home/amishr17/Desktop/BEV_CNN/src/opimgs'
output_file = 'output_video.mp4'
create_video_from_images(image_folder, output_file)
