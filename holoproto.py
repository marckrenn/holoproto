import cv2
import numpy as np
import os

def main():
    # Load the depth and RGB images (depth as grayscale)
    depth_img = cv2.imread('input/depth.webp', cv2.IMREAD_GRAYSCALE)
    rgb_img = cv2.imread('input/rgb.jpg')
    
    if depth_img is None or rgb_img is None:
        print("Error: Could not load one or both images")
        return

    # Print image dimensions for debugging
    print(f"RGB image shape: {rgb_img.shape}")
    print(f"Depth image shape: {depth_img.shape}")

    # Posterize depth image to 12 levels
    levels = 12
    level_size = 256 / levels
    depth_img = np.uint8(np.floor(depth_img / level_size) * level_size)

    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)

    # Save original RGB image as layer0
    cv2.imwrite('output/layer0.png', rgb_img)

    # Create and save masked layers
    for level in range(levels):
        # Calculate the pixel value for this level
        level_value = int(level * level_size)
        
        # Create single-channel binary mask
        mask = np.zeros(depth_img.shape[:2], dtype=np.uint8)
        mask[depth_img == level_value] = 255
        
        # Create white background
        white_bg = np.full_like(rgb_img, 255)
        
        # Apply mask to RGB image and combine with white background
        masked_rgb = cv2.bitwise_and(rgb_img, rgb_img, mask=mask)
        masked_rgb_with_white = cv2.bitwise_and(white_bg, white_bg, mask=~mask) + masked_rgb
        
        # Save the masked image
        output_path = f'output/layer{level+1}.png'
        cv2.imwrite(output_path, masked_rgb_with_white)

    # For debugging, you can display the images
    cv2.imshow('Depth Image', depth_img)
    cv2.imshow('RGB Image', rgb_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__ == "__main__":
    main()
