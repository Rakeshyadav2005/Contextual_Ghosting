from src.image_engine import ImageGhoster
import os

ghoster = ImageGhoster()
input_img = "data/raw/input.jpg"
output_img = "data/processed/ghosted.jpg"

if os.path.exists(input_img):
    protected = ghoster.apply_protection(input_img)
    protected.save(output_img)
    print(f"Success! Ghosted image saved to {output_img}")
else:
    print(f"Please add a sample image to {input_img}")