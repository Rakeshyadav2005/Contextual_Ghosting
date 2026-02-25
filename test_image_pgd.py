from src.image_engine import ImageGhoster
import os

engine = ImageGhoster()
input_file = "data/raw/input.jpg"
output_file = "data/processed/ghosted_pgd.jpg"

if os.path.exists(input_file):
    print("Starting PGD Protection (Option B)...")
    # alpha should be smaller than epsilon!
    # iters is usually 10-20 for a good balance.
    protected = engine.apply_pgd_protection(input_file, epsilon=0.01, alpha=0.002, iters=10)
    protected.save(output_file)
    print(f"Success! Ghosted image saved to {output_file}")
    print("Check the image: It should look much cleaner than the FGSM version.")
else:
    print("Error: Please put an image in data/raw/input.jpg first.")