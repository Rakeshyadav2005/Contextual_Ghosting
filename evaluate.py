import time
from src.image_engine import ImageGhoster
from src.text_engine import TextGhoster

def evaluate_system():
    img_engine = ImageGhoster()
    txt_engine = TextGhoster()
    
    # --- Test 1: Image Latency ---
    start = time.time()
    # Replace with a path to a real image in your data/raw folder
    img_engine.apply_pgd_protection("data/raw/ui_upload.jpg", epsilon=0.01)
    img_time = time.time() - start
    
    # --- Test 2: Text Variance ---
    sample_text = "The talented developer created a sophisticated application."
    start = time.time()
    ghosted_text = txt_engine.shuffle_text(sample_text)
    txt_time = time.time() - start
    
    # Calculate Change Percentage
    original_words = sample_text.split()
    ghosted_words = ghosted_text.split()
    changes = sum(1 for a, b in zip(original_words, ghosted_words) if a != b)
    change_pct = (changes / len(original_words)) * 100

    print("--- PHASE 3 EVALUATION REPORT ---")
    print(f"Image Processing Latency: {img_time:.2f} seconds")
    print(f"Text Processing Latency:  {txt_time:.2f} seconds")
    print(f"Linguistic Perturbation:  {change_pct:.1f}% of style-words modified")

if __name__ == "__main__":
    evaluate_system()