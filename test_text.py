from src.text_engine import TextGhoster

def run_test():
    # Initialize the engine
    print("--- Initializing TextGhoster ---")
    engine = TextGhoster()

    # Sample sentences to test different POS tags
    samples = [
        "The quick brown fox jumps over the lazy dog.",
        "The research project is very successful and the code is performing well.",
        "Artificial intelligence is changing the world rapidly."
    ]

    print("\n--- Starting Stylometric Shuffling Test ---")
    for i, original in enumerate(samples, 1):
        print(f"\nTest {i}:")
        print(f"Original: {original}")
        
        # Apply the ghosting
        ghosted = engine.shuffle_text(original)
        print(f"Ghosted : {ghosted}")

if __name__ == "__main__":
    run_test()