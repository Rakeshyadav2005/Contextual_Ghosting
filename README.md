👻 Contextual Ghosting: Hybrid Adversarial Privacy Framework:

    Contextual Ghosting is a dual-engine privacy framework designed to protect digital identity in the age of AI scraping. It provides a robust defense against facial recognition and stylometric authorship attribution by using adversarial machine learning and NLP.

🛠️ Project Architecture:

    ->  Image Protection Engine: Implements the Projected Gradient Descent (PGD) algorithm ($L_\infty$ bounded). It injects non-perceivable adversarial noise into images, causing state-of-the-art classifiers (like ResNet50) to misidentify or "ghost" the subject.

    ->  Text Protection Engine: Uses Part-of-Speech (POS) Tagging and WordNet-based Synonym Shuffling to modify the stylistic fingerprint of a text block while maintaining its semantic meaning.

📊 Phase 3: System Performance:
    We evaluated the system for efficiency and effectiveness. The following metrics were achieved during the Phase 3 audit:
    |--------------------------------------------------------------------------------------------|
    |Metric	                |     Result	    |    Interpretation                              |
    |-----------------------|-------------------|------------------------------------------------|
    |Image Latency	        |     1.92s	        |    Optimized for real-time web deployment.     |
    |Text Latency	        |     2.85s	        |    High-speed NLP processing for large blocks. |
    |Linguistic Perturbation|	 38.5% - 57.1%	|    Successfully masks authorship signatures.   |
    |Defense Robustness	    |     PGD (L∞​)	     |    Resistant to simple denoising filters.      |
    ----------------------------------------------------------------------------------------------

📁 Repository Structure:

    Contextual_Ghosting/
    ├── src/                # Core Logic (Engines)
    ├── data/               # Raw and Processed testing samples
    ├── models/             # Local model weight storage (Offline mode)
    ├── notebooks/          # R&D experiments and PGD math validation
    ├── app.py              # Streamlit Cyber-Security Dashboard
    └── evaluate.py         # Performance testing script

🚀 Getting Started:
    1. Prerequisites
    Ensure you have Conda or Python installed.
    Command:
        git clone https://github.com/Rakeshyadav2005/Contextual_Ghosting.git
        cd Contextual_Ghosting

    2. Environment Setup
    Command:
        pip install -r requirements.txt
        python -m textblob.download_corpora
    
    3. Running the App
    Command:
        streamlit run app.py
    

🛡️ Mathematical Foundation:
    The image defense relies on the PGD update rule:
        ![alt text](image.png)
    This ensures the "noise" added to your photos is mathematically optimized to confuse AI while remaining invisible to the human eye. 