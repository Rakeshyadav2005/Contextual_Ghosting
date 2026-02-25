import nltk
from textblob import TextBlob
from nltk.corpus import wordnet
import random

# Download all necessary NLTK data to ensure cross-platform compatibility
# These cover tokenization, part-of-speech tagging, and synonym lookup
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)                 # Required for modern tokenization
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True) # Modern English POS tagger
nltk.download('omw-1.4', quiet=True)

class TextGhoster:
    def __init__(self):
        pass

    def get_synonym(self, word, tag):
        synonyms = set()
        # Map TextBlob tags (Penn Treebank) to WordNet categories
        # JJ = Adjective, NN = Noun, VB = Verb, RB = Adverb
        tag_map = {"JJ": wordnet.ADJ, "NN": wordnet.NOUN, "VB": wordnet.VERB, "RB": wordnet.ADV}
        wn_tag = tag_map.get(tag[:2])
        
        if not wn_tag:
            return word

        for syn in wordnet.synsets(word, pos=wn_tag):
            for lemma in syn.lemmas():
                # WordNet uses underscores for multi-word lemmas
                synonyms.add(lemma.name().replace('_', ' '))
        
        # Filter out the original word to ensure actual shuffling occurs
        valid_syns = [s for s in synonyms if s.lower() != word.lower()]
        return random.choice(valid_syns) if valid_syns else word

    def shuffle_text(self, text):
        if not text.strip():
            return ""
            
        blob = TextBlob(text)
        ghosted_words = []
        
        # Part-of-Speech tagging helps us replace words with the same grammatical role
        for word, tag in blob.tags:
            # We target Adjectives (JJ) and Verbs (VB) as they are stylometric signatures
            if tag.startswith(('JJ', 'VB')):
                ghosted_words.append(self.get_synonym(word, tag))
            else:
                ghosted_words.append(word)
        
        return " ".join(ghosted_words)