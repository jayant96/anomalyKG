import spacy

# You can directly load the installed model using its name
model_name = "en_ner_bc5cdr_md"  # Model name for version 3.1.0
try:
    nlp = spacy.load(model_name)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
