from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import re

app = Flask(__name__)
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

def preprocess_text(text):
    """
    Ultra-robust preprocessing for maximum similarity and search recall
    """
    # Convert to string and strip
    text = str(text).strip()
    if not text:
        return ""
    
    # Convert to lowercase for better matching
    text = text.lower()
    
    # Normalize whitespace (multiple spaces/tabs/newlines → single space)
    text = re.sub(r'\s+', ' ', text)
    
    # Expand contractions for semantic consistency
    contractions = {
        "won't": "will not", "can't": "cannot", "don't": "do not",
        "shouldn't": "should not", "wouldn't": "would not", "couldn't": "could not",
        "mustn't": "must not", "needn't": "need not", "daren't": "dare not",
        "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
        "doesn't": "does not", "didn't": "did not",
        "i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are",
        "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
        "i'll": "i will", "you'll": "you will", "we'll": "we will", "they'll": "they will",
        "i'd": "i would", "you'd": "you would", "we'd": "we would", "they'd": "they would",
        "let's": "let us", "that's": "that is", "who's": "who is", "what's": "what is",
        "where's": "where is", "when's": "when is", "why's": "why is", "how's": "how is",
        "there's": "there is", "here's": "here is", "it's": "it is"
    }
    
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Normalize phone numbers for consistent matching
    # +91 9876543210 → +919876543210
    # +91-9876-543210 → +919876543210
    # 91 9876543210 → +919876543210
    text = re.sub(r'(\+?91)[\s\-]?(\d{10})', r'+91\2', text)
    text = re.sub(r'(\+\d{1,3})[\s\-]+(\d)', r'\1\2', text)
    
    # Normalize common order/tracking references
    text = re.sub(r'order[\s#]*(\d+)', r'order \1', text)
    text = re.sub(r'tracking[\s#]*(\d+)', r'tracking \1', text)
    text = re.sub(r'invoice[\s#]*(\d+)', r'invoice \1', text)
    
    # Replace URLs with generic token for better grouping
    text = re.sub(r'https?://\S+', '[url]', text)
    text = re.sub(r'www\.\S+', '[url]', text)
    
    # Replace email addresses with generic token
    text = re.sub(r'\S+@\S+\.\S+', '[email]', text)
    
    # Normalize punctuation (remove excessive, keep sentence structure)
    text = re.sub(r'([.!?]){2,}', r'\1', text)
    text = re.sub(r'[^\w\s\+\@\.\!\?\,\-]', ' ', text)  # Remove special chars except essential ones
    
    # Common typos and abbreviations for better matching
    typo_fixes = {
        'plz': 'please', 'pls': 'please', 'thx': 'thanks', 'thnx': 'thanks',
        'u': 'you', 'ur': 'your', 'im': 'i am', 'msg': 'message',
        'asap': 'as soon as possible', 'fyi': 'for your information',
        'eta': 'estimated time', 'cod': 'cash on delivery',
        'refund': 'refund', 'cancelled': 'canceled', 'colour': 'color',
        'grey': 'gray', 'realise': 'realize', 'centre': 'center'
    }
    
    # Apply typo fixes (word boundaries to avoid partial matches)
    for typo, fix in typo_fixes.items():
        text = re.sub(r'\b' + typo + r'\b', fix, text)
    
    # Normalize product/item references
    text = re.sub(r'\b(this|that)\s+(item|product|thing)\b', 'product', text)
    text = re.sub(r'\bpcs?\b', 'pieces', text)
    text = re.sub(r'\bkg\b', 'kilogram', text)
    text = re.sub(r'\blb\b', 'pound', text)
    
    # Normalize time references for better context matching
    text = re.sub(r'\b(yesterday|1 day ago)\b', 'yesterday', text)
    text = re.sub(r'\b(last week|1 week ago)\b', 'last week', text)
    text = re.sub(r'\b(last month|1 month ago)\b', 'last month', text)
    
    # Clean up extra spaces after all processing
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Truncate at word boundary if too long
    if len(text) > 400:
        text = text[:400].rsplit(' ', 1)[0] + '...'
    
    return text

@app.route('/embed', methods=['POST'])
def embed():
    data = request.get_json()
    texts = data.get('texts', [])
    if not texts:
        return jsonify({"error": "No texts"}), 400

    # Preprocess all texts
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Filter out empty texts after preprocessing
    valid_texts = [text for text in processed_texts if text]
    if not valid_texts:
        return jsonify({"error": "No valid texts after preprocessing"}), 400

    vectors = model.encode(valid_texts).tolist()

    return jsonify({"embeddings": vectors})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100)