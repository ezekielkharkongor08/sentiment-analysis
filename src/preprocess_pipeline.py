
import json
import os
import re
from pathlib import Path
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_FILE = str(BASE_DIR / "data" / "raw" / "master_combined_dataset.json")
DEFAULT_OUTPUT_DIR = str(BASE_DIR / "data" / "processed")

for resource in ['wordnet', 'omw-1.4', 'stopwords']:
    try:
        nltk.download(resource, quiet=True)
    except Exception as e:
        pass

lemmatizer = WordNetLemmatizer()

BASE_STOPWORDS = set(stopwords.words('english'))
NEGATION_WORDS = {'no', 'not', 'nor', 'neither', 'never', 'none', 'cannot', 'without'}
CUSTOM_STOPWORDS = BASE_STOPWORDS - NEGATION_WORDS
GOV_WHITELIST = {
    'aadhaar', 'digilocker', 'cpgrams', 'dbt', 'pmkisan', 'pm-kisan', 'trai',
    'mygov', 'parivahan', 'sarathi', 'swayam', 'evidya', 'pm-evidya', 'abha',
    'csc', 'panchayat', 'nic', 'darpg', 'meity', 'telecom', 'bharatnet'
}
CONTRACTIONS = {
    r"can\'t": "cannot",
    r"won\'t": "will not",
    r"n\'t": " not",
    r"i\'m": "i am",
    r"it\'s": "it is",
    r"they\'re": "they are",
    r"there\'s": "there is",
    r"we\'re": "we are",
    r"you\'re": "you are",
    r"what\'s": "what is",
    r"that\'s": "that is",
    r"who\'s": "who is",
    r"let\'s": "let us",
    r"\'ve": " have",
    r"\'ll": " will",
    r"\'d": " would"
}
def expand_contractions(text):
    """Expands short contractions into full words to preserve negations."""
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def remove_emojis_and_noise(text):
    """Strips Unicode emojis, web links, HTML artifacts, and unwanted symbols."""
    # URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    #html
    text = re.sub(r'<.*?>|&[a-z]+;', ' ', text)
    #emoji
    emoji_pattern = re.compile(
        "["
        "\U00010000-\U0010ffff"
        "\u2600-\u26ff"
        "\u2700-\u27bf"
        "\ufe00-\ufe0f"
        "\U0001f300-\U0001f5ff"
        "\U0001f600-\U0001f64f"
        "\U0001f680-\U0001f6ff"
        "\U0001f900-\U0001f9ff"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub(' ', text)

    text = re.sub(r'[^a-zA-Z0-9\s-]', ' ', text)
    return text

def normalize_elongated_words(word):
    """Reduces repetitive characters (e.g., 'plzzz' -> 'plz', 'baaaad' -> 'bad')."""
    if word in GOV_WHITELIST:
        return word
    return re.sub(r'(.)\1{2,}', r'\1', word)

def clean_and_lemmatize_text(raw_text):
    """
    Master pipeline: Contractions -> Noise/Emoji Removal -> Lowercasing ->
    Elongation Fix -> Stopwords Removal -> Lemmatization (consulting -> consult).
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return ""
    text = expand_contractions(raw_text)
    text = remove_emojis_and_noise(text)
    text = text.lower()
    words = text.split()
    processed_tokens = []
    for w in words:
        w_norm = normalize_elongated_words(w)
        if len(w_norm) > 1 and (w_norm not in CUSTOM_STOPWORDS or w_norm in GOV_WHITELIST):
            lemma = lemmatizer.lemmatize(w_norm, pos='v')
            lemma = lemmatizer.lemmatize(lemma, pos='n')
            processed_tokens.append(lemma)

    return " ".join(processed_tokens)

def run_preprocessing_pipeline(
    input_file=None,
    output_dir=None
):
    if input_file is None:
        input_file = DEFAULT_INPUT_FILE
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("[*] STARTING DATA PREPROCESSING & NLP PIPELINE")
    print(f"[*] Input File: {input_file}")
    print("=" * 65 + "\n")

    if not os.path.exists(input_file):
        print(f"[-] Error: Input file '{input_file}' not found.")
        return None

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    initial_count = len(df)
    print(f"[+] Loaded {initial_count} raw records.")

    print("[*] Handling missing values...")
    df['state'] = df['state'].fillna('National / All-India')
    df['district'] = df['district'].fillna('Not Specified')
    df['likes'] = df['likes'].fillna(0).astype(int)
    df['category'] = df['category'].fillna('Public Feedback')
    df['status'] = df['status'].fillna('Submitted')
    df['user_name'] = df['user_name'].fillna('Anonymous Citizen')

    df.dropna(subset=['comment_text'], inplace=True)
    df = df[df['comment_text'].str.strip() != '']

    print("[*] Checking and removing duplicates...")
    df.drop_duplicates(subset=['comment_text'], keep='first', inplace=True)
    dedup_count = len(df)
    print(f"    -> Removed {initial_count - dedup_count} duplicate records. Remaining: {dedup_count}")

    print("[*] Applying NLP normalization (Emoji removal, Lemmatization, Contractions)...")
    df['raw_comment'] = df['comment_text']  # Preserve original verbatim text
    df['cleaned_comment'] = df['raw_comment'].apply(clean_and_lemmatize_text)

    df = df[df['cleaned_comment'].str.strip() != '']

    df.reset_index(drop=True, inplace=True)
    df['id'] = range(1, len(df) + 1)

    standard_columns = [
        'id',
        'source_portal',
        'topic_or_ministry',
        'user_name',
        'raw_comment',
        'cleaned_comment',
        'date_posted',
        'likes',
        'state',
        'district',
        'category',
        'status'
    ]
    final_cols = [c for c in standard_columns if c in df.columns] + [c for c in df.columns if c not in standard_columns and c != 'comment_text']
    df = df[final_cols]

    json_output = os.path.join(output_dir, "preprocessed_comments.json")
    csv_output = os.path.join(output_dir, "preprocessed_comments.csv")

    df.to_json(json_output, orient="records", indent=2, force_ascii=False)
    df.to_csv(csv_output, index=False, encoding="utf-8")

    print("\n" + "=" * 65)
    print("[OK] PREPROCESSING COMPLETED SUCCESSFULLY!")
    print(f"[OK] Total Cleaned Records : {len(df)}")
    print(f"[OK] JSON Output File      : {json_output}")
    print(f"[OK] CSV Output File       : {csv_output} (Ready for Power BI)")
    print("=" * 65)

    return df

if __name__ == "__main__":
    run_preprocessing_pipeline()
