# pdf_processor_logic.py

import fitz 
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import torch
import warnings
warnings.filterwarnings("ignore")

SUMMARIZER_MODEL_NAME = "sshleifer/distilbart-cnn-12-6" # Lite Model cho Tóm tắt
KEYBERT_MODEL_NAME = 'sentence-transformers/paraphrase-MiniLM-L3-v2' # Lite Model cho KeyBERT
MAX_SUMMARY_LENGTH = 150
MAX_ARTICLE_LENGTH = 1024 

global summarizer_pipeline, kw_model
summarizer_pipeline = None
kw_model = None


def initialize_nlp_models():
    global summarizer_pipeline, kw_model
    
    device = -1 # Force CPU for stability
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL_NAME)
        summarizer_pipeline = pipeline(
            "summarization", 
            model=model, 
            tokenizer=tokenizer, 
            device=device
        )
        print(f"Đã tải mô hình Tóm tắt: {SUMMARIZER_MODEL_NAME}")
    except Exception as e:
        print(f"LỖI TẢI MÔ HÌNH TÓM TẮT: {e}")
        
    # Khởi tạo mô hình Trích xuất Từ khóa
    try:
        kw_model = KeyBERT(KEYBERT_MODEL_NAME) 
        print(f"Đã tải mô hình KeyBERT: {KEYBERT_MODEL_NAME}")
    except Exception as e:
        print(f"LỖI TẢI MÔ HÌNH KEYBERT: {e}")


def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        text = ' '.join(text.split())
        return text
    except Exception as e:
        print(f"LỖI trích xuất PDF: {e}")
        return None


def process_single_article(pdf_path):
    
    if summarizer_pipeline is None or kw_model is None:
        initialize_nlp_models() # Đảm bảo mô hình được tải
        if summarizer_pipeline is None or kw_model is None:
            return None, "Lỗi: Không thể tải mô hình NLP."

    article_text = extract_text_from_pdf(pdf_path)
    if not article_text or len(article_text) < 100:
        return None, "Lỗi: Không thể trích xuất văn bản hoặc nội dung quá ngắn."
    

    tokenizer = summarizer_pipeline.tokenizer
    inputs = tokenizer(article_text, return_tensors='pt', truncation=True, max_length=MAX_ARTICLE_LENGTH)
    article_truncated = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    
    summary = summarizer_pipeline(
        article_truncated,
        max_length=MAX_SUMMARY_LENGTH,
        min_length=30,
        do_sample=False
    )[0]['summary_text']
    
    keywords = kw_model.extract_keywords(
        article_truncated, 
        keyphrase_ngram_range=(1, 3), 
        stop_words='english',
        use_mmr=True, 
        diversity=0.5,
        top_n=7 
    )
    keywords_only = [k[0] for k in keywords]
    
    return {
        "summary": summary,
        "keywords": ", ".join(keywords_only),
        "filename": os.path.basename(pdf_path)
    }, None