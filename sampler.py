import pandas as pd
import re
import random
import numpy as np
from trafilatura import extract
import nltk
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
#import matplotlib.pyplot as plt
#import seaborn as sns
from sentence_transformers import SentenceTransformer
#from datetime import datetime
from scipy import stats
import math
from bs4 import BeautifulSoup
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

tqdm.pandas()
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


class DomainSampler():
    # downloading the embedding model
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cuda:0")

    # getting the stop words to be used later
    global_stop_words_set = set()
    for lang in stopwords.fileids():
        global_stop_words_set.update(stopwords.words(lang))

    def __init__(self, pop_size:int, confidence:float = 0.95, margin_error:float = 0.05):
        self.pop_size = pop_size
        self.confidence = confidence
        self.margin_error = margin_error
    
    def get_min_sample_size(self)->int:
        """
        Calculates minimum sample size for a finite population.

        :param pop_size: Total number of individuals in the population
        :param confidence: Confidence level (e.g., 0.95 for 95%)
        :param margin_error: Acceptable margin of error (e.g., 0.05 for 5%)
        """
        # 1. Get Z-score based on confidence level
        # Use 1 - (1 - confidence) / 2 to get the two-tailed critical value
        z = stats.norm.ppf(1 - (1 - self.confidence) / 2)

        # 2. Set estimated proportion (0.5 provides the safest/maximum sample size)
        p = 0.5

        # 3. Cochran's formula for infinite population
        n_0 = (z**2 * p * (1 - p)) / (self.margin_error**2)

        # 4. Adjust for finite population (Finite Population Correction)
        n = n_0 / (1 + ((n_0 - 1) / self.pop_size))

        return math.ceil(n)
    
    def clean_text(self, text: str, stop_words = global_stop_words_set):
        ''' remove stop words, urls and numbers'''
        if not isinstance(text, str):
            # raise TypeError("text must be a string")
            return ""

        # remove urls
        text = re.sub(r"http\S+", " link ", text)

        # replace any digit with 'number'
        text = re.sub(r"\d+", " number ", text)

        # set space before and after any punctuation
        text = re.sub(r"([^\w\s])", r" \1 ", text)

        # remove extra spaces
        text = re.sub(r"\s+", " ", text)
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in stop_words]
        text = " ".join([ w for w in filtered_tokens if len(w) > 1 ] )
        return text.lower().strip()
    
    def preprocess_html(self, articles:list[str]):
        articles_text = []
        for response_html in tqdm(articles):
            try:
                result = json.loads(extract(response_html, with_metadata=True, output_format="json", include_comments=False,
                                            include_tables=False))
                articles_text.append({"title":result["title"],
                        "author":result["author"],
                        "domain_name":result["hostname"],
                        "date":result["date"],
                        "categories": result["categories"],
                        "tags":result["tags"],
                        "text":self.clean_text(result["text"])})
            except:
                    soup = BeautifulSoup(response_html, "html.parser")
                    for tag in soup(["script", "style", "noscript"]):
                        tag.decompose()
                    articles_text.append(self.clean_text(soup.get_text(separator=" ", strip=True)))
    


