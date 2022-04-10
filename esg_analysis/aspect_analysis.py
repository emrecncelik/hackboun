from __future__ import annotations

import torch
import logging
import numpy as np
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter
from data_collection import ESGNews

logger = logging.getLogger(__name__)

ASPECTS = {
    "gender_equality": {
        "keys": ["gender", "equality"],
        "keys_exact_match": ["gender", "equality"],
    },
    "product_safety": {
        "keys": ["product", "safety", "accidents"],
        "keys_exact_match": ["safety"],
    },
    "environmental_impact": {
        "keys": ["enviromental", "impact"],
        "keys_exact_match": ["environment"],
    },
    "data_privacy": {
        "keys": ["data", "privacy"],
        "keys_exact_match": ["privacy", "data"],
    },
    "cultural_diversity": {
        "keys": ["cultural", "diversity"],
        "keys_exact_match": ["diversity"],
    },
}


class ESGAnalyzer:
    def __init__(
        self,
        sentiment_model_name: str = "finiteautomata/bertweet-base-sentiment-analysis",
        label_mapping: list[str] = ["negative", "neutral", "positive"],
        device: str = None,
    ) -> None:
        self.sentiment_model_name = sentiment_model_name
        self.label_mapping = label_mapping
        self.device = device if device else self._auto_device()
        self._init_model()

    def _auto_device(self):
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def _init_model(self):
        logger.info(f"Loading model {self.sentiment_model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "finiteautomata/bertweet-base-sentiment-analysis"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "finiteautomata/bertweet-base-sentiment-analysis"
        ).to(self.device)

    def analyze(self, news_texts: list[str]):
        logger.info("Predicting overall labels for news articles.")
        docs_sent_tokenized = [sent_tokenize(text) for text in news_texts]
        doc_labels = []
        for sents in tqdm(docs_sent_tokenized):
            with torch.no_grad():
                features = self.tokenizer(
                    sents, padding=True, truncation=True, return_tensors="pt"
                ).to(self.device)
                logits = self.model(**features).logits
                logits = logits.cpu().detach().numpy()
                logits[:, 1] = -1000
                doc_label = self.label_mapping[np.argmax(logits.sum(axis=0))]
                doc_labels.append(doc_label)

        counter = Counter(doc_labels)
        print(counter)
        return counter.most_common()


def get_aspect_analysis_for_company(
    company_name: str, analyzer: ESGAnalyzer = None, n_news: int = 25
):
    results = {}
    if not analyzer:
        analyzer = ESGAnalyzer()

    for aspect in ASPECTS:
        logging.info(f"Analyzing aspect {aspect}")
        google_news = ESGNews(
            language="en", country="US", period=None, max_results=n_news
        )
        news = google_news.get_news_multi_keywords(
            keys=ASPECTS[aspect]["keys"] + company_name.split(),
            keys_exact_match=ASPECTS[aspect]["keys_exact_match"] + [company_name],
        )

        if len(news.columns) == 0:
            results[aspect] = []
        else:
            labels = analyzer.analyze(news_texts=news["text"])
            results[aspect] = labels

    return results


if __name__ == "__main__":
    import json

    results = {
        "general electric": None,
        "ford": None,
        "tesla": None,
        # "mercedes": None,
        # "amazon": None,
        # "facebook": None
    }
    analyzer = ESGAnalyzer(device="cpu")
    for company in results.keys():
        torch.cuda.empty_cache()
        results[company] = get_aspect_analysis_for_company(
            company, analyzer=analyzer, n_news=50
        )
    print(results)
    with open("output.json", "w") as f:
        json.dump(results, f, indent=4)
