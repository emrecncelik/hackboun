from __future__ import annotations

import logging
import pandas as pd
from gnews import GNews
from gnews.utils.constants import BASE_URL
from tqdm import tqdm


logger = logging.getLogger(__name__)


class ESGNews(GNews):
    def __init__(
        self,
        language="en",
        country="US",
        max_results=100,
        period=None,
        exclude_websites=None,
        proxy=None,
        verbose: bool = True,
    ):
        super().__init__(
            language, country, max_results, period, exclude_websites, proxy
        )
        self.verbose = verbose

    def get_news_multi_keywords(
        self,
        keys: list[str],
        keys_exact_match: list[str],
        return_df: bool = True,
    ):
        logging.info("Getting metadata for articles.")
        all_keys = []
        keys_exact_match = ['"' + key + '"' for key in keys_exact_match]
        all_keys.extend(keys)
        all_keys.extend(keys_exact_match)

        search_term = "%20".join(all_keys)
        url = BASE_URL + "/search?q={}".format(search_term) + self._ceid()

        news = self._get_news(url)

        if return_df:
            return pd.DataFrame(news)
        else:
            return news

    def get_news_article_texts(self, urls: list[str]):
        logger.info("Getting articles from URLS.")
        articles = [self.get_full_article(url) for url in tqdm(urls)]
        logger.info("Extracting texts from articles.")
        texts = [article.text for article in tqdm(articles)]
        return texts
