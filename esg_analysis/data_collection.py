from __future__ import annotations

import logging
import pandas as pd
from gnews import GNews
from gnews.utils.constants import BASE_URL
from tqdm import tqdm
import snscrape.modules.twitter as sntwitter


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
        keys: list[str] = [],
        keys_exact_match: list[str] = [],
    ):
        if not keys and not keys_exact_match:
            raise ValueError("One of keys or keys_exact_match is required")

        logging.info("Getting metadata for articles.")
        all_keys = []
        all_keys.extend(keys)
        search_term = "%20".join(keys)
        url = BASE_URL + "/search?q={}".format(search_term) + self._ceid()
        logging.info(f"Search URL: {url}")

        news = pd.DataFrame(self._get_news(url))
        news = self._get_news_article_texts(news, keys_exact_match)
        if len(news.columns) == 0:
            return news
        else:
            return pd.DataFrame(news)

    def _get_news_article_texts(
        self, news: pd.DataFrame, keys_exact_match: list[str] = []
    ):
        logger.info("Getting articles from URLS.")
        texts = []

        for url in tqdm(news["url"]):
            try:
                texts.append(self.get_full_article(url).text)
            except Exception as err:
                logger.error(err.args[0])
                texts.append(pd.NA)

        news["text"] = texts
        news = news.dropna()
        valid_rows = []
        for _, row in news.iterrows():
            is_valid = all(
                [
                    True if key in row["text"].lower() else False
                    for key in keys_exact_match
                ]
            )
            if is_valid:
                valid_rows.append(row)

        filtered_news = pd.DataFrame(valid_rows)
        if len(filtered_news.columns) == 0:
            return filtered_news
        else:
            filtered_news.columns = news.columns
            return filtered_news


def collect_tweets(key: str, n_tweets: int = 1000):
    texts = []
    for n, tweet in enumerate(
        sntwitter.TwitterSearchScraper(key + ' lang:"en"').get_items()
    ):
        if n == n_tweets:
            break
        texts.append(tweet.content)

    return texts


def get_tweets(keyword):
    locs = []
    contents = []
    for i, tweet in enumerate(
        sntwitter.TwitterSearchScraper(keyword + ' lang:"en"').get_items()
    ):  #  + ' since:2022-01-01 until:2022-01-10, lang:"en"'
        if i >= 100:
            break
        contents.append(tweet.content)
        locs.append(tweet.user.location)

    return contents, locs


if __name__ == "__main__":

    google_news = ESGNews(language="en", country="US", period=None, max_results=30)

    news = google_news.get_news_multi_keywords(
        keys_exact_match=["tesla"],
    )
