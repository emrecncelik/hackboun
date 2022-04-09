from __future__ import annotations
from gnews import GNews
from gnews.utils.constants import BASE_URL


class ESGNews(GNews):
    def __init__(
        self,
        language="en",
        country="US",
        max_results=100,
        period=None,
        exclude_websites=None,
        proxy=None,
    ):
        super().__init__(
            language, country, max_results, period, exclude_websites, proxy
        )

    def get_news_multi_keywords(
        self,
        keys: list[str],
        keys_exact_match: list[str],
    ):
        all_keys = []
        keys_exact_match = ['"' + key + '"' for key in keys_exact_match]
        all_keys.extend(keys)
        all_keys.extend(keys_exact_match)

        search_term = "%20".join(all_keys)
        url = BASE_URL + "/search?q={}".format(search_term) + self._ceid()
        return self._get_news(url)
