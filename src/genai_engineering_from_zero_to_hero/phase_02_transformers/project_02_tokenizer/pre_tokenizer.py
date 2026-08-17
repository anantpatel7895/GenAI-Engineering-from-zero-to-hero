import re


class BasicPreTokenizer:
    _pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def pre_tokenize(self, text: str) -> list[str]:
        return self._pattern.findall(text)