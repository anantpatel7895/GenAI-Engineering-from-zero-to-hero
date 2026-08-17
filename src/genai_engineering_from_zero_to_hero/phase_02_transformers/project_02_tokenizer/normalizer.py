class BasicNormalizer:
    def __init__(self, lowercase: bool = False):
        self.lowercase = lowercase

    def normalize(self, text: str) -> str:
        if self.lowercase:
            text = text.lower()

        return text