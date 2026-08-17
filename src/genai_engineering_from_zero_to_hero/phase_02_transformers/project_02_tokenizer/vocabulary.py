class Vocabulary:
    def __init__(self, tokens: list[str]):
        self.token_to_id = {
            token: idx
            for idx, token in enumerate(tokens)
        }

        self.id_to_token = {
            idx: token
            for token, idx in self.token_to_id.items()
        }

    def token_to_id_lookup(self, token: str) -> int:
        return self.token_to_id[token]

    def id_to_token_lookup(self, token_id: int) -> str:
        return self.id_to_token[token_id]

    def __len__(self) -> int:
        return len(self.token_to_id)