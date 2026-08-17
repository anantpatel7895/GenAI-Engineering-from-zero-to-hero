# Project 2 — Build a Tokenizer

| #    | Topic                           | Status |
| ---- | ------------------------------- | ------ |
| 2.1  | Why Tokenization Exists         | ⬜      |
| 2.2  | Characters vs Words vs Subwords | ⬜      |
| 2.3  | Vocabulary                      | ⬜      |
| 2.4  | Token → ID Mapping              | ⬜      |
| 2.5  | Special Tokens                  | ⬜      |
| 2.6  | Encoding                        | ⬜      |
| 2.7  | Decoding                        | ⬜      |
| 2.8  | Unknown Tokens                  | ⬜      |
| 2.9  | BPE Intuition                   | ⬜      |
| 2.10 | BPE Training Algorithm          | ⬜      |
| 2.11 | BPE Merge Rules                 | ⬜      |
| 2.12 | Mini BPE Tokenizer              | ⬜      |
| 2.13 | Testing                         | ⬜      |
| 2.14 | Production Tokenizers           | ⬜      |
| 2.15 | Performance & Trade-offs        | ⬜      |



| Step | Problem we're solving      | Modern LLM equivalent                                 |
| ---- | -------------------------- | ----------------------------------------------------- |
| 1    | Read training corpus       | Billions of documents from web, books, code           |
| 2    | Normalize text             | Unicode normalization, lowercasing (model-dependent)  |
| 3    | Pre-tokenize               | Regex, whitespace, punctuation, byte splitting        |
| 4    | Convert to initial symbols | Characters (educational) vs UTF-8 bytes (GPT-2/Llama) |
| 5    | Learn BPE merges           | Offline tokenizer training                            |
| 6    | Build vocabulary           | `vocab.json` / tokenizer vocabulary                   |
| 7    | Save tokenizer             | `tokenizer.json`, `merges.txt`, `tokenizer.model`     |
| 8    | Load tokenizer             | Inference server startup (vLLM, TGI, Ollama)          |
| 9    | Encode new text            | Every API request to the model                        |

