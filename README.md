# GenAI-Engineering-from-zero-to-hero

## How to run perticular file

```bash
uv run python -m genai_engineering_from_zero_to_hero.phase_02_transformers.project_01_mathematics.01_tensor_memory
```


## UV Command

#### 1.
```bash
uv add --dev pytest --system-certs
```


## Find command

#### 1. find all files in the current directory and its subdirectories, excluding the `.git` directory:
```bash
find . -path "./.git" -prune -o -type f -print
```

#### 2. find all files in the current directory and its subdirectories, excluding the `.git` and `.venv` directories:
```bash
find . \( -path "./.git" -o -path "./.venv" \) -prune -o -type f -print
```

#### 3. find all files in the current directory and its subdirectories, excluding the `.git` and `.venv` directories, and print their relative paths:
```bash
find . \( -path "./.git" -o -path "./.venv" \) -prune -o -print
```


# To run the debug scripts

```bash
uv run python debug/debug_tokenizer.py
```
