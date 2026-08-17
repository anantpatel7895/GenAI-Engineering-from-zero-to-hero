# Project 2 — Build a Tokenizer

## 2.1 — Why Does Tokenization Exist?

### Q1. Why do LLMs need tokenization?

Neural networks operate on numerical tensors rather than raw text.

Tokenization converts raw text into discrete tokens, and those tokens are mapped to integer IDs that can be passed through an embedding layer.

The basic pipeline is:

→ Raw Text
→ Tokens
→ Token IDs
→ Embeddings
→ Transformer

### Q2. Why can't a Transformer directly consume text?

A Transformer operates on numerical tensors. Text must therefore be converted into numerical representations before it can be processed.

### Q3. What are the main tokenization strategies?

The major approaches are:

- Character-level
- Word-level
- Subword-level
- Byte-level approaches

Each makes a different trade-off between `vocabulary size`, `sequence length`, and `ability to represent unseen text`(OUT OF VOCABULARY problem).

### Q4. Why is tokenization important for production LLM systems?

Tokenization affects sequence length.

More tokens can result in:

- More computation
- Higher latency
- Larger KV cache
- Greater memory usage
- Higher serving cost

Therefore tokenizer efficiency affects both `model behavior` and `inference economics`.

### Engineering Principle

A tokenizer provides the interface between human-readable language and the numerical representation consumed by a Transformer.

---

## What problem does the next step solve?

We know that text must be converted into tokens.

But we haven't answered the most important question:

**What should one token actually represent?**

Should it be:

- a character?
- a word?
- a subword?

Step 2.2 solves this problem.

# 2.2 — Characters vs Words vs Subwords

## Q1. What is character-level tokenization?

Character-level tokenization represents each character as an individual token.

Example:

"hello"
→ ["h", "e", "l", "l", "o"]

Advantages:

- Very small vocabulary
- Almost no unknown-token problem (there is no out of vocabulary problem)

Disadvantage:

- Very long sequences

---

## Q2. What is word-level tokenization?

Word-level tokenization represents each word as a token.

Example:

"I love Transformers"
→ ["I", "love", "Transformers"]

Advantages:

- Short sequences
- Simple representation

Disadvantages:

- Very large vocabulary
- Unknown-word problem or OOV (out of vocabulary problem)
- Poor handling of new words, technical terms, typos, and rare words

---

## The Fundamental Trade-Off

We now have:

```text
Character
    │
    ├── Small vocabulary
    └── Long sequences
```

versus:

```text
Word
    │
    ├── Short sequences
    └── Huge vocabulary / OOV problem
```

This is the fundamental tokenizer trade-off.

Neither end can improve one property without giving up the other — the two axes pull against each other:

```text
  Vocabulary Size
        ▲
        │  ● Word
        │   (short sequences, huge vocab, OOV)
        │
        │
        │           ● Subword
        │            (balanced)
        │
        │
        │                        ● Character
        │                         (tiny vocab, long sequences)
        └────────────────────────────────────────────►
                                        Sequence Length
```

Pushing the vocabulary down forces sequences to get longer.

Pushing sequences shorter forces the vocabulary to grow.

We want something in the middle.

---

## Q3. What is subword tokenization?

Subword tokenization represents text using reusable pieces smaller than or equal to words.

Example:

"playing"
→ ["play", "ing"]

The same pieces can be reused across many words.

---

## Q4. Why are subwords useful for LLMs?

Subwords provide a compromise between character-level and word-level tokenization.

They provide:

- Smaller vocabulary than word-level tokenization
- Shorter sequences than character-level tokenization
- Better handling of rare and unseen words
- Reusable linguistic pieces

---

## Q5. What problem does BPE solve?

Once we decide to use subwords, we need a method to determine which subword units should be included in the vocabulary.

BPE learns subword units by repeatedly merging frequently occurring token pairs.

---

## Q6. Why does tokenizer design affect LLM inference?

Tokenization determines sequence length.

More tokens generally mean:

- More prefill computation
- More attention computation
- Larger KV cache
- Higher latency
- Potentially higher serving cost

Therefore tokenizer efficiency affects production LLM performance.

---

## Production Perspective

Tokenization affects the entire LLM serving stack.

Suppose a request contains:

```text
10,000 characters
```

Tokenizer A produces:

```text
2,000 tokens
```

Tokenizer B produces:

```text
4,000 tokens
```

The model now has to process **twice as many token positions** with B.

This impacts:

```text
                    Token Count
                         │
            ┌────────────┼─────────────┐
            ▼            ▼             ▼
         Prefill      KV Cache      Latency
            │            │             │
            └────────────┼─────────────┘
                         ▼
                    GPU Cost
```

Note that the cost does not always scale linearly with the token count:

* **KV cache** grows **linearly** — 2× tokens means 2× cache memory per request, which directly reduces how many requests fit on a GPU concurrently.
* **Attention** grows **quadratically** in sequence length — 2× tokens can mean roughly 4× attention computation during prefill.

So a tokenizer that is 2× less efficient can cost considerably more than 2× to serve.

Therefore, tokenizer design is part of **LLM architecture and infrastructure**, not merely NLP preprocessing.

---

## Engineering Principle

Character tokenization optimizes vocabulary size.

Word tokenization optimizes sequence length.

Subword tokenization attempts to balance both.

Modern LLM tokenizers generally use subword- or byte-oriented approaches rather than simple word tokenization.

---

## What problem does the next step solve?

We have chosen subword tokenization.

But a tokenizer needs a finite collection of tokens and a mapping from each token to an integer.

Therefore the next problem is:

**How do we design and manage the tokenizer vocabulary?**

Step 2.3 solves this problem.

# 2.3 — Vocabulary

## Q1. What is a tokenizer vocabulary?

A vocabulary is a mapping between tokens and integer IDs.

Example:

| Token | ID |
|---|---:|
| `<PAD>` | 0 |
| `<UNK>` | 1 |
| `hello` | 2 |
| `world` | 3 |

Therefore:

"hello world"
→ ["hello", "world"]
→ [2, 3]

---

## Q2. Why does the model need token IDs?

The Transformer operates on numerical tensors.

The token ID acts as an index into the embedding matrix.

Example:

token:
"hello"

vocabulary:
"hello" → 2

embedding lookup:

embedding_matrix[2]
→ [0.12, -0.41, 0.83, ...]

The token ID is therefore an address into the model's learned embedding table.

---

## Q3. Does token ID contain semantic meaning?

No.

If:

hello → 10
world → 11

the number 11 does not mean that "world" is greater than "hello".

The ID is simply an index.

Semantic information is learned in the embedding and subsequent model parameters.

---

## Q4. How is vocabulary size related to the embedding matrix?

If:

vocab_size = V
embedding_dimension = D

then the input embedding matrix has shape:

(V, D)

Example:

vocab_size = 50,000
embedding_dimension = 4,096

Embedding matrix:

(50,000, 4,096)

Number of parameters:

50,000 × 4,096
= 204,800,000

---

## Q5. Why must tokenizer IDs remain compatible with a pretrained model?

The model learns parameters associated with specific token IDs.

If the tokenizer changes the mapping:

hello → 10

to:

hello → 20

then the model will retrieve a different embedding vector.

Therefore:

Tokenizer vocabulary
and
Model embedding matrix

must remain compatible.

---

## Q6. What happens when a word is not directly present in the vocabulary?

A modern subword tokenizer does not necessarily require the complete word to exist as a vocabulary entry.

Instead, it attempts to represent the text using smaller known subword or byte-level units.

### Example

Suppose the vocabulary contains:

"un"
"believ"
"able"

The word:

"unbelievable"

does not need to exist as one vocabulary entry.

The tokenizer can represent it as:

"unbelievable"
→ ["un", "believ", "able"]

→ [101, 2045, 309]

Therefore the model can represent a word that was not stored as a single vocabulary token.

### Why is this useful?

The vocabulary cannot contain every possible word.

There are:

- rare words
- new words
- names
- technical terms
- URLs
- typos
- code
- multilingual text
- newly created words

Subword tokenization allows these strings to be composed from reusable pieces.

### Modern LLM behavior

Modern tokenizers commonly use subword- and/or byte-oriented approaches so that arbitrary input can usually be represented without collapsing the entire string into `<UNK>`.

Byte-level fallback provides an additional mechanism for representing unusual or previously unseen text.

### Important distinction

"Not present as a complete vocabulary token"

does NOT mean:

"Cannot be tokenized."

For example:

"unbelievable"

may not exist as one token, but it can still be represented as:

["un", "believ", "able"]

### Engineering Principle

A modern tokenizer should maximize its ability to represent arbitrary input while keeping the vocabulary and resulting token sequence reasonably efficient.

---

## Q7. Why does vocabulary size matter?

A larger vocabulary can represent common strings using fewer tokens.

However, it also increases model parameters and memory requirements, particularly for embedding-related components.

Therefore vocabulary size is a model architecture and inference trade-off.

The tokenizer designer has to balance:

```text
Vocabulary size
        │
        ├── Sequence length
        ├── Embedding parameters
        ├── Model memory
        ├── Multilingual coverage
        ├── Compression efficiency
        └── Tokenizer throughput
```

These pull in opposite directions:

* **Sequence length** — a larger vocabulary encodes the same text in fewer tokens.
* **Embedding parameters** — but every added token costs `D` more parameters in the `(V, D)` embedding matrix (and again in the output projection).
* **Model memory** — those parameters occupy GPU memory for the entire life of the model.
* **Multilingual coverage** — a vocabulary tuned to one language fragments the others into many short pieces, so covering many languages demands more entries.
* **Compression efficiency** — how many characters the tokenizer packs into one token on real traffic; the practical measure of whether the vocabulary is being spent well.
* **Tokenizer throughput** — the encode step itself costs CPU time, and more merge rules make it slower.

There is no single correct vocabulary size — it depends on the target languages, the domain, and the serving constraints.

---

## Example: Complete Flow

Vocabulary:

| Token | ID |
|---|---:|
| `<PAD>` | 0 |
| `<UNK>` | 1 |
| `I` | 2 |
| `love` | 3 |
| `AI` | 4 |

Input:

"I love AI"

Tokenization:

["I", "love", "AI"]

Vocabulary lookup:

[2, 3, 4]

Embedding lookup:

[
  embedding[2],
  embedding[3],
  embedding[4]
]

The resulting vectors become the input representation for the Transformer.

---

## Engineering Principle

A tokenizer vocabulary is the contract between the tokenizer and the model's embedding layer.

The tokenizer produces IDs.

The model interprets those IDs using its learned embedding parameters.

---

## What problem does the next step solve?

We now have:

Token
→ ID

But we haven't yet defined how the tokenizer should handle the reverse operation:

ID
→ Token

We also need to understand why both directions are required.

Step 2.4 solves this by building the complete token ↔ ID mapping and encoding/decoding foundation.


# 2.4 — Token ↔ ID Mapping, Encoding and Decoding

## Q1. What is encoding?

Encoding converts tokens into integer token IDs using the vocabulary.

Example:

Vocabulary:

| Token | ID |
|---|---:|
| `<UNK>` | 0 |
| `I` | 1 |
| `love` | 2 |
| `AI` | 3 |

Tokens:

["I", "love", "AI"]

Encoding:

["I", "love", "AI"]
→ [1, 2, 3]

---

## Q2. What is decoding?

Decoding converts token IDs back into tokens and eventually reconstructs text.

Example:

[1, 2, 3]
→ ["I", "love", "AI"]
→ "I love AI"

---

## Q3. What is the difference between vocabulary and tokenizer?

The vocabulary provides the mapping:

token ↔ ID

The tokenizer coordinates the complete process:

text
→ tokens
→ IDs

and:

IDs
→ tokens
→ text

---

## Q4. Why do we need both encoding and decoding?

Encoding is required to convert user input into the integer IDs consumed by the model.

Decoding is required to convert model output token IDs back into human-readable text.

Complete flow:

User text
→ Tokenizer
→ Token IDs
→ Transformer
→ Output Token IDs
→ Tokenizer
→ Text

---

## Q5. What is a round-trip test?

A round-trip test verifies that encoding followed by decoding preserves the original representation.

Example:

tokens = ["I", "love", "AI"]

encode(tokens)
→ [1, 2, 3]

decode([1, 2, 3])
→ ["I", "love", "AI"]

Therefore:

decode(encode(tokens)) == tokens

This is an important tokenizer correctness property.

---

## Q6. Why can't we simply use `text.split(" ")`?

Whitespace splitting is too simplistic for production tokenization.

Example:

"hello, world!"

A simple split produces:

["hello,", "world!"]

But the tokenizer may need to treat punctuation and whitespace separately.

Modern tokenizers have normalization, pre-tokenization, and subword-processing stages.

---

## Q7. Why can whitespace be part of a token?

Some tokenizer designs encode whitespace as part of the following token.

Conceptually:

"I love AI"

may become:

["I", " love", " AI"]

The leading spaces allow the decoder to reconstruct the original text more naturally.

The exact whitespace representation depends on the tokenizer design.

---

## Example: Complete Encoding Flow

Input:

"I love AI"

Assume tokenization has already produced:

["I", "love", "AI"]

Vocabulary:

"I" → 1
"love" → 2
"AI" → 3

Encoding:

["I", "love", "AI"]
→ [1, 2, 3]

The model receives:

[1, 2, 3]

After generation, suppose the model produces:

[1, 2, 3]

Decoding:

[1, 2, 3]
→ ["I", "love", "AI"]
→ "I love AI"

---

## Modern LLM Perspective

A production tokenizer is more than a token-to-ID dictionary.

A simplified pipeline is:

Raw Text
→ Normalization
→ Pre-tokenization
→ Subword algorithm
→ Vocabulary lookup
→ Token IDs

Decoding reverses the process:

Token IDs
→ Vocabulary lookup
→ Token pieces
→ Decoding/post-processing
→ Text

Our current implementation only implements:

Tokens ↔ Token IDs

We will implement text segmentation separately.

---

## Engineering Principle

The vocabulary is the mapping layer.

The tokenizer is the complete text-processing layer.

A pretrained model and its tokenizer must remain compatible because the model's learned parameters depend on the tokenizer's token IDs.

---

## What problem does the next step solve?

We can now convert:

Tokens
↔
Token IDs

But we still cannot reliably process raw text.

We need to handle tokens that have a special role in the model's input structure.

Examples include:

<PAD>
<BOS>
<EOS>
<UNK>

The next step solves:

**How should a tokenizer represent special control tokens?**

Step 2.5 — Special Tokens.

# 2.5 — Special Tokens

## Q1. What are special tokens?

Special tokens are vocabulary entries used to represent structural or control information rather than ordinary text.

Examples:

<PAD>
<UNK>
<BOS>
<EOS>

---

## Q2. What is the purpose of `<PAD>`?

`<PAD>` allows sequences of different lengths to be placed into a common batch shape.

Example:

Sequence A:

["I", "love", "AI"]

Sequence B:

["I", "love", "Transformers", "are", "powerful"]

Padding to length 5:

A:
["I", "love", "AI", "<PAD>", "<PAD>"]

B:
["I", "love", "Transformers", "are", "powerful"]

The corresponding attention mask can be:

A:
[1, 1, 1, 0, 0]

B:
[1, 1, 1, 1, 1]

`1` represents a real token and `0` represents padding.

---

## Q3. How is `<PAD>` connected to attention masking?

Padding tokens are not meaningful input content.

Therefore the Transformer can use an attention mask to prevent padded positions from participating in attention inappropriately.

This connects tokenization directly to Transformer attention.

---

## Q4. What is `<BOS>`?

`<BOS>` means Beginning Of Sequence.

It can explicitly mark the beginning of a sequence.

Example:

<BOS> I love AI

However, not every modern LLM uses a BOS token in exactly this form or in every context.

Special-token behavior is model-specific.

---

## Q5. What is `<EOS>`?

`<EOS>` means End Of Sequence.

It can tell a generation system that the sequence has ended.

Example:

The answer is 42 <EOS>

During generation:

normal token
→ continue

EOS
→ stop

The exact EOS token is model-specific.

---

## Q6. What is `<UNK>`?

`<UNK>` means Unknown.

A basic tokenizer can use it when it cannot represent an input using its vocabulary.

Example:

"unknownword"
→ <UNK>

However, modern subword- and byte-oriented tokenizers are designed to represent arbitrary input without relying heavily on `<UNK>`.

---

## Q7. Are special tokens used by every modern LLM?

No.

Different models use different tokenizer configurations and control tokens.

The exact special tokens, IDs, and behavior are part of the model's tokenizer configuration.

Therefore we should not assume:

<BOS> always has the same ID
<EOS> always has the same ID
<PAD> is always required

---

## Q8. Why are special tokens important for chat LLMs?

Chat models need to represent conversation structure such as:

- system messages
- user messages
- assistant messages
- message boundaries

Modern models commonly use model-specific control tokens and chat templates for this purpose.

Conceptually:

system
→ control representation

user
→ control representation

assistant
→ control representation

The exact token format depends on the model.

---

## Example: Special Tokens + Normal Tokens

Vocabulary:

| ID | Token |
|---:|---|
| 0 | `<PAD>` |
| 1 | `<UNK>` |
| 2 | `<BOS>` |
| 3 | `<EOS>` |
| 4 | `I` |
| 5 | `love` |
| 6 | `AI` |

Input:

<BOS> I love AI <EOS>

Token IDs:

[2, 4, 5, 6, 3]

---

## Example: Padding

Two sequences:

A:
I love AI

B:
I love Transformers are powerful

After tokenization:

A:
[I, love, AI]

B:
[I, love, Transformers, are, powerful]

After padding:

A:
[I, love, AI, <PAD>, <PAD>]

B:
[I, love, Transformers, are, powerful]

Attention masks:

A:
[1, 1, 1, 0, 0]

B:
[1, 1, 1, 1, 1]

---

## Modern LLM Perspective

Special tokens are part of the tokenizer/model contract.

They can control:

- sequence boundaries
- padding
- generation termination
- conversation structure
- other model-specific control behavior

The exact special-token set is model-dependent.

---

## Engineering Principle

Never assume that special-token IDs are universal.

Always obtain the tokenizer configuration associated with the specific pretrained model.

---

## What problem does the next step solve?

We now understand:

Raw text
→ tokens
→ vocabulary
→ IDs
→ special tokens

But we still haven't solved the most important practical problem:

**How does the tokenizer convert arbitrary raw text into the correct sequence of tokens?**

For example:

"I love Transformers!"

How should it become:

["I", " love", " Transformers", "!"]

And how should whitespace, punctuation, Unicode, and word boundaries be handled?

The next step solves this with:

# Special Tokens in Modern LLMs

Special tokens are model-specific.

There is no universal special-token vocabulary shared by all LLMs.

## Comparison

| Model | Important tokens | Purpose |
|---|---|---|
| Llama 3 | `<|begin_of_text|>` | Start prompt |
| Llama 3 | `<|start_header_id|>` / `<|end_header_id|>` | Message role boundaries |
| Llama 3 | `<|eot_id|>` | End of turn |
| Llama 3 | `<|eom_id|>` | End of message / tool-related control |
| Llama 3 | `<|end_of_text|>` | End of text |
| Qwen2 | `<|im_start|>` | Message start |
| Qwen2 | `<|im_end|>` | Message end |
| Qwen2 | `<|endoftext|>` | EOS / model-specific padding/BOS role |
| Gemma | `<bos>` | Beginning of sequence |
| Gemma | `<eos>` | End of sequence |
| Gemma | `<unk>` | Unknown token |
| Gemma | `<pad>` | Padding |
| Mistral Instruct | `<s>` | Beginning of sequence |
| Mistral Instruct | `</s>` | End of sequence |
| Mistral Instruct | `<unk>` | Unknown token |
| Mistral Instruct | `[INST]` | Instruction start |
| Mistral Instruct | `[/INST]` | Instruction end |

## Example: Llama 3

Conceptually:

<|begin_of_text|>
<|start_header_id|>user<|end_header_id|>

Explain attention.

<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>

The model generates the answer here.

The tokens are not merely representing natural-language words.

They also represent conversation structure.

## Example: Qwen2

Conceptually:

<|im_start|>user
Explain attention.
<|im_end|>

<|im_start|>assistant

The tokenizer configuration defines these tokens and the chat template uses them to serialize messages.

## Example: Gemma

Conceptually:

<bos>
Explain attention.
<eos>

Gemma exposes the more traditional BOS/EOS/UNK/PAD concepts.

## Example: Mistral Instruct

Conceptually:

<s>[INST] Explain attention. [/INST]
The answer.</s>

`[INST]` and `[/INST]` define the instruction boundary, while `<s>` and `</s>` represent sequence boundaries.

## Important Engineering Principle

Special tokens are NOT universal.

Never hard-code assumptions such as:

`<EOS>` always exists.

`<EOS>` always has the same ID.

`<PAD>` is always required.

`<BOS>` is always automatically added.

Instead, load the tokenizer configuration belonging to the exact model/version.

## Special Tokens vs Chat Templates

These are different concepts.

Special tokens:
→ vocabulary entries with specific token IDs.

Chat template:
→ rules for converting structured messages into the exact token/control-token sequence expected by the model.

Pipeline:

Messages
→ Chat Template
→ Special/Control Tokens
→ Tokenizer
→ Token IDs
→ Model

**Step 2.6 — Normalization and Pre-tokenization.**

# 2.6 — Normalization and Pre-tokenization

## Q1. What is normalization?

> Normalization is, in a nutshell, a set of operations you apply to a raw string to make it less random or “cleaner”

Normalization transforms raw text into a more consistent representation before tokenization.

Possible operations include:

- lowercasing
- Unicode normalization
- whitespace normalization
- accent transformations
- model-specific transformations

### Example

Input:

"HELLO WORLD"

With lowercase normalization:

"hello world"

Normalization is model-specific. We should not assume that lowercasing is always correct.

---

## Q2. What is pre-tokenization?

> Pre-tokenization is the act of splitting a text into smaller objects that give an upper bound to what your tokens will be at the end of training. A good way to think of this is that the pre-tokenizer will split your text into “words” and then, your final tokens will be parts of those words.

Pre-tokenization splits normalized text into candidate pieces before the subword tokenization algorithm is applied.

### Example

Input:

"Hello, world!"

Possible pre-tokenization:

["Hello", ",", "world", "!"]

The BPE algorithm can then operate within these candidate pieces.

---

## Q3. Why don't we simply use `text.split(" ")`?

Whitespace splitting does not properly separate punctuation.

Example:

"Hello, world!"

Simple whitespace splitting:

["Hello,", "world!"]

A pre-tokenizer can instead produce:

["Hello", ",", "world", "!"]

This provides cleaner boundaries for subsequent subword tokenization.

---

## Q4. What is the difference between normalization and pre-tokenization?

Normalization changes the representation of the text.

Example:

"HELLO"
→ "hello"

Pre-tokenization splits text into candidate pieces.

Example:

"Hello, world!"
→ ["Hello", ",", "world", "!"]

Pipeline:

Raw text
→ Normalization
→ Pre-tokenization
→ Candidate pieces

---

## Q5. Why is pre-tokenization important for BPE?

Pre-tokenization establishes boundaries for the pieces that BPE can further split and merge.

Example:

"unbelievable"

Pre-tokenization:

["unbelievable"]

BPE can then learn:

["un", "believ", "able"]

Without appropriate boundaries, a tokenizer could learn undesirable merges across multiple words.

---

## Q6. Is BPE the entire tokenizer?

No.

BPE is the tokenization model/algorithm.

A production tokenizer can contain several components:

→ Normalizer
→ PreTokenizer
→ BPE/Unigram/WordPiece
→ PostProcessor
→ Decoder

Therefore:

Tokenizer ≠ BPE alone.

---

## Q7. How do modern tokenizers handle bytes?

Byte-oriented tokenizers can represent text at the byte level.

Conceptually:

Text
→ UTF-8 bytes
→ byte representation
→ subword/BPE processing
→ token IDs

A byte-level representation can provide coverage for arbitrary input without requiring an `<UNK>` token for every unseen string.

---

## Example: Complete Pipeline

Input:

"Hello, world!"

Normalization:

"Hello, world!"

Pre-tokenization:

["Hello", ",", "world", "!"]

BPE:

["Hello", ",", "world", "!"]

Vocabulary lookup:

[token_id_1, token_id_2, token_id_3, token_id_4]

The exact BPE segmentation depends on the vocabulary and learned merge rules.

---

## Modern LLM Perspective

Modern tokenizer implementations commonly separate:

1. Normalization
2. Pre-tokenization
3. Tokenization model
4. Post-processing
5. Decoding

For example, Hugging Face's Tokenizers architecture follows this pipeline.

Different model families can use different choices for each component.

Therefore tokenizer behavior is model-specific.

---

## Engineering Principle

Never assume that a tokenizer is simply:

text.split()

A production tokenizer is a pipeline of deterministic transformations whose exact configuration must remain compatible with the pretrained model.

---

## What problem does the next step solve?

We can now convert:

Raw text
→ normalized text
→ candidate pieces

But we still cannot convert:

"unbelievable"

into useful reusable subwords.

The next problem is:

**How can we learn subword units automatically from a training corpus?**

Step 2.7 will introduce the core idea behind BPE.

# 2.7 — BPE Intuition

## Q1. What problem does BPE solve?

BPE solves the problem of learning useful reusable subword units from a training corpus.

Instead of manually defining every possible word, BPE starts with small units and repeatedly merges frequent adjacent pairs.

---

## Q2. What is the core BPE algorithm?

The simplified algorithm is:

1. Start with small units.
2. Count adjacent token pairs.
3. Find the most frequent pair.
4. Merge the pair.
5. Add the merged token to the vocabulary.
6. Record the merge rule.
7. Repeat until the desired vocabulary size or merge count is reached.

---

## Q3. Example of BPE training

Corpus:

low
lower
lowest

Initial representation:

l o w
l o w e r
l o w e s t

Pair frequencies:

(l, o) → 3
(o, w) → 3
(w, e) → 2
(e, r) → 1
(e, s) → 1
(s, t) → 1

Choose:

(l, o)

Merge:

l + o → lo

Corpus becomes:

lo w
lo w e r
lo w e s t

Next frequent pair:

(lo, w)

Merge:

lo + w → low

Corpus becomes:

low
low e r
low e s t

BPE has discovered reusable subword units from corpus statistics.

---

## Q4. Does BPE understand language morphology?

Not fundamentally.

BPE primarily learns frequent sequences.

For example:

playing
→ play + ing

may look linguistically meaningful, but BPE discovered the pieces because they occur frequently in the training corpus.

BPE is therefore statistical rather than a linguistic parser.

---

## Q5. Why is BPE useful for rare words?

Suppose the vocabulary contains:

play
ing
ed
er

Then:

playing
→ play + ing

played
→ play + ed

player
→ play + er

The tokenizer can reuse known pieces rather than requiring every complete word to be stored independently.

---

## Q6. What are BPE merge rules?

Merge rules record which adjacent token pairs should be merged and their priority/order.

Example:

1. l + o → lo
2. lo + w → low
3. low + e → lowe

These rules are learned during tokenizer training and reused during inference.

---

## Q7. What is the difference between BPE training and BPE inference?

### Training

Corpus
→ count pairs
→ merge frequent pairs
→ learn merge rules
→ learn vocabulary

### Inference

New text
→ initial units
→ apply learned merge rules
→ final tokens
→ token IDs

Pair frequencies are not relearned for every user request.

---

## Q8. Why does merge order matter?

Multiple pairs can be present at the same time.

Example:

a b c

Possible pairs:

(a, b)
(b, c)

A deterministic tokenizer needs to know which merge has priority.

Therefore BPE maintains an ordered/ranked list of learned merge rules.

---

## Q9. How does BPE control vocabulary size?

The tokenizer can stop after a target number of merges.

Conceptually:

Initial vocabulary
+
Number of merges
≈
Final vocabulary size

The exact accounting depends on the tokenizer implementation and special tokens.

---

## Example: BPE Inference

Suppose training learned:

t + h → th
th + e → the
i + n → in

Input:

"thin"

Initial:

[t, h, i, n]

Apply:

t + h → th

Result:

[th, i, n]

Apply:

i + n → in

Result:

[th, in]

Therefore:

"thin"
→ ["th", "in"]

---

## Modern LLM Perspective

BPE remains an important tokenizer design, but not every modern LLM uses exactly the same tokenizer.

Modern models can use:

- BPE
- byte-level BPE
- SentencePiece/BPE variants
- Unigram
- other model-specific tokenizer designs

Therefore our implementation teaches the underlying BPE mechanism rather than claiming that all LLMs use identical tokenization.

---

## Engineering Principle

BPE is a statistical vocabulary-learning algorithm.

It learns reusable subword units by repeatedly merging frequent adjacent pairs.

The trained tokenizer stores the resulting vocabulary and ordered merge rules so that inference can tokenize new text deterministically.

---

## What problem does the next step solve?

We understand BPE conceptually.

But we have not implemented it.

The next problem is:

**How do we efficiently count adjacent token pairs and perform BPE merges in code?**

## Complete Example — Raw Text to BPE Token IDs

Training corpus:

low
lower
lowest

### 1. Raw text

"lowest"

### 2. Normalization

"lowest"

No normalization is required for this example.

### 3. Pre-tokenization

["lowest"]

### 4. Initial representation

For our educational implementation, start with characters:

[l, o, w, e, s, t]

### 5. BPE training

Training corpus:

low
→ l o w

lower
→ l o w e r

lowest
→ l o w e s t

Pair frequencies:

(l, o) → 3
(o, w) → 3
(w, e) → 2
(e, r) → 1
(e, s) → 1
(s, t) → 1

First merge:

l + o → lo

Second merge:

lo + w → low

Third merge:

low + e → lowe

Learned merge rules:

1. l + o → lo
2. lo + w → low
3. low + e → lowe

### 6. BPE inference

Input:

"lowest"

Initial:

[l, o, w, e, s, t]

Apply merge #1:

[lo, w, e, s, t]

Apply merge #2:

[low, e, s, t]

Apply merge #3:

[lowe, s, t]

Final tokens:

["lowe", "s", "t"]

### 7. Vocabulary lookup

Suppose:

lowe → 9
s → 5
t → 6

Then:

["lowe", "s", "t"]
→ [9, 5, 6]

The Transformer receives:

[9, 5, 6]

### Important observation

BPE does not necessarily discover linguistically meaningful units such as:

["low", "est"]

The result depends on:

- training corpus
- pair frequencies
- number of merges
- vocabulary size
- merge ordering
- tokenizer implementation

### Training vs Inference

BPE Training:

Corpus
→ pair counting
→ repeated merges
→ vocabulary + merge rules

BPE Inference:

New text
→ initial units
→ learned merge rules
→ tokens
→ token IDs

Pair frequencies are learned during tokenizer training and are not recalculated for every user prompt.

### Modern LLM Difference

Our educational example starts with characters.

Production byte-level BPE tokenizers can start from UTF-8 bytes instead:

Text
→ UTF-8 bytes
→ BPE merges
→ token IDs

This provides robust coverage for arbitrary Unicode input.

We will study byte-level tokenization separately.

```text
                 RAW TEXT
                    │
                    ▼
              "lowest"
                    │
                    ▼
              Normalization
                    │
                    ▼
              "lowest"
                    │
                    ▼
             Pre-tokenization
                    │
                    ▼
              ["lowest"]
                    │
                    ▼
           Initial character units
                    │
                    ▼
            [l, o, w, e, s, t]
                    │
                    ▼
              BPE Merge #1
              l + o → lo
                    │
                    ▼
             [lo, w, e, s, t]
                    │
                    ▼
              BPE Merge #2
             lo + w → low
                    │
                    ▼
             [low, e, s, t]
                    │
                    ▼
              BPE Merge #3
             low + e → lowe
                    │
                    ▼
              [lowe, s, t]
                    │
                    ▼
                Vocabulary
                    │
                    ▼
                 [9, 5, 6]
                    │
                    ▼
               Transformer
```

Step 2.8 will implement the first BPE training algorithm on a tiny corpus.
# 2.8 — Implementing the BPE Trainer

## Q1. What does the BPE trainer produce?

The BPE trainer learns two important artifacts:

1. Vocabulary
2. Ordered merge rules

Example:

Vocabulary:

l
o
w
e
r
s
t
lo
low
lowe

Merge rules:

1. l + o → lo
2. lo + w → low
3. low + e → lowe

These artifacts are used by the tokenizer during inference.

---

## Q2. How does pair counting work?

Given:

[l, o, w, e, r]

Adjacent pairs are:

(l, o)
(o, w)
(w, e)
(e, r)

The tokenizer counts these pairs across the entire training corpus.

Example:

Corpus:

low
lower
lowest

Pair frequencies:

(l, o) → 3
(o, w) → 3
(w, e) → 2
(e, r) → 1
(e, s) → 1
(s, t) → 1

---

## Q3. How does BPE choose a merge?

BPE chooses the most frequent adjacent pair.

Example:

(l, o) → 3
(o, w) → 3
(w, e) → 2

One of the highest-frequency pairs is selected according to the tokenizer's deterministic tie-breaking behavior.

---

## Q4. What happens after selecting a pair?

The selected pair is merged everywhere in the training corpus.

Example:

[l, o, w, e]

Merge:

l + o → lo

Result:

[lo, w, e]

The new token `lo` is added to the vocabulary.

The merge rule is also recorded.

---

## Q5. Why do we store merge rules?

The tokenizer must use the learned behavior during inference.

Example:

Training learned:

l + o → lo
lo + w → low

When new text contains:

[l, o, w]

the tokenizer can apply:

[l, o, w]
→ [lo, w]
→ [low]

Without the merge rules, the tokenizer would not know how to reproduce the learned segmentation.

---

## Q6. What is the difference between BPE training and inference?

### Training

Training corpus
→ initial units
→ count pairs
→ merge most frequent pair
→ record rule
→ repeat

Result:

Vocabulary + Merge Rules

### Inference

New text
→ initial units
→ apply learned merge rules
→ final tokens
→ token IDs

Pair frequencies are not learned again for every user request.

---

## Q7. Why should the trainer avoid mutating the caller's corpus?

A function such as:

trainer.train(corpus)

should not unexpectedly modify the caller's nested lists.

Therefore the trainer can create its own copy before training.

This prevents hidden side effects and makes the API safer.

---

## Q8. What happens with overlapping pairs?

Input:

[a, a, a]

Merge:

a + a → aa

A left-to-right non-overlapping merge produces:

[aa, a]

The first two symbols are consumed by the merge.

The remaining symbol is processed afterward.

---

## Q9. Is our BPE implementation production-ready?

No.

It is an educational implementation designed to expose the algorithm.

Our implementation repeatedly:

1. scans the corpus
2. counts pairs
3. finds the best pair
4. scans the corpus again
5. performs the merge

This can become expensive for large corpora.

Production tokenizer implementations use optimized algorithms and data structures.

We will study these performance considerations later.

---

## Example — Complete BPE Training

Corpus:

low
lower
lowest

Initial:

l o w
l o w e r
l o w e s t

Iteration 1:

(l, o) → 3

Merge:

l + o → lo

Corpus:

lo w
lo w e r
lo w e s t

Iteration 2:

(lo, w) → 3

Merge:

lo + w → low

Corpus:

low
low e r
low e s t

Iteration 3:

(low, e) → 2

Merge:

low + e → lowe

Final learned rules:

1. l + o → lo
2. lo + w → low
3. low + e → lowe

---

## Modern LLM Perspective

The BPE learning concept remains simple:

frequent pair
→ merge
→ new vocabulary token

However, production tokenizers optimize:

- pair counting
- merge lookup
- memory usage
- vocabulary storage
- Unicode/byte handling
- inference speed
- batch processing

Therefore the algorithmic idea is simple while production implementation is significantly more sophisticated.

---

## Engineering Principle

BPE training and BPE inference are separate phases.

Training learns:

Vocabulary + Merge Rules

Inference uses:

Merge Rules + Vocabulary

The tokenizer must preserve these artifacts exactly because they define how text is converted into the token IDs expected by the trained model.

---

## What problem does the next step solve?

We can now TRAIN BPE.

But we cannot yet use our learned BPE tokenizer on a new input.

For example:

"lowest"

must become something like:

[lowe, s, t]

using the learned merge rules.

The next step solves:

**How do we apply ranked BPE merge rules during inference?**

Step 2.9 — BPE Inference.

## Q. Why do we separate tokenizer training from inference?

Modern LLMs separate **tokenizer training** from **tokenization during inference**.

The tokenizer is trained **once** on a large corpus to learn how text should be segmented into tokens. After training, the learned tokenizer is saved to disk and reused for every future request.

The model never retrains its tokenizer when a user sends a prompt.

---

### Training Phase (Offline)

During training, the tokenizer learns patterns from a large text corpus.

Pipeline:

```text
Training Corpus
        │
        ▼
Read Raw Text
        │
        ▼
Normalization
        │
        ▼
Pre-tokenization
        │
        ▼
Initial Symbols
        │
        ▼
BPE Training
        │
        ▼
Vocabulary
+
Merge Rules
        │
        ▼
Save tokenizer.json
```

**Output of training:**

- Vocabulary
- Merge Rules
- Normalizer configuration
- Special Tokens
- Tokenizer configuration

Example saved artifact:

```text
tokenizer.json
├── vocabulary
├── merge_rules
├── normalizer
├── special_tokens
└── configuration
```

Once these artifacts are saved, the original training corpus is no longer required.

---

### Inference Phase (Online)

When a user sends a prompt, the tokenizer does **not** learn anything new.

Instead, it loads the previously trained tokenizer.

Pipeline:

```text
User Prompt
      │
      ▼
Load tokenizer.json
      │
      ▼
Normalization
      │
      ▼
Pre-tokenization
      │
      ▼
Initial Symbols
      │
      ▼
Apply Learned Merge Rules
      │
      ▼
Final Tokens
      │
      ▼
Vocabulary Lookup
      │
      ▼
Token IDs
      │
      ▼
Transformer Model
```

No pair counting or vocabulary learning happens during inference.

The tokenizer simply follows the rules learned during training.

---

## Example

### Training

Training corpus:

```text
I love machine learning.
I love deep learning.
Machine learning is powerful.
```

During training, BPE learns merge rules such as:

```text
l + e    → le
le + a   → lea
lea + r  → lear
lear + n → learn
```

The tokenizer also builds a vocabulary:

```text
i
love
machine
learn
learning
powerful
.
```

Finally, everything is saved into:

```text
tokenizer.json
```

---

### Inference

User enters:

```text
I love learning.
```

The tokenizer loads:

```text
tokenizer.json
```

Processing:

```text
Raw Text

I love learning.

        │
        ▼

Normalization

i love learning.

        │
        ▼

Pre-tokenization

[
    "i",
    "love",
    "learning",
    "."
]

        │
        ▼

Initial Symbols

l e a r n i n g

        │
        ▼

Apply Learned Merge Rules

le
↓

lea
↓

lear
↓

learn

        │
        ▼

Final Tokens

[
    "i",
    "love",
    "learn",
    "ing",
    "."
]

        │
        ▼

Vocabulary Lookup

[
    12,
    91,
    42,
    67,
    5
]
```

These integer IDs are the actual input to the Transformer.

---

## Why is this separation important?

If the tokenizer retrained itself for every user request:

- Vocabulary would change continuously.
- Token IDs would no longer match the model's embeddings.
- Previously trained model weights would become invalid.
- Every request would require expensive BPE training.

Instead, the tokenizer is trained once and reused indefinitely.

This guarantees that every occurrence of the same text is converted into the same token IDs.

---

## Modern LLM Workflow

```text
                     OFFLINE (Training)

Large Corpus
      │
      ▼
Tokenizer Training
      │
      ▼
Vocabulary
+
Merge Rules
      │
      ▼
Save tokenizer.json
═══════════════════════════════════════════════
                     ONLINE (Inference)

Load tokenizer.json
      │
      ▼
User Prompt
      │
      ▼
Normalization
      │
      ▼
Pre-tokenization
      │
      ▼
Apply Merge Rules
      │
      ▼
Tokens
      │
      ▼
Token IDs
      │
      ▼
Transformer Model
```

---

## How do modern LLMs work?

Exactly the same principle.

Examples:

- GPT-2 loads `vocab.json` and `merges.txt`.
- Llama loads `tokenizer.model` (SentencePiece).
- Qwen loads `tokenizer.json`.
- Mistral loads `tokenizer.model`.
- Gemma loads `tokenizer.model`.

Regardless of the tokenizer algorithm (BPE, WordPiece, SentencePiece, or Unigram), **training happens once**, while **encoding happens for every user request**.

---

## Key Takeaway

Tokenizer training is an **offline learning process** that creates the tokenizer.

Tokenization during inference is an **online encoding process** that uses the already learned tokenizer.

The tokenizer is part of the model and must remain unchanged after training so that token IDs always correspond to the embeddings learned by the Transformer.

## What problem are we solving in Normalization step?

Raw text can contain inconsistent capitalization.

Example:

I Love Machine Learning
i LOVE machine learning
I love machine learning

Without normalization, these are treated as different text patterns.

Normalization converts them into a consistent representation before tokenization begins, improving vocabulary quality and reducing unnecessary duplicate tokens.

In modern LLMs, normalization is configured as part of the tokenizer and is applied both during training and inference.

## Q. Why do we perform pre-tokenization before BPE?

BPE does not work directly on raw sentences.

Instead, the text is first divided into smaller units called **pre-tokens**.

A pre-tokenizer identifies natural boundaries such as:

- Whitespace
- Punctuation
- Digits
- Symbols

This prevents BPE from merging across unrelated words or punctuation.

### Example

Raw text:

```text
I love machine learning.
```

After normalization:

```text
i love machine learning.
```

After pre-tokenization:

```python
[
    "i",
    "love",
    "machine",
    "learning",
    "."
]
```

Instead of learning merges across the entire sentence, BPE now learns inside each pre-token.

---

### Why is this important?

Without pre-tokenization:

```text
learning.machine
```

could incorrectly become one sequence of symbols.

With pre-tokenization:

```python
[
    "learning",
    ".",
    "machine"
]
```

the tokenizer understands the boundaries between words and punctuation.

---

### Modern LLMs

Different models use different pre-tokenization strategies.

| Model | Pre-tokenization |
|--------|------------------|
| GPT-2 | Regex + Byte-level |
| BERT | Whitespace + punctuation |
| Llama | SentencePiece (integrated) |
| Gemma | SentencePiece |
| Mistral | SentencePiece |

Although the implementations differ, the goal is the same:

> Break raw text into manageable pieces before subword tokenization begins.

---

## Modern LLM Comparison

This is one place where our educational tokenizer **intentionally differs** from production systems.

| Aspect | Educational Project | Modern GPT-2 / Llama |
| ------ | ------------------- | -------------------- |
| Initial unit | Characters | UTF-8 bytes (GPT-2) or SentencePiece pieces (Llama) |
| Example | `learning` → `l e a r n i n g` | `learning` → bytes or learned pieces |
| Main benefit | Easy to visualize | More efficient and language-independent |

We start with characters because they make the BPE algorithm much easier to understand.

Every merge is readable, so you can watch the vocabulary being built step by step.

The trade-off is that a character-based starting point depends on which characters happened to appear in the training corpus, so genuinely unseen characters have no representation. Byte-level and SentencePiece approaches avoid this — every possible input decomposes into units the tokenizer already knows.

The algorithm itself does not change. Only the starting units differ.

## Q. Why does BPE count adjacent symbol pairs?

The goal of BPE is to build larger subword tokens by repeatedly merging the **most frequent adjacent symbol pair**.

Before any merge can happen, the tokenizer must know how often each adjacent pair appears across the entire training corpus.

### Example

Word:

```text
learning
```

Symbols:

```text
l e a r n i n g
```

Adjacent pairs:

```text
(l, e)
(e, a)
(a, r)
(r, n)
(n, i)
(i, n)
(n, g)
```

Only adjacent symbols are counted.

Pairs like `(l, a)` or `(e, r)` are ignored because they are not neighbors.

---

### Why?

If a pair appears frequently across many words, it is likely to represent a meaningful subword.

For example, if `(l, e)` appears thousands of times in the corpus, merging it into `le` reduces the total number of symbols while preserving common language patterns.

---

### Modern LLMs

GPT-2's BPE trainer follows the same idea:

1. Count adjacent symbol pairs.
2. Select the most frequent pair.
3. Merge that pair.
4. Repeat until the desired vocabulary size is reached.

This simple frequency-based algorithm is what gradually transforms characters (or bytes) into meaningful subword tokens.

## Q. Why does BPE merge only one pair at a time?

After counting adjacent symbol pairs, BPE selects the **single most frequent pair** and merges it across the entire corpus.

Only one merge is performed per iteration.

### Example

Before:

```text
learning

↓

l e a r n i n g
```

Most frequent pair:

```text
(l, e)
```

After merging:

```text
le a r n i n g
```

The merge rule is recorded:

```text
l + e → le
```

---

### Why not merge multiple pairs at once?

Because every merge changes the corpus.

For example:

Before:

```text
l e a r
```

If `(l, e)` is merged:

```text
le a r
```

The pair `(e, a)` no longer exists.

Therefore, BPE must recompute pair frequencies after every merge.

---

### Modern LLMs

GPT-2, RoBERTa, and other BPE-based tokenizers follow exactly the same greedy algorithm:

1. Count adjacent pairs.
2. Select the most frequent pair.
3. Merge that pair everywhere.
4. Repeat until the target vocabulary size is reached.

Although production tokenizers are heavily optimized, the learning algorithm is fundamentally the same.

## Q. Why is `('i', 'n')` the most frequent pair instead of `('l', 'e')`?

The BPE algorithm counts adjacent symbol pairs **across the entire corpus**, not just within a single word.

For example:

### machine

```text
m a c h i n e
        ↑ ↑
       i n
```

This contributes one occurrence of `(i, n)`.

### learning

```text
l e a r n i n g
          ↑ ↑
         i n
```

This also contributes one occurrence of `(i, n)`.

If the corpus contains:

- `machine` → 3 times
- `learning` → 5 times

then:

```text
(i, n) = 3 + 5 = 8
```

On the other hand, `(l, e)` only appears inside `learning`, so its count is:

```text
(l, e) = 5
```

This illustrates an important property of BPE:

> Pair frequencies are computed globally across the entire training corpus, not within individual words.

As a result, the most frequent pair often comes from multiple different words rather than just one.

## Q. How does BPE choose which pair to merge?

After counting all adjacent symbol pairs, BPE selects the pair with the highest frequency.

Example:

| Pair | Count |
|------|------:|
| (i, n) | 8 |
| (a, r) | 7 |
| (l, e) | 5 |

The tokenizer chooses:

```text
(i, n)
```

because it appears most often in the training corpus.

This pair becomes the next merge rule.

---

### Why only one pair?

Each merge changes the symbol sequences in the corpus.

After merging one pair, the frequencies of other pairs may also change.

Therefore, BPE must:

1. Count pairs.
2. Choose one pair.
3. Merge it.
4. Count again.

This process repeats until the desired vocabulary size is reached.

## Q. Why does BPE merge symbol pairs?

After identifying the most frequent adjacent pair, BPE replaces every occurrence of that pair with a new merged symbol.

Example:

Before:

```text
m a c h i n e
```

Merge rule:

```text
i + n → in
```

After:

```text
m a c h in e
```

The same merge is applied everywhere in the corpus.

---

### Why is this useful?

Frequent symbol combinations become a single token.

Instead of repeatedly processing:

```text
i
n
```

the tokenizer can process:

```text
in
```

as one unit.

Over many iterations, these merged symbols become meaningful subwords and eventually complete words.

---

### Why use a `while` loop?

When two symbols are merged into one, both symbols have already been consumed.

The algorithm must skip the second symbol.

For example:

```text
i n e
```

↓

```text
in e
```

After merging `i` and `n`, the next symbol to process is `e`.

A `while` loop allows the index to advance by **2** after a merge, while a `for` loop always advances by **1**.

## Q. Why do we need a `BPETrainer` class?

A BPE tokenizer does not stop after one merge.

Instead, it repeatedly performs the following steps:

1. Count adjacent symbol pairs.
2. Select the most frequent pair.
3. Merge that pair throughout the corpus.
4. Record the merge rule.
5. Repeat.

A `BPETrainer` encapsulates this iterative process.

### Benefits

- Keeps the training logic separate from low-level BPE functions.
- Stores the learned merge rules.
- Makes the code modular and easier to maintain.
- Mirrors the architecture of production tokenizer libraries such as Hugging Face's `BpeTrainer`.

### Training Flow

```text
Initial Corpus
      │
      ▼
Count Pairs
      │
      ▼
Find Best Pair
      │
      ▼
Merge
      │
      ▼
Save Merge Rule
      │
      ▼
Repeat
```

## Q. Why does the trainer maintain a vocabulary?

Every successful BPE merge creates a new symbol.

Example:

Before:

```text
i
n
```

After merging:

```text
in
```

The new symbol `in` becomes part of the tokenizer's vocabulary.

The trainer stores all learned symbols so they can later be assigned token IDs.

### Initial Vocabulary

Before any merges, the vocabulary contains only the initial symbols (characters in our implementation).

Example:

```text
a
b
c
...
```

### Vocabulary Growth

After each merge:

```text
(i, n) → in
```

the trainer adds:

```text
in
```

to the vocabulary.

Over thousands of iterations, the vocabulary grows from characters into meaningful subword units and eventually complete words.

### Why use a `set`?

The same merged symbol may appear many times in the corpus.

A `set` automatically keeps only one copy of each symbol, ensuring the vocabulary contains unique entries.

# Q. Why do we save the tokenizer after training?

Tokenizer training is computationally expensive and is performed only once.

After training, the tokenizer is serialized to disk (for example, `tokenizer.json`) so it can be reused during inference without retraining.

---

## Training Phase

```text
Corpus
    │
    ▼
Train Tokenizer
    │
    ▼
Vocabulary
Merge Rules
    │
    ▼
tokenizer.json
```

---

## Inference Phase

```text
User Prompt
      │
      ▼
Load tokenizer.json
      │
      ▼
Encode
      │
      ▼
LLM
```

The inference system never needs access to the original training corpus.

---

## Why save merge rules?

Merge rules describe how characters should be combined into larger subword tokens.

Without the learned merge rules, the tokenizer cannot reproduce the same tokenization used during model training.

---

## Why save the vocabulary?

The vocabulary contains every token that the tokenizer has learned.

During inference, these tokens are mapped to integer IDs before being passed to the language model.

## Q. Why are `BPETrainer` and `BPETokenizer` separate classes?

Tokenizer training and tokenizer inference are two different stages.

### BPETrainer

The trainer is used only once.

Responsibilities:

- Learn merge rules from a corpus.
- Build the vocabulary.
- Save the tokenizer to disk.

### BPETokenizer

The tokenizer is used every time a user sends a prompt.

Responsibilities:

- Load the saved tokenizer.
- Normalize input text.
- Pre-tokenize the text.
- Apply the learned merge rules.
- Convert tokens into token IDs.

### Architecture

```text
Training Corpus
      │
      ▼
BPETrainer
      │
      ▼
tokenizer.json
      │
──────────────────────────
      │
      ▼
BPETokenizer
      │
      ▼
Encoded Prompt
```

Separating training from inference keeps the system modular, reusable, and scalable. This is the same design used in Hugging Face Tokenizers, SentencePiece, and TikToken.

## Q. Why does the tokenizer replay merge rules during inference?

During training, the tokenizer learns merge rules in a specific order.

Example:

```text
1. (i, n) → in
2. (in, g) → ing
3. (l, e) → le
```

During inference, the tokenizer **does not search for the most frequent pair again**.

Instead, it simply replays the learned merge rules in the exact order they were discovered.

### Example

Input:

```text
machine
```

Initial symbols:

```text
m a c h i n e
```

Apply merge rules sequentially:

```text
m a c h in e
↓

m a ch in e
↓

m ach in e
↓

mach in e
↓

machine
```

### Why is the order important?

Later merge rules often depend on tokens created by earlier merge rules.

If the merge order changes, the tokenizer may produce different tokens than those used during model training, resulting in incorrect input IDs for the language model.

## Q. Why is a verbose mode useful in a tokenizer?

A tokenizer performs multiple stages internally:

1. Normalization
2. Pre-tokenization
3. Character splitting
4. Applying BPE merge rules
5. Producing final subword tokens

Without intermediate output, it is difficult to understand how the tokenizer transforms raw text.

A verbose mode prints every stage of the pipeline, making it easier to:

- Understand the algorithm.
- Debug incorrect tokenization.
- Verify learned merge rules.
- Visualize how characters become subword tokens.

### Example

```text
Raw Text
↓

Normalization
↓

Pre-tokenization
↓

Characters
↓

Merge Rule 1
↓

Merge Rule 2
↓

Merge Rule 3
↓

Final Tokens
```

Production tokenizers usually disable verbose output for performance reasons, but a debug mode is extremely valuable during development and education.