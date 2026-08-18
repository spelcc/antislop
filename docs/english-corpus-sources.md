# English corpus sources

The English classifier must use an English-only population universe. French profiles, French `wordfreq`, or translated test documents are not interchangeable with English evidence.

## LLM corpora

### Compar:IA

Primary recent source for the current model panel. The same official Ministry of Culture Parquet used by the French recipe contains English conversations and preserves model names, prompt identity, categories and language tags. The checked-in `corpora/comparia-en-editorial.json` recipe keeps the same editorial/knowledge category filter, removes code/markup, requires >=350 cleaned prose words and currently targets 25 documents/model because English availability is much lower for the rarest current models.

Source: `https://www.data.gouv.fr/datasets/comparia`

### LMSYS-Chat-1M

One million real-world conversations collected from Vicuna/Chatbot Arena, with 25 model names, detected language and full conversations. It is particularly useful as an independent English validation corpus because collection conditions differ from Compar:IA. The public dataset is gated and its 2023 model panel is older, so use it primarily for external validation and historical fingerprints rather than pretending it represents the 2026 model frontier.

Source: `https://huggingface.co/datasets/lmsys/lmsys-chat-1m`

### Chatbot Arena Conversations

33k cleaned pairwise conversations covering 20 models, with model names, detected language and full response text. It is also gated and older than the current Compar:IA panel, but provides a clean independent arena-style validation set.

Source: `https://huggingface.co/datasets/lmsys/chatbot_arena_conversations`

### WildChat

AllenAI's WildChat contains hundreds of thousands of real user conversations with GPT-3.5/GPT-4 and explicitly stores the underlying model and language. Its model diversity is narrow, but its unconstrained prompts make it useful for checking whether an OpenAI fingerprint survives outside arena/editorial collection conditions. Dataset license: ODC-BY.

Source: `https://huggingface.co/datasets/allenai/WildChat`

## Human corpora

Human reference data should preferably predate widespread LLM publication and have identifiable provenance. Generic recent web crawl text is risky because it can already contain generated prose.

### Europarl v10

English European Parliament proceedings with a monolingual corpus and document/speaker metadata. This is a large, pre-generative-AI formal-prose control.

Source: `https://www.statmt.org/europarl/v10/training-monolingual/`

### News Commentary

Human news/opinion commentary distributed by WMT. It supplies contemporary expository prose that is stylistically closer to IRZ than parliamentary transcripts.

Source index: `https://www2.statmt.org/wmt24/mtdata/`

### WMT News Crawl

Large English news corpora collected since 2007, available with document boundaries for English. Prefer pre-2022 snapshots for detector calibration to reduce the risk of LLM-contaminated human labels.

Source index: `https://www2.statmt.org/wmt24/mtdata/`

### GlobalVoices via OPUS

News and citizen-media stories in many languages, including English. It gives less institutional, more editorial prose and is useful as a separate human population rather than merging everything into one generic baseline.

Source: `https://opus.nlpl.eu/corpora`

### Project Gutenberg

Digitized books and other written works, overwhelmingly based on works not restricted by US copyright. Gutenberg is useful for long-form literary/historical English, but it should remain a separate population because era and literary genre differ sharply from modern web editorial prose.

Source: `https://www.gutenberg.org/`

## Recommended English design

For CI, keep source populations separate and split each source before building any fingerprint:

1. **candidate/signature** documents build the population fingerprint and style profile;
2. **calibration train** fits the balanced Human/LLM logistic layer;
3. **calibration test** is never present in the baseline or candidate signatures;
4. external sources such as LMSYS/WildChat are reserved for cross-dataset validation whenever possible.

The first English pilot currently uses 28 Compar:IA models × 25 clean responses plus GlobalVoices, News Commentary and Europarl. Its result is useful but less mature than the French 300-doc/model classifier. The next English corpus revision should add pre-2022 WMT News Crawl and Gutenberg human populations, then use LMSYS/WildChat as external LLM validation rather than training on every available source at once.
