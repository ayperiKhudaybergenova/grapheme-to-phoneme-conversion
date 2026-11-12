# Grapheme-to-Phoneme Conversion: NLP & Deep Learning

A Natural Language Processing (NLP) project focused on converting written words into their phonemic representations using machine learning and sequence-to-sequence learning.

The project explores how computational models can learn the relationship between orthographic representations (graphemes) and phonological representations (phonemes).

## Project Overview

Grapheme-to-phoneme (G2P) conversion is a fundamental NLP task that maps written language to its pronunciation.

For example:

```text
Input:  computer
Output: kəmpjutɚ
```

The task involves learning mappings between two different sequences:

- **Input:** Orthographic character sequences (graphemes).
- **Output:** Phonemic sequences represented using the International Phonetic Alphabet (IPA).

**Objective:** Develop models that predict phonemic transcriptions from orthographic word representations.

## Technical Skills & Technologies

- **Programming:** Python
- **Natural Language Processing:** Grapheme-to-phoneme conversion, phonological processing
- **Machine Learning:** Sequence transduction, supervised learning
- **Deep Learning:** Encoder-decoder architectures, neural sequence modeling
- **Data Processing:** Unicode and IPA text processing
- **Libraries:** PyTorch / TensorFlow, OpenNMT, NumPy
- **Evaluation:** Exact Match Accuracy, Levenshtein Distance
- **Development:** Git, GitHub

## Methodology

G2P conversion can be formulated as a sequence-to-sequence learning problem.

A model learns to transform an input sequence of graphemes into an output sequence of phonemes.

The general workflow consists of:

1. Preprocessing orthographic and phonemic representations.
2. Preparing character and phoneme vocabularies.
3. Encoding input sequences.
4. Learning grapheme-to-phoneme mappings.
5. Generating phonemic sequences.
6. Evaluating predicted transcriptions.

## Machine Learning Approaches

Potential approaches for this task include:

- Character-level sequence modeling
- Encoder-decoder neural networks
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory networks (LSTM)
- Attention-based sequence-to-sequence architectures
- Transformer-based architectures

## Dataset

The project uses a grapheme-to-phoneme dataset originating from a SIGMORPHON shared task.

Each training instance contains:

- An orthographic word
- Its corresponding phonemic transcription
- Phoneme segments represented as tokenized IPA sequences

The test dataset contains orthographic words for which phonemic transcriptions must be generated.

## Evaluation

Model performance is evaluated using:

- **Exact Match Accuracy:** Percentage of words with completely correct predicted phonemic sequences.
- **Average Levenshtein Distance:** Measures the number of edits required to transform a predicted sequence into the reference transcription.

These metrics evaluate both complete transcription accuracy and the quality of individual predictions.

## Project Structure

```text
grapheme-to-phoneme-conversion/
│
├── nlp2-a4.train
├── nlp2-a4.test
├── a4-<method>.py
├── a4-<method>.predictions
├── README.md
└── .gitignore
```

For OpenNMT implementations, the corresponding YAML configuration and referenced supporting files should also be included.

## Running the Project

Install the dependencies required by the selected implementation.

Example:

```bash
pip install numpy pandas torch
```

Run a model:

```bash
python a4-<method>.py
```

The script generates phonemic predictions in the required tab-separated format.

## Applications

Grapheme-to-phoneme conversion is relevant to:

- Text-to-speech (TTS) systems
- Automatic speech processing
- Pronunciation dictionaries
- Speech synthesis
- Computational linguistics
- Language technology applications

## Learning Outcomes

This project demonstrates practical experience with:

- Sequence-to-sequence learning
- Neural network architectures
- Character-level sequence modeling
- Grapheme and phoneme representation
- IPA and Unicode text processing
- Neural text generation
- Evaluation of sequence prediction models

## Academic Context

Originally developed as part of a university assignment in Statistical Natural Language Processing (SNLP).

The repository has been organized as a portfolio project to demonstrate practical experience with NLP, sequence transduction, and neural machine learning.

---

**Technologies:** Python · NLP · Deep Learning · Sequence-to-Sequence Learning · G2P · IPA
