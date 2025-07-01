[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/4bhnZrOw)
# Assignment 4: Sequence transduction

**Deadline:** Jul 25, 2025

The problem we are trying to solve in this assignment is 
_grapheme to phoneme conversion_ (G2P).
The task is predicting the phonemic transcription
of a given (orthographic) word.
For example for the word `computer`,
we want to obtain a phonemic form `kəmpjutɚ` represented in IPA.

As before, you can find the data in your repository as two separate files.
The file [nlp2-a4.train](nlp2-a4.train) contains the training instances,
and the file [nlp2-a4.test](nlp2-a4.test) contains the test instances.
Each line in the training file contains
orthographic and phonemic representations
of a word separated by a tab character.
The phonemic representations are tokenized.
That is, some of the _segments_ are multiple IPA symbols
and/or Unicode characters.
Each segment in the phonetic form is separated by a space.
The test file only has a word (grapheme) per line.
The data comes from an earlier
[SIGMORPHON](https://sigmorphon.github.io/) shared task.

The expected output is similar to the training file.
Each line should contain an orthographic word, a tab, and the
corresponding prediction (phoneme sequence).
You are not required to keep segment boundaries in the phoneme sequence.
For example, both of the following lines are acceptable
for as a prediction for word "language" (you should include only one):

```
language	l æ ŋ ɡ w ɪ d͡ʒ
language	læŋɡwɪd͡ʒ
```

You are allowed to use any additional unlabeled / unsegmented data.
However, you are not allowed to use any data set particularly
prepared for any G2P task.

Note that although simplified (shorter sequences, fewer units),
the task is essentially the same as common tasks like
machine translation, summarization or paraphrasing.
As a result typical approaches to the task involve
encoder-decoder neural networks.

The evaluation will be based on exact match,
and average Levenshtein distance.

For submissions, follow the same procedure as the previous assignments.
Your prediction and source file names should follow the convention,
`a4-<method>.predictions`, and `a4-<method>.py`, respectively.
If you use [OpenNMT](https://opennmt.net/) for this task,
submit your configuration file with the same convention
(with `.yaml` extension). In this case, all files referenced
from the configuration file should also be pushed to the repository.

Similar to assignment 3, we do not provide a baseline.
You are strongly recommended to make use of the lab sessions
where example solutions (for similar tasks) will be provided,
and you can ask questions about or discuss your solutions.
