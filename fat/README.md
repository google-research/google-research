# (FAT) Fact Augmented Text

This was [Vidhisha Balachandran](https://github.com/vidhishanair)'s summer 2019
research intern project. It is no longer being used.

## Answering Natural Questions with Background Knowledge

Natural Questions (NQ) is a newly-release question-answering (QA) dataset. We
will focus on the problem of extracting short-answer questions from passages.
Many of these answers can also be supported with external knowledge graphs
(KGs), and we conjecture that state-of-the-art QA systems can be improved by
aligning external KG triples with the passage text. The summer project is to (1)
use the SLING entity-linking system and Wikidata repository, combined with RWR
on the Wikidata KG, to find KG facts that are loosely-aligned with the text in
the NQ passages, or potentially a larger corpora (2) fine-tune the BERT-based
masked language model on the fact-aligned text and (3) further fine-tune that
model as part of a short-answer QA model for NQ.
