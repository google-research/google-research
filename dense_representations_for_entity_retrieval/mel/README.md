# Mewsli Datasets

Welcome to the landing page of Mewsli! The name is short for *Multilingual
Entities in News, linked*.

This is a suite of publicly available datasets for academic research into
multilingual entity linking.

The basic task is to link an entity mention in the context of a WikiNews article
to a single correct entity in the WikiData knowledge base, typically by doing
retrieval over canonical textual representations of candidate entities. These
entity representations are short descriptions from the beginning of an entity's
Wikipedia page in a randomly chosen language, making for a highly cross-lingual
task.

Ground truth entity annotations are automatically derived from cross-wiki
hyperlink anchor text and their targets, as placed in WikiNews and Wikipedia
articles by human editors in the normal course of writing and editing.

### Example 1

-   **Snippet in an English article:**

    At a brief ceremony, Prime Minister Girija Prasad Koirala hoisted the
    **national flag** where previously only the royal flag had flown, and
    unveiled a plaque reading "Narayanhity National Museum".

-   **Entity description to retrieve (Ukrainian):**

    Прапор Непалу — один з офіційних символів Непалу. <br>
    *(translation: The flag of Nepal is one of the official symbols of Nepal.)*

-   **WikiData identifier**: Q159741

### Example 2

-   **Snippet in an Arabic article:**

    .الأربعاء، 12 أغسطس 2009 ثلاثة عشر شخصًا توفوا بعد تحطم طائرة خطوط بيه إن جى الجوية في **بابوا غينيا الجديدة**  <br>
    *(translation: Wednesday, August 12, 2009 Thirteen people have died after a PNG Airlines plane crashed in **Papua New Guinea**.)*

-   **Entity description to retrieve (Korean):**

    파푸아뉴기니 독립국, 약칭 파푸아뉴기니는 오세아니아의 나라이다. <br>
    *(translation: Independent State of Papua New Guinea, abbreviated as Papua New Guinea, is a country in Oceania.)*

-   **WikiData identifier**: Q691

## Editions

👉
**[Mewsli-9](https://github.com/google-research/google-research/blob/master/dense_representations_for_entity_retrieval/mel/mewsli-9.md)**
accompanies
*[Entity Linking in 100 Languages (Botha et al., 2020)](https://www.aclweb.org/anthology/2020.emnlp-main.630)*.
This edition provides for large-scale evaluation, with an emphasis on WikiData
entities that do not have English Wikipedia pages.

👉
**[Mewsli-X](https://github.com/google-research/google-research/blob/master/dense_representations_for_entity_retrieval/mel/mewsli-x.md)**
accompanies *[XTREME-R: Towards More Challenging and Nuanced Multilingual
Evaluation (Ruder et al.,
2021)](https://aclanthology.org/2021.emnlp-main.802.pdf)*. This edition includes
a full set of resources for reproducibly training and evaluating models. The
scale of evaluation is smaller, emphasizing zero-shot cross-lingual transfer,
zero-shot entity retrieval and accessibility.

-   Mewsli-X is one of the tasks in the
    **[XTREME-R benchmark suite & leaderboard](https://sites.research.google/xtremer)**
    &mdash; new submissions welcome!

### Comparison

|                                                                                        | **Mewsli-9**                                                               | **Mewsli-X**
| -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ------------
| **Salient features**                                                                   | Large-scale task. Special focus on entities absent from English Wikipedia. | Scaled down task for improved accessibility. Special focus on zero-shot.
| **Mention languages**                                                                  | (9) ar, de, en, es, fa, ja, *sr*, ta, tr                                   | (11) ar, de, en, es, fa, ja, *pl*, *ro*, ta, tr, *uk*
| **WikiNews evaluation instances**                                                      | 289,087 (no predefined splits)                                             | 17,615 (2,991 dev + 14,624 test)
| **Other released data**                                                                | None                                                                       | Candidate set (multilingual Wikipedia descriptions);<br> Fine-tuning train & dev (English Wikipedia mentions)
| ***Attributes***                                                                       |                                                                            |
| **Text tokenization**                                                                  | Not released                                                               | Sentence boundaries are released for the raw text, in support of token-free modeling research.
| **Noise filtering**                                                                    | Minimal                                                                    | Extensive
| **Controlled sampling**                                                                | None                                                                       | WikiNews instances approx. balanced by language and global entity frequency
| ***Entity candidate set***                                                             |                                                                            |
| **Description languages**                                                              | (104) All mBERT-languages                                                  | (50) All XTREME-R languages
| **Size**                                                                               | 20M                                                                        | 1M
| **Has nuisance entities associated with Wikipedia 'list' and 'disambiguation' pages?** | yes                                                                        | no

## Disclaimer

This is not an official Google product.

<!--
--->
