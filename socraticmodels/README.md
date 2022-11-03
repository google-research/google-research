# Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language

Project website: https://socraticmodels.github.io/

**Abstract** Large pretrained (e.g., “foundation”) models exhibit distinct capabilities depending on the domain of data they are trained on. While these domains are generic, they may only barely overlap. For example, visual-language models (VLMs) are trained on Internet-scale image captions, but large language models (LMs) are further trained on Internet-scale text with no images (e.g., spreadsheets, SAT questions, code). As a result, these models store different forms of commonsense knowledge across different domains.  In this work, we show that this diversity is symbiotic, and can be leveraged through Socratic Models (SMs):  a modular framework in which multiple pretrained models may be composed zero-shot i.e., via multimodal-informed prompting, to exchange information with each other and capture new multimodal capabilities, without requiring finetuning. With minimal engineering, SMs are not only competitive with state-of-the-art zero-shot image captioning and video-to-text retrieval, but also enable new applications such as (i) answering free-form questions about egocentric video, (ii) engaging in multimodal assistive dialogue with people (e.g., for cooking recipes) by interfacing with external APIs and databases (e.g., web search), and (iii) robot perception and planning.

## Instructions

This repository provides self-contained colabs with prototype implementations of Socratic Models for various applications.
