# Context-aware Multilingual ERC

This repository contains the codes and datasets for project ""A Framework for Context-Aware Multilingual Emotion Recognition with Transformers".

## Introduction
This project presents an original framework in advanced multilingual machine learning models to evaluate the accuracy of emotion recognition, focusing on Spanish translation, large language models (LLMs), and pre-trained transformers within an in-context learning paradigm. This research uses deep learning to translate text into Spanish and
develops a transformer model that captures both historical and future context. The evaluation
also incorporates the GPT-4o API with in-context learning to improve dialogue understanding.

## Content

- **run.py** : Main function to run the codes.
- **arguments.py** : Containing all arguments we use.
- **datasets** : Containing four ERC datasets, including IEMOCAP and MELD.
- **models** : Containing simple implementations of models.
- **utils** : Containing tools related to data access and processing.
- **scripts** : Providing some scripts for training and testing, and some checkpoints for quick start.
- **roberta-base** : To run the codes, one should download the pre-trained RoBERTa models from [huggingface](https://huggingface.co/models).
- **roberta-base-bne** : (trained with Spanish corpus). To run the codes, one should download the pre-trained RoBERTa models from [huggingface](https://huggingface.co/models).
