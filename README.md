# Teacher Assistant Selection for Knowledge Distillation

An academic research project investigating whether neural network representation similarity can be used to select effective Teacher Assistant models for Knowledge Distillation.

**Status:** Planning

## Overview

Knowledge Distillation (KD) transfers knowledge from a larger Teacher model to a smaller Student model. Teacher Assistant Knowledge Distillation (TAKD) introduces an intermediate model to bridge the gap between Teacher and Student.

This project investigates whether representation similarity between neural networks can help predict which Teacher Assistant will lead to better Student performance.

## Research Question

**Can neural network representation similarity predict the effectiveness of a Teacher Assistant in Knowledge Distillation?**

## Planned Methodology

1. Select Teacher, Student, and candidate Teacher Assistant architectures.
2. Establish baseline performance using direct Teacher-to-Student distillation.
3. Measure representation similarity between selected models.
4. Perform knowledge distillation using different Teacher Assistants.
5. Compare Student performance and examine the relationship between similarity scores and distillation outcomes.

## Similarity Measures

The following measures are being considered:

- **Centered Kernel Alignment (CKA)** – similarity between learned representations.
- **Feature similarity** – comparison of intermediate neural network features.
- **Logit similarity** – comparison of model outputs.
- **KL Divergence** – comparison of predictive probability distributions.

## Model Architectures

Initial experiments are planned around convolutional neural networks, with potential comparisons across different model families, including:

- ResNet
- VGG
- DenseNet

The final architecture selection and experimental configuration have not yet been established.

## Evaluation

The experiments aim to investigate:

- Student performance after knowledge distillation.
- Differences between Teacher Assistant candidates.
- The relationship between representation similarity and final Student performance.
- Whether similarity-based selection can outperform simple architecture- or size-based selection strategies.

## Tech Stack

**Planned:** Python, PyTorch, NumPy

Additional tools and dependencies will be documented as the implementation progresses.

## Current Status

The project is currently in the planning stage. Experiments and results will be added as development progresses.

## License

No license has been selected yet.
