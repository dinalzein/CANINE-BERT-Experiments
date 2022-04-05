# BERT vs CANINE on NLP tasks

__This repository aims at presenting CANINE fine-tuning experiments and compare it to BERT on some NLP tasks based on the paper cited in the [Citation](#citation) section.__

Three Jupyter Notebooks are provided to reproduce the results that are presented in the [Results for SST-2](#Results-for-sst-2), [Results for SQuAD](#Results-for-SQuAD), and [Results for CoLA](#results-for-cola). They can be directly launched in Google Colab from here:

- <a href="https://colab.research.google.com/github/dinalzein/CANINE-BERT-Experiments/blob/main/SST2_experiments.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> for [SST-2](https://nlp.stanford.edu/sentiment/index.html) task.  

- <a href="https://colab.research.google.com/github/dinalzein/CANINE-BERT-Experiments/blob/main/CoLA_experiments.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> for [CoLA](https://nyu-mll.github.io/CoLA/) task.  


A [report](./report.pdf) has been made to give an overview of the paper cited in [Citation](#citation) and analyze the results of the experiments.


## Results for [SST-2](https://nlp.stanford.edu/sentiment/index.html)
For the accuracy results, larger is better.
The results below present a comparison between BERT and CANINE on the SST-2 task.
| Model           |Input     | Training Loss | Evaluation Loss | Accuracy
|---              |---       |---            |---    					 |---      
BERT         			| Subwords | 0.060         | 0.334           | 0.927
CANINE-S          | Subwords | 0.157         | 0.625           | 0.851
CANINE-C          | Chars    | 0.169         | 0.572           | 0.856

BERT outperforms both CANINE-S and CANINE-C by +0.076 and +0.071 accuracy respectively. CANINE-C improves over CANINE-S by a 0.005 --negligible-- accuracy.

## Results for [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
For the F1-score, larger is better.
The results below present a comparison between BERT and CANINE on SQuAD.
| Model           |Input     | Training Loss | Evaluation Loss | Exact Match |F1-score
|---              |---       |---            |---    					 |---          |---
BERT         			| Subwords | 0.773         | 1.157           | 76.746      | 85.134
CANINE-S          | Subwords | 0.629         | 1.461           | 72.526      | 82.183
CANINE-C          | Chars    | 0.664         | 1.355           | 72.375      | 82.300

BERT outperforms both CANINE-C and CANINE-S by +2.834 F1 and +2.951 F1 respectively

## Results for [CoLA](https://nyu-mll.github.io/CoLA/)
For the Matthes Correlation results (ranges between -1 and 1): 1 indicates perfect match, -1 is indicates perfect disagreement, and 0 indicates uninformed guessing.
The results below present a comparison between BERT and CANINE on the CoLA task.
| Model           |Input     | Training Loss | Evaluation Loss | Matthes Correlation
|---              |---       |---            |---    					 |---      
BERT         			| Subwords | 0.149         | 0.766           | 0.565
CANINE-S          | Subwords | 0.615	       | 0.638           | 0
CANINE-C          | Chars    | 0.567         | 0.668           | 0.064

Fine-tuning using CANINE shows an inefficient performance.



### Citation
To cite this work in your publications in BibTeX format:

```
@inproceedings{wolf2020transformers,
  title={Transformers: State-of-the-art natural language processing},
  author={Wolf, Thomas and Debut, Lysandre and Sanh, Victor and Chaumond, Julien and Delangue, Clement and Moi, Anthony and Cistac, Pierric and Rault, Tim and Louf, R{\'e}mi and Funtowicz, Morgan and others},
  booktitle={Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations},
  pages={38--45},
  year={2020}
}
```

```
@article{clark2021canine,
  title={Canine: Pre-training an efficient tokenization-free encoder for language representation},
  author={Clark, Jonathan H and Garrette, Dan and Turc, Iulia and Wieting, John},
  journal={arXiv preprint arXiv:2103.06874},
  year={2021}
}
```


### Used materials and 3rd party code
The experiments are based on the [*huggingface*](https://github.com/huggingface/transformers) library and the paper "Canine: Pre-training an efficient tokenization-free encoder for language representation" by Jonathan H. Clark, Dan Garrette, Iulia Turc, John Wieting [[paper]](https://arxiv.org/pdf/2103.06874.pdf).
