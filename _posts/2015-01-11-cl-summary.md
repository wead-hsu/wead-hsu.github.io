---
title: Computational Linguistics - A Summary
categories: NLP
---

## Segmenation

In English, splitting words or phrases is called "Tokenize", while in Chinese where the charactors are concatenated, the splitting is called "Segmentation".




- Table-based Segmentation Methods:
	* Maximal Matching Method
	* Reverse Maximal Matching Method

- How to evaluate the Segmentation Result?
	* Precision: the right segmentation / all segmentation in results
	* Recall: the right segmentation /  all segmentation in gold set
	* F-score: 2 * Precision * Recall / (Precision + Recall)

- Ambiguity:
	* Intersection: two words have two joint segmentation, e.g. ABC can be divided as AB/C and A/BC
	* Combination: one words can be divided or not, e.g. AB can be view as one single word or splitted as A and B
	* Joint: both Intersection and Combination
	* The length of Intersection Ambiguity: the maximal length with ambiguity.
	* False ambiguity: it is ambiguity without context but in the sentence, the ambiguity dispears.
	* How to find the ambiguity: using MM and RMM may help
	* How to deal with ambiguity: using a table or rules or statistical models
	* Several types of unknown word: person name, POI, organization
	* To recognize person name: probability model

## Language Model

In any languge, the probability of sentences are different. A language model is a model to approximate the probability of a given sentence.

N-gram language model: Conditional probability of $$w_i$$ given $$w_{i-1},..,w_{i-n+1}$$.

How to train a N-gram language model: maximal likelihood? Noise and sparsity.

Zipf's Law: frequency * rank = const

- How to smooth, i.e. deal with sparsity?
	* Add-one/delta method: add one/delta more count to any word.
	* Held-out Estimation: $$T_r/T*1/N_r$$ using the ratio in devset of words with same occurance in trnset to smooth the model
	* Deleted Estimation: two parts Held-out Estimation
	* Good-Turing Estimation: using high occurance words' ratio to smooth the low occrance words
	* Linear Interpolation: using different n-gram
	* Back-off model: if the high order n-gram is of high confidence, use it or use low order model

## Entropy in Language Model

* Entropy: $$H(x)$$ note the minus in the beginning.
* Joint Entropy: $$H(X,Y)$$
* Conditional Entropy: $$H(X\|Y)$$
* Entropy Rate: Entropy of a sentence / length of sentence
* Mutual Information: $$I(X,Y) = I(Y,X)$$
* Pointwise Mutual Information: $$I(x,y)$$
* Relative Entropy: $$D(p\|q)$$
* Cross Entropy: $$H(X,q)$$
* Perplexity: $$2^{CrossEntropy}$$

## Hidden Markov Model

- Three fundamental problem:  
	* Given a model and observation, how to calculate the probability of this observation? Vertebi algorithm with sum operation
	* Given a model and observation, how to decode the state sequence? Vertebi algorithm with max operation  
	* Given a data set, how to obtain the HMM to maximal the "joint" probability? EM algorithm

## Sequence Labelling:

- Three ways:
	* Rule based method: ..
	* Statistical method: hidden markov model: bigram -> trigram
	* Joint method: fix bugs using rule method

## Machine Learning in NLP

* Maximal Entropy Model: Maximize the entropy with same expectation of feature in dataset and model. Training algorithm: GIS, IIS, L-BFGS
* Conditional Markov Model (directional graph model): HMM => decision model, i.e. given $$o$$, cal $$p(s\|0)$$ => maximal entropy principle => labelling bias
* Conditional Random Fiekd (undirectional graph model): decision model => clique => maximal entropy principle => high training cost

## Parsing

- Earley: 
	* Complexity: $$O(n^3)$$
	* Scanner: if the word after the `.` is non-derminal and in pos table, scan.  
	* Predictor: if the word after the `.` is non-derminal and not in pos table, predicte.
	* Completer: if no word after the `.`, complete.
	* Everytime new chart is generated, expand states after predicting.
	* Do not delete preceding chart, add new state in following chart.
- LR:
	* Simple LR: reduction once it can be reduced.
	* Generative LR: 
		* operate points. If a new subtree is generated, remove leaves and add one point pointing to the root of subtree.
		* mutiple pathes. If the subtree has another parsing root and can be included in another root, split the parse tree.
- CKY: dymanic algorithm with 1/0.
- PCKY: dymanic algorithm with 0-1. 
	* Inside: standard DP.
	* Outside: reverse inside algorithm
	* Vertebi
	* EM to train via unsupervised method
	* Word-based PCKY

## Unification

* Feature Structure: key, value
* Reentrant Structure: share key
* Unification: if same key has same value
* Earley with unification

## Topic Model

- Linear Matrix Decomposition:
	* given a matrix $$C$$ whose column is a document and row is a word, $$C=U\Sigma V^T$$ where U is the feature vector of $$CC^T$$, V is the feature vector of $$C^TC$$ and $$\Sigma$$ is the eign-feature. All feature vector are normalized.
	* $$CC^T$$ and $$C^TC$$ have the same eign-feature.
	* $$CC^T=U\Sigma V^T*V\Sigma U^T=U\Sigma \Sigma ^TU^T$$ ..

- Probablisitic Topic Model:
	* PLSA: non-negative matrix decomposition, EM-trianing algorithm.
	* LDA: bayesian pLSA, EM, variational inference, gibbs sampling

## Translation Model

- Sentence-level Alignment
	* In Gales's method: length of sentences is the only one needs to be considered. Because $$argmax_A p(A\|LS,LT)=argmax_A p(LS,LT\|A)p(A)$$, we can find the most probable alignment if given $$p(LS,LT\|A)$$ and $$p(A)$$. A dymanic algorithm can accelerate the process.

- Word-level Alignment
	* IBM-I: every alignment has the same probability.
	* IBM-II: the alignment probability is the function of length and index in source sentence.
	* Vogel: HMM-based model, the alignment is the hidden state.
	* Fertility: one word corrosponds to several words.
	* Och: smooth
	* How to find the alignment? Dymanic algorithm or greedy.
	* How to train? EM algorithm.
