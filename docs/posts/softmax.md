Recent years have seen the big progress in making training neural network effecient.
One research topic in practice is to reduce the complexity in optimizing the softmax logits.
It is really important in natural language processing as the size of output vocabulary can be very large (usually > 10K).

To calculate the objective function that is a function of output probability, we have to compute the logits (probability) of each words, even if only one word is required.
For instance, when a cross entropy loss function is used in estimating the likelihood, only the probability of target word is required.
However, since calculating the probabity needs to sum over all the logits for the partition function, all logits should be computed simultaneously.

$$ p(w_i) = \frac{\exp(l_i)}{\sum_j \exp(l_j)} ,$$

$$ C(p, t) = \log p(w_t) ,$$

where the $$l_j$$ is the logit of $$j$$th word.

To estimate the likelihood of a sentence, we have to compute the logits for each word at each timestep, which will be very computationally expensive.

Therefore, several techniques are proposed to remedy this problem.
The most famous algorithm maybe the Sampled Softmax algorithm, which is introduced in the [Tensorflow webpage](https://www.tensorflow.org/tutorials/seq2seq).
In my own experiments, if the sampled softmax is used, the runtime in a single batch will be greatly decreased from 1s per batch to 0.2s per batch.
This is indeed very effecient if the size of vocabulary is larger than 10K.
To fully understand the algorithm, I direct you to the [original paper](https://arxiv.org/abs/1412.2007).

Here I want to give a berief introduction.
The idea of Sampled softmax is to reduce the computation complexity by only sampling several logits from another distribution.
The logits that are not sampled will remain unknown.
Therefore the partition function $$\sum_j \exp(l_j)$$ can not be inferred.
Instead of computing the exact probability, sampled softmax turns to another objective.
Note that we actually know which logits are sampled, and which word is the target 
We can compute the posterior probability for each logit indicating whether it itself is the target word.
(This explanation is from the [Tensorflow manuscipt](https://www.tensorflow.org/extras/candidate_sampling.pdf), the paper explains more directly).
Formally, when $$|L|$$ words are sampled from another distribution $$Q$$, the target word is $$t$$, we combine $$L$$ and $$t$$ as $$C=L + {t}$$ and select one term $$y$$ from the pool, the posterior probabilty that this term is the true target is:
$$ p(t=y|C) = \frac{\exp(l_y)}{Q(y)} /K(C), $$
where the $$l_y$$ is the desired logit of $$y$$ and $$K(C)$$ is a term that is not relative to $$y$$.

This method is much similar with [Noise Contrastive Estimation](http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf) (NCE).
NCE differs sampled softmax in allowing mutiple $$y$$s.

In recent days, I encounter with a similar problem. In this problem, the objective function is replaced by a REINFORCE-Learning like function:
$$ L = \sum_j p(w_j) R_j ,$$
where $$R_j$$ is the given reward for predicting $$w_j$$.
And computing $$p(w_j)$$ will be very difficult since each logit is obtained via a matching model.
If I have 1000 categories, we have to do matching for 1000 times, which is not acceptable.
Therefore, I figured out a sampling based method to make this procedure efficient.

$$ \Delta = \sum_i q(w_i) R_i \nabla \log q(w_i) $$

$$ = \sum_i [q(w_i) R_i (\nabla l_i - \sum_k q(w_k) \nabla l_k)]$$

$$ = \sum_i [q(w_i) R_i - \sum_k q(w_i) R_k q(w_k)] \nabla l_i $$

$$ = \sum_i q(w_i) [R_i - \sum_k R_k q(w_k)] \nabla l_i$$

$$ = \sum_{w_i\sim q} [R_i - \sum_k R_k q(w_k)] \nabla l_i$$

$$ = \sum_{w_i \sim q} \frac{\sigma_j}{\sum_{j\in q} \sigma_j} [R_i - \sum_k R_k q(w_k)] \nabla l_i$$

$$ = \sum_{w_i \sim q} \frac{\sigma_j}{\sum_{j\in q} \sigma_j} [R_i - \frac{\sigma_j}{\sum_{j\in q} \sigma_j}  R_k q(w_k)] \nabla l_i$$

where $$\sigma_i = \exp(l_i)/q(w_i)$$.

By using twice importance sampling to approximate, we do not have to compute all the logits to obtain the exact probability.
I wonder there may be some papers talking about this method.
Please let me know if you have any idea.
