Many machine learning models involve discrete variables. For instance, the latent variable of generative models are sometimes discrete. And hard version of attention mechanism takes the hidden attention at each step as discrete stochastic variable. In this case, the backpropagation is not directly applicable to the stochastic networks, which makes it very difficult to train such networks. This blog summarizes several methods to deal with the gradient propagation problem, from the straight-through estimator (2013) to Gumbel-softmax (2016).

Problem defination
-------
Consider a simple stochastic model with discrete random variable $$x$$ whose probability is given by $$p_\theta(x)$$, and a loss function $$f(x)$$. The objective of training is to minimize the expected lost $$L(\theta)={E}_{p_\theta(x)}(f(x))$$. There are roughly three kinds of methods to deal with this issue. They are introduced in the following.

Straight-through estimator
-------
This method is proposed by [Bengio, 2013](https://arxiv.org/pdf/1308.3432.pdf). The idea behind straight-through estimator is to backpropagate through the thresholding function as if it were the identity function. When the stochastic variables are binary, the estimator is simply $$f(s)p'(s)$$, where $$s$$ is sampled by $$s=1_{z_i>p_\theta(x)},z\sim[0,1]$$. This estimator is biased but has a low variance. For more information, I direct you to [here](http://r2rt.com/binary-stochastic-neurons-in-tensorflow.html).

Likelihood-ratio estimator
-------
Likelihood-ratio estimator (also known as score-function based estimator or REINFORCE) plays an important part in dealing with discrete variables. The objective function implies that the distribution of latent variable $$x$$ can be regarded as a policy network in reinforcement learning problems with loss function corresponding to the reward signal. Hence the policy gradient method, especially REINFORECE algorithm, was a popular method to be adopted. The estimator can be given as:

$$E_{p_\theta(x)}f(x) \nabla_\theta \log p_\theta(x).$$

However, it is well known that this method has a problem of high variance. Hence many variance reduction techniques are put forward.

- **Baseline methods**
	- **Centering the learning signal** was proposed in [NVIL](https://arxiv.org/pdf/1402.0030.pdf), which uses the moving averaging as the baseline, i.e., $$E_{p_\theta(x)}(f(x)-b) \nabla_\theta \log p_\theta(x).$$
	- **Input-dependent baseline** was also used in [NVIL](https://arxiv.org/pdf/1402.0030.pdf), which substracts from the learning siginal a predicted reward, i.e., $$E_{p_\theta(x \mid x_0)}[f(x, x_0) -b-\phi(x_0)]\nabla_\theta \log p_\theta(x \mid x_0).$$
	- [**VIMCO**](https://arxiv.org/pdf/1602.06725.pdf) samples multiple instances and uses the mean of other samples $$b=1/m\sum_{j\ne i} f(x_j)$$ as the baseline.
- **Variance normalization** keeps track of the moving averaging of the signal variance $$v$$, and divides the learning signal by $$\max(1, \sqrt{v})$$. This does not correspond to a baseline, and is a type of adaptive gradient.
- **Baseline variant** is a method similar with baseline method. However, it substract from the loss function a term that will affect the expected gradient.
	- [**DARN**](https://arxiv.org/pdf/1310.8499.pdf) uses $$b=f(\hat{x}) + f'(\hat{x})(x-\hat{x})$$, where the baseline corresponds to the first-order Taylor approximation of $$f(x)$$ from $$f(\hat{x})$$. The solution depends on the shape of $$f$$. If $$f$$ is a linear function, any $$\hat{x}$$ can be used. If $$f$$ is a quadratic function, $$\hat{x}$$ has to be $$0.5$$ w.r.t. Bernoulli variables.
	- [**MuProp**](https://arxiv.org/pdf/1511.05176.pdf) also models the baseline as a first-order Taylor expansion: $$b=f(\hat{x}) + f'(\hat{x})(x-\hat{x})$$ and $$\mu_b=f'(\hat{x})\nabla_\theta E_x[x]$$. To overcome backpropagation through discrete sampling, a mean-field approximation is used.

Gumbel-softmax
-------
The Gumbel softmax was recently used in [Jang, 2016](https://arxiv.org/pdf/1611.01144.pdf) and [Maddison, 2016](https://arxiv.org/pdf/1611.00712.pdf) to propagate through discrete variables. It is a 'reparameterization trick' for the categorical distribution. More specifically, it is actually a re-parameterization trick for a distribution that we can smoothly deform into the categorical distribution. Refer to [Gumbel-softmax tutorial](http://blog.evjang.com/2016/11/tutorial-categorical-variational.html).
