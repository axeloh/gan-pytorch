
#### Models based on numerous papers:
- Original GAN paper: [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) by Goodfellow et. al.
- WGAN-GP is based on [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) by Gulrajani et. al., with an architecture for CIFAR-10 based on [Spectral Normalization For Generative Adversarial Networks](https://arxiv.org/pdf/1802.05957.pdf) by Miyato et. al.
- BiGAN is based on [Adversarial Feature Learning](https://arxiv.org/pdf/1605.09782.pdf) by Donahue et. al.
- CycleGAN is based on [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf) by Zhu et. al.

Models by no means serve to reproduce the original results in the papers.

# GANs in general
From the abstract of [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661):

> We propose a new framework for estimating generative models via an adversar-
ial process, in which we simultaneously train two models: a generative model *G*
that captures the data distribution, and a discriminative model *D* that estimates
the probability that a sample came from the training data rather than *G*. The train-
ing procedure for *G* is to maximize the probability of *D* making a mistake. This
framework corresponds to a minimax two-player game. In the space of arbitrary
functions *G* and *D*, a unique solution exists, with *G* recovering the training data
distribution and *D* equal to 0.5 everywhere.

To learn the generator's distribution *p<sub>g</sub>* over data **x**, we define a prior on input noise variables *p<sub>z</sub>(**z**)*, then represent a mapping to data space as *G(z; <img src="https://i.imgur.com/Z17Vu5N.png" width="10"/><sub>g</sub>)*. We also define *D(x; <img src="https://i.imgur.com/Z17Vu5N.png" width="10"/><sub>d</sub>)* that outputs a single scalar. *D(**x**)* represents the probability that **x** came from the data rather than *p<sub>g</sub>*. We train *D* to maximize the probability of assigning the correct label to both training examples and samples from *G*. We simultaneously train *G* to minimize log(1 - *D(G(**z**))*). In other words, *D* and *G* play the following two-player minimax game with value function *V(G, D)*:

<img src="https://i.imgur.com/luTBzcR.png" alt="gan_minmax" width="500"/>

## GAN on 1D toy data
First, the original minmax GAN objective (above) is used to train the generator and the discriminator. Both the generator and discriminator were modeled using an MLP.
Often, in practice, when using the original minmax objective, the generator suffers from saturation. What this means is that if the generator cannot learn as quickly as the discriminator, the discriminator wins, the game ends, and the model cannot be trained effectively. This is because early in training, when G is poor, D can reject samples with high confidence and the loss function will not provide sufficient gradients for G to learn well. To overcome this it is common to use a modification to the generator loss, know as the non-saturing formulation of the GAN objective. It is a subtle change that involves the generator to maximize the probability of images being predicted as real, instead of minimizing the probability of images being predicted as fake.

Formally, the losses for the generator L<sup>(G)</sup> and the discriminator L<sup>(D)</sup> is now the following:

<img src="https://i.imgur.com/kdmpbXH.png" width="300"/>

#### Results
The results show samples drawn from the generator after epoch 1, and after training is finished. 
*real* shows the distribution of real data, *fake* shows distribution of generated samples, *discrim* shows the output from the discriminator for each point. For the ideal generator, the discriminator cannot distinguish between real and generated samples, and thus outputs 0.5 everywhere. From the results we see that the generator in both cases has almost learned the data distribution perfectly. 

Using original GAN minmax objective:
After 1 epoch | After training 
:--- | :--- 
![](https://i.imgur.com/hSjlgy0.png) | ![](https://i.imgur.com/aQCLQ4f.png)

Using non-saturing formulation of GAN objective:
After 1 epoch | After training 
:--- | :--- 
![](https://i.imgur.com/8GFx7C0.png) | ![](https://i.imgur.com/XWkuqIz.png)



## WGAN-GP on CIFAR-10



## BiGAN on MNIST



## CycleGAN on MNIST and SVHN/Colored MNIST




