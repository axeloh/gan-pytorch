
#### Inspiration:
- Original GAN paper: [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) by Goodfellow et. al.
- WGAN-GP is based on [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) by Gulrajani et. al., with an architecture for CIFAR-10 based on [Spectral Normalization For Generative Adversarial Networks](https://arxiv.org/pdf/1802.05957.pdf) by Miyato et. al.
- BiGAN is based on [Adversarial Feature Learning](https://arxiv.org/pdf/1605.09782.pdf) by Donahue et. al.
- CycleGAN is based on [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf) by Zhu et. al.

Models by no means serve to reproduce the original results in the papers.

# GANs in general
From the abstract of [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661):

> We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model *G*
that captures the data distribution, and a discriminative model *D* that estimates
the probability that a sample came from the training data rather than *G*. The training procedure for *G* is to maximize the probability of *D* making a mistake. This
framework corresponds to a minimax two-player game. In the space of arbitrary
functions *G* and *D*, a unique solution exists, with *G* recovering the training data
distribution and *D* equal to 0.5 everywhere.

To learn the generator's distribution *p<sub>g</sub>* over data **x**, we define a prior on input noise variables *p<sub>z</sub>(**z**)*, then represent a mapping to data space as *G(z; <img src="https://i.imgur.com/Z17Vu5N.png" width="10"/><sub>g</sub>)*. We also define *D(x; <img src="https://i.imgur.com/Z17Vu5N.png" width="10"/><sub>d</sub>)* that outputs a single scalar. *D(**x**)* represents the probability that **x** came from the data rather than *p<sub>g</sub>*. We train *D* to maximize the probability of assigning the correct label to both training examples and samples from *G*. We simultaneously train *G* to minimize log(1 - *D(G(**z**))*). In other words, *D* and *G* play the following two-player minimax game with value function *V(D, G)* :

<img src="https://i.imgur.com/luTBzcR.png" alt="gan_minmax" width="500"/>

## GAN on 1D toy data
First, the original minmax GAN objective (above) is used to train the generator and the discriminator. Both the generator and discriminator were modeled using an MLP.
Often, in practice, when using the original minmax objective, the generator suffers from saturation. What this means is that if the generator cannot learn as quickly as the discriminator, the discriminator wins, the game ends, and the model cannot be trained effectively. This is because early in training, when G is poor, D can reject samples with high confidence and the loss function will not provide sufficient gradients for G to learn well. To overcome this it is common to use a modification to the generator loss, know as the non-saturing formulation of the GAN objective. It is a subtle change that involves the generator to maximize the probability of images being predicted as real, instead of minimizing the probability of images being predicted as fake.

Formally, the loss for the generator L<sup>(G)</sup> and the discriminator L<sup>(D)</sup> is now the following:

<img src="https://i.imgur.com/69NGkI9.png" width="300"/>

<img src="https://i.imgur.com/jrB0U8F.png" width="170"/>

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
In general, GANs can be very hard to train, much because of the convergence properties of the value function being
optimized. [Arjovsky & Bottou](https://arxiv.org/abs/1701.04862) proposes using Wasserstein distance to produce a value function which has better theoretical properties than the original.
Wasserstein distance is a measure of distance between two distributions. It is also called Earth Mover's Distance because it informally can be interpreted as the minimum energy cost of moving and transforming a pile of dirt in the shape of one probability distribution to the shape of the other distribution. 
WGAN requires that the discriminator must lie within the space of 1-Lipschitz functions, which the authors enforce through weight clipping. This clipping has later been shown to lead to undesirable behaviour. [Gulrajani et. al.](https://arxiv.org/abs/1704.00028) proposes using  *Gradient Penalty* (GP) instead, and shows that it does not suffer the same problems. This resulted in the new GAN called WGAN-GP.

We use the CIFAR-10 architecture from the [SN-GAN paper](#inspiration) , with <img src="https://i.imgur.com/z9uM38O.png" width="70"/>, <img src="https://i.imgur.com/xsAtngp.png" width="110"/>. Instead of upsampling via transposed convolutions and downsampling via pooling or striding, we use DepthToSpace and SpaceToDepth methods (see repo) for changing the spatial configuration of the hidden states. 
We use the Adam optimizer with <img src="https://i.imgur.com/8zrB5hO.png" width="270"/>, a batch size of 256 and 128 filters within the ResBlocks. Model was trained for approximately 40,000 gradient steps, with the learning rate linearly annealed to 0 over the course of training. 

#### Results
The model was trained on the CIFAR-10 dataset. Below are samples generated after 30 and 230 epochs, respectively. 
It got an Inception Score of 8.042 out of 10.  

After 30 epoch | After 230 epochs 
:--- | :--- 
![](/wgan-gp/results/samples_epochs30.png) | ![](/wgan-gp/results/samples_epochs230.png)



## Representation Learning with BiGAN on MNIST
In BiGAN, in addition to training a generator *G* and a discriminator *D*, we train an encoder *E* that maps from real images *x* to latent codes *z*. The discriminator must now learn to jointly identify fake *z*, fake *x*, and paired *(x,z)* that don't belong together. In the original [BiGAN paper](#inspiration), they prove that the optimal *E* learns to invert the generative mapping <img src="https://i.imgur.com/qvVgI24.png" width="70"/>. Our overall minimax term is now: 

<img src="https://i.imgur.com/GyR6Rxn.png" width="600"/>

##### Architecture
We closely follow the MNIST architecture outlined in the original BiGAN paper, with one modification: instead of 
<img src="https://i.imgur.com/cc07tEg.png" width="130"/>, we use <img src="https://i.imgur.com/U2FQC6H.png" width="80"/>, with <img src="https://i.imgur.com/Xa7dRkH.png" width="60"/>.

##### Hyperparameters
We make several modifications to what is listed in the BiGAN paper. We apply l2 weight decay to all weights and decay the step size 𝛼 linearly to 0 over the course of training. Weights are initialized via the default PyTorch manner.


##### Testing the representations
We want to see how good a linear classifier *L* we can learn such that <img src="https://i.imgur.com/yW52Q60.png" width="90"/>, where *y* is the appropriate label. We fix *E* and learn a weight matrix *W* such that the linear classifier is composed of passing *x* through *E*, then multiplying by *W*, then applying a softmax nonlinearity. This is trained via gradient descent with cross-entropy loss.

As a baseline, we randomly initialize another netowrk 


#### Samples  


##### Reconstructions
We take the first 20 images from the MNIST training set and display the reconstructions 



## CycleGAN on MNIST and SVHN/Colored MNIST
TODO



