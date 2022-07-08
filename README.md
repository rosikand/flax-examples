# flax-examples
> Minimum viable examples of machine learning models in Jax + Flax.

The goal of this project is to produce minimum viable examples for common machine learning models in jax + flax. By minimum viable, we mean a very minimal implementation of the model just to show a working example. The data used in each model is a mini version of the fashion MNIST dataset. 

## Models 

- **MLP image classifier**: `mlp.ipynb`
- **Autoencoder**: `autoencoder.ipynb`

Note: going to add a subfolder called `cool-jax-concepts` which will be atomic notebooks displaying a specific jax concept in each one. Might add docs that look like [this](https://gobyexample.com/): "Jax by example" or "Jax in $n$ steps"... 

Some ML topics I want to cover: 
- Meta-learning on a vision classification task 
- DDPM 
- Neural differential equations 
- Graph neural networks
- Self-supervised learning for vision 

Some jax concepts I want to cover: 
- `pmap` and where to use it with flax (i.e., distributing data and model params/state + distributed update function) 
- Train state in flax 
