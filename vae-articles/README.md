# Variational Auto-Encoders in Torch

This repository contains Torch code corresponding to a series on articles on
[davidstutz.de](http://davidstutz.de/) discussing variational auto-encoders (VAEs)
[1] and their variants. Specifically, beyond the standard model, a denoising VAE [2]
and a Bernoulli VAE [3,4] is implemented.

    [1] D. P. Kingma, S. Mohamed, D. J. Rezende, and M. Welling. Semi-
        supervised learning with deep generative models. In Advances in
        Neural Information Processing Systems, pages 3581–3589, 2014.
    [2] D. J. Im, S. Ahn, R. Memisevic, and Y. Bengio. Denoising criterion
        for variational auto-encoding framework. In AAAI Conference on
        Artificial Intelligence, pages 2059–2065, 2017.
    [3] E. Jang, S. Gu, and B. Poole. Categorical reparameterization with
        gumbel-softmax. CoRR, abs/1611.01144, 2016.
    [4] C. J. Maddison, A. Mnih, and Y. W. Teh. The concrete distribu-
        tion: A continuous relaxation of discrete random variables. CoRR,
        abs/1611.00712, 2016.

## Requirements

It is recommended to install [torch/distro](https://github.com/torch/distro).

## Examples

The provided examples include:

* `vae.lua`: an implementation of the standard VAE;
* `conv_vae.lua`: a convolutional VAE;
* `denoising_vae.lua`: an implementation of a denoising VAE;
* `bernoulli_vae.lua`: an implementation of a Bernoulli VAE;

See the articles on [davidstutz.de](http://davidstutz.de/) for
an introduction and mathematical bacgrkound. The examples are further
commented well.

## License

Copyright (c) 2018 David Stutz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.