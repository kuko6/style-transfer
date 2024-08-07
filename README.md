# Style Transfer
Implementation of Style Transfer based on the **Adaptive Instance Normalisation (AdaIN)** proposed by Huang et al. [1]. Live demo is available on [Hugging Face Spaces](https://kuko6-style-transfer.hf.space).

## Method description
Overall, the model's architecture consists of: an encoder (first 4 layers from VGG-19), AdaIN layer and a decoder.

### AdaIN layer 
![Untitled](imgs/Untitled%202.png)

The AdaIN layer works by scaling the style ($y$) and content ($x$) image features acquired from the encoder. Specifically, AdaIN aligns the channel-wise mean and variance of the content features to the style features. The operation can be expressed as:

$$
AdaIN(x, y) = \sigma(y)(\frac{x-\mu(x)}{\sigma(x)})+\mu(y)
$$

The scaled feature maps from the AdaIN layer provide the input for the decoder which *generates* the final styled image ($t$).

### Loss function
The loss used for training of the decoder is a combination of content and style losses.

$$
L=L_C+\lambda L_s
$$

The content loss $L_C$ meassures the distance (difference) between the original content image and the generated image. It ensures that the content of the acquired image matches that of the original image. $L_C$ is defined as Mean Squared Error (MSE) between the features of the generated image and the scaled features from the AdaIN operation (which represents the original content).

$$
L_{C} = || f(t) - t ||^2
$$

On the other hand, the style loss $L_S$ ensures that the style of the generated image matches that of the input style image. It is computed as the sum of the distances between the mean values and the standard deviations of the outputs from the individual layers of the encoder.

$$
L_{S} = \sum_{j=1} || \mu(\phi_j(f(g))) - \mu(\phi_j(y)) ||^2 + \sum_{j=1} || \sigma(\phi_j(f(g))) - \sigma(\phi_j(y)) ||^2
$$

The $\lambda$ parameter adjusts the degree to which the style from the style image is transferred to the content image.

<!-- ![training graphs](imgs/training_graphs.png)

Evolution of the loss during training. -->

### Dataset
The training dataset consisted of style images acquired from the [WikiArt dataset](https://www.kaggle.com/competitions/painter-by-numbers/data) and content images from the [COCO dataset](https://cocodataset.org/#download).

### Results
![Untitled](imgs/Untitled%203.png)

<!-- ![Untitled](imgs/Untitled%206.png) -->

## Try it locally
set up a virtual environment with:
```sh
python3 -m venv venv
```

activate the virtual environment:
```sh
source venv/bin/activate
```

install the requirements:
```sh
pip install -r requirements.txt
```

run the gradio app:
```sh
python3 app.py
```

open `127.0.0.1:7860` in your browser.

## References

[1] Huang, Xun, and Serge Belongie. "Arbitrary style transfer in real-time with adaptive instance normalization." *Proceedings of the IEEE international conference on computer vision*. 2017.
