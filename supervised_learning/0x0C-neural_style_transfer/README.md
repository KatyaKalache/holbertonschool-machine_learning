## Description
Neural Style Transfer

![sf_houses](https://i.imgur.com/dgL140s.jpg?1)
![keith haring](https://i.imgur.com/21vw6dK.jpg?1)
![generated](https://i.imgur.com/onnS4Wg.png?1)

## Dependencies
* Tensorflow
* NumPy
* Vgg19

## Process
0. Upload and resize both content and style images to equal shape
1. Load VGG19 Keras as base model
2. Adding method that calculates gram matricies
3. Extracting the features used to calculate neural style cost
4. Calculating the style, content and total costs for generated image
5. Computing the gradients for the generated image
6. Generating  the neural style transfered image

Ekaterina Kalache: [github account](https://github.com/KatyaKalache), [twitter](https://twitter.com/KatyaKalache)
