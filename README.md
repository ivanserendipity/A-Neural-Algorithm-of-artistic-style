# A neural algorithm of artistic style

This project is trying to implement the algorithm from the paper “A neural algorithm of artistic style(https://arxiv.org/abs/1508.06576) (Leon A. Gatys, Alexander S. Ecker, Matthias Bethge)”. The system uses neural representations to separate and recombine content and style of arbitrary images, providing a neural algorithm for the creation of artistic images. Basically, the idea can be described like this:

![](http://static.boredpanda.com/blog/wp-content/uploads/2015/08/computer-deep-learning-algorithm-painting-masters-12.jpg)
(Image from google, author unknown)

## Approach

### Algorithm
1.	Used CNN to catch the representation of images. The lower layer of CNN tends to represent more about the details, the       higher layer of CNN tends to represent more about the semantic information of the image.

2.	Defined a Gram matrix from one layer to represent the style. This gram matrix represents the correlation of two features.

3.  Then we have content and style representation, we can iteratively operate a noise image to optimize it so that its content is similar to the original content image and its style is similar to the original style image.

### Keras Implementation
In order to implement this algorithm, we need to do :
1. A trained neural network:
   Based on the research I did and the suggestion provided by the paper, I used VGG16 neural network to pretrain image. The VGG16 neural network has 13 layers CNN and 5 layers pool. In Keras 1.0 version it already has this network in Keras.Application model.
   
2. A white noise image for generating target image:
   Use methods from numpy library to generate a noise image.
   
3. Feature correaltions and loss functions(content loss, style loss,total loss)

4. A function to do gradient so that we can get the minimum loss value.
   Keras provides a function called K.gradients to calculate gradients of the generated image with regard to the loss. And in each iteration, it uses an optimization model from scipy to update image x. The optimized function here is called fmin_l_bfgs_b, it is a optimization to minimize a function and its algorithm is L-BFGS-B.
So in each iteration, we just need to do:
   1. Given input image x, get its gradient with regard to the loss function
   2. Update x

### Test Result
  In my experiment, I defined total variation weight as 8.5e-5, style weight as 1.0, content weight as 0.025. And run it for 150 iterations, and Each iteration takes around 300 seconds in my i5 CPU. At the beginning 20th iteration, the loss value decreases sharply, and as iteration increases the loss value decreases slowly and after 100th iteration it almost remains the same.
   
   
