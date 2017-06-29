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
