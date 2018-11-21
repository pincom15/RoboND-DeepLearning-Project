## Project: Deep Learning Project

---

[//]: # (Image References)

[architecture]: ./misc_images/fcn_architecture.jpg
[train]: ./misc_images/training_curve.jpg
[result]: https://youtu.be/FZUxv9PI72o

### 1. Fully Convolutional Network
Fully convolutional network(FCN) is one of state of the art deep learning based algorithm to solve semantic segmentation problem in computer vision task. Unlike a typical convolutional neural network(CNN), FCN allows us to preserve spatial information with variable sized inputs. An FCN model consists of two parts; encoder and decoder. Fully Connected layer is replaced with 1x1 convolutional layer in order to preserve spatial information. To improve performance, an FCN uses skip layer fusion technique. Skip connections pass information that may be lost due to convolutions or encoding.

### 2. Network Architecture
![FCN architecture][architecture]
In this project, above FCN model is implemented to find a specific person in images from a simulated quad-copter. This model consists of three encoder layers, 1x1 convolutional layer, and three decoder layers. Encoder layers includes separable convolution layers. and decoder has various deconvolution layers. 

Encoder reduces to a deeper 1x1 convolution layer, which is replaced by the fully connected layers in CNN. This leads to preserving spacial information from the image. An encoder block includes a separable convolution layer, which performs depth-wise spatial convolution on each input channel and mix output channels via a 1x1 convolution. Then each output combines into an output layer. A separable convolution reduces parameters. As a result, it makes more efficient and better performance. An encoder block also includes batch normalization layer. It can adaptively normalize each layer's inputs even if the mean and variance of the values change during training. A batch normalization layer is typically used after a convolutional layer.

```python
output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides, padding='same', activation='relu')(input_layer)
    
output_layer = layers.BatchNormalization()(output_layer) 
```

One decoder block is comprised of three parts; a bilinear upsampling layer, skip connections and two additional separable deconvolution layers. Deconvolution learns the parameters for upsampling. When the features are reduced in dimensions, it is upsampled again to the image size by deconvolution in decoder. Skip connection concatenates the upsampled and input layers.

```python
def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer

def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # Upsample the small input layer using the bilinear_upsample() function.
    upsample = bilinear_upsample(small_ip_layer)
    
    # Concatenate the upsampled and large input layers using layers.concatenate
    concat = layers.concatenate([upsample, large_ip_layer])
    
    # Add some number of separable convolution layers
    output_layer1 = separable_conv2d_batchnorm(concat, filters, strides=1)
    output_layer2 = separable_conv2d_batchnorm(output_layer1, filters, strides=1)
    
    return output_layer2
```

```python
def fcn_model(inputs, num_classes):
    
    # Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    encode_layer1 = encoder_block(inputs, 32, 2)
    encode_layer2 = encoder_block(encode_layer1, 64, 2)
    encode_layer3 = encoder_block(encode_layer2, 128, 2)
    
    # Add 1x1 Convolution layer using conv2d_batchnorm().
    conv_layer = conv2d_batchnorm(encode_layer3, 256, 1, 1)
    
    # Add the same number of Decoder Blocks as the number of Encoder Blocks
    decode_layer3 = decoder_block(conv_layer, encode_layer2, 128)
    decode_layer2 = decoder_block(decode_layer3, encode_layer1, 64)
    decode_layer1 = decoder_block(decode_layer2, inputs, 32)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(decode_layer1)
```

### 3. Hyper Parameters

```python
learning_rate = 0.01
batch_size = 64
num_epochs = 10
steps_per_epoch = 200
validation_steps = 50
workers = 4
```

Hyper parameters are tuned by using heuristic method. I set the learning rate as 0.01. When I tried smaller rates, training requires more epochs and time. Increasing epochs might result in overfitting. So, I picked optimal value right before validation loss is about to increse. Below is my result of training and valication loss that I find optimum.
![Training Curves][train]

[Follow me project result video][result]

### 4. Fully Connected layer and 1x1 Convolutional layer
In FCN, fully Connected layer in typical CNN is replaced with 1x1 convolutional layer that preserves spatial information and results in a mini neural network over each filter. A 1x1 convolution layer reduces dimensionality. Thus, this reduces computation power. A fully connected layer of the same size would result in the same number of features.

### 5. Encoders and Decoders
The encoder is a series of convolutional layers like VGG and ResNet. The goal of the encoder is to extract features from the image. The decoder up-scales the output of the encoder. It results in segmentation or prediction of each individual pixel in the original image.

### 6. Model Limitations
Following other objects such as dog, cat, car and etc, is not possible since this model trained to follow a specific person. However, if we change training and valication set and re-train this model, we might be able to follow any objects as we trained.

### 7. Future Enhancements

First, pretrained networks such as VGG, and ResNet will reduce training time and increase accuracy.
Second, better hyper parameters might be chosen in a brute force way.
Lastly, more dataset which includes more angles and distances might improve accuracy.

