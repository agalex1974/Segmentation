# Segmentation

Basically all I did is to take as a base the splendid code of segnet https://github.com/hszhao/semseg and extend it also with an implementation of 
DeepLab V3. The basically add on I offered to deeplab is also to use the auxuliary loss function of pspnet. 
My aim is for use with Google Colab. I expect in the future that Google Colab or a name it will be tranformed into or spawn will offer services for training.
Right now Colab as in 11/06/2020 Colab restrict the resources to short running periods, this means that you can only do evaluation of your neural network.
13/09/2020: Google updated their GPUs to Volta 100 Graphics cards. Google drive offer sufficient bandwidth speed for the images to be processed quickly not becoming a bottleneck. This means you can train quite compex neural networks.
