# Computer Vision Assignment - Deep Learning on Food101
In this report, t the results of the experiments using three different categories of neural
networks on the Food101 dataset: a basic CNN, an all-convolutional net, and a model
with regularization, are shown. I used Vanilla PyTorch for implementing the models and
trained them for up to 5 epochs. Tqdm was
used to display the progress as the model was being trained. After each epoch, training
and test loss, as well as training and test accuracies, were reported.


All the training was done using the following global parameters:
`````
NUM_EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
HIDDEN_LAYER_SIZE = 101
`````

## Dependencies
`````
torchvision
pytorch
tqdm
`````

## Models & Dataset Used
The dataset used in this project is the Food-101 dataset, which can be found [here](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/). The dataset is split into a training set and a test set. The training set consists of 75,750 images, while the test set has 25,250 images. Each class has 1,000 images, with 750 used for training and 250 used for testing.

The following models were tested:

- Basic CNN (ConvFC)
- All Convolutional Net
- Basic CNN with Regularization (Data Augmentation)
- Transfer Learning (ResNet-34)



## BasicCNN
The given model, ConvFC, consists of two main blocks: a convolutional block and a fully
connected block.
1. The convolutional block (conv_block_1) includes the following layers:
- Conv2D layer with 3 input channels, 8 output channels, a kernel size of (3, 3), stride
(1, 1), and padding (1, 1).
- ReLU activation function.
- Conv2D layer with 8 input channels, 16 output channels, a kernel size of (3, 3),
stride (1, 1), and padding (1, 1).
- ReLU activation function.
- MaxPool2d layer with a kernel size of 2 and stride of 2.
- Conv2D layer with 16 input channels, 32 output channels, a kernel size of (3, 3),
stride (1, 1), and padding (1, 1).
- ReLU activation function.
- Conv2D layer with 32 input channels, 64 output channels, a kernel size of (3, 3),
stride (1, 1), and padding (1, 1).
- ReLU activation function.
- MaxPool2d layer with a kernel size of 2 and stride of 2.
2. The fully connected block (fc_block) includes the following layers:
- Flatten layer, which flattens the input tensor along dimensions 1 to -1.
- Linear layer with 16,384 input features and 101 output features.
- ReLU activation function.
- Linear layer with 101 input features and 101 output features.
- ReLU activation function.
- Linear layer with 101 input features and 101 output features.
- ReLU activation function.
- Linear layer with 101 input features and 101 output features.

## Contact
If you have any questions or feedback, please feel free to contact the author.
