# Deep Neural Networks and Transfer Learning
 COMP8220 Machine Learning
 
Google Colaboratory was used to perform the tasks below.

## Packages Used
* os
* random
* numpy
* pandas
* seaborn
* matplotlib
* keras
* functools
* sklearn

## Building a Small CNN from Scratch

The first part of the notebook creates two deep neural networks from scratch. Both versions have two convolution layers and two maxpooling layers. One version contains dropout layers and the other does not.  

Both networks were trained on the Caltech-101 dataset, leaving out the 5 largest categories. The Caltech-101 dataset contains relatively small images only, small filters were used in the architecture. Small and local features on the images need to be identified so different kernel sizes for the convolution layer and the pool size for the maxpool layer were explored. Due to computational limitations, the hidden layers will only have 512 neurons. Grid search was used to determine the best dropout rate based on a list of possible rates.

## Apply Transfer Learning to a Pretrained Network

The second part of the notebook leverages on a pretrained network, ResNet50V2, accessed via keras pretrained models. The model was adapted and retrained on the Caltech-101 dataset. In the first retraining, all layers were kept frozen except the last one.  Retraining was performed again, unfreezing one extra layer. The results were compared against each other and also with the CNNs built from scratch above.

## Apply Data Augmentation to a Pretrained Network

The model was retrained again including some data augmentation and compared the results with those above.
