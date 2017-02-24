# Visual Segmentation Of Chromosomal Preparations
In cytogenetics, experiments typically starts from chromosomal preparations fixed on glass slides. Occasionally a chromosome can fall on another one, yielding overlapping chromosomes in the image. Before computers and images processing with photography, chromosomes were cut from a paper picture and then classified (at least two paper pictures were required when chromosomes are overlapping). More recently, automatic segmentation methods were developped to overcome this problem. Most of the time these methods rely on a geometric analysis of the chromosome contour and require some human intervention when partial overlap occurs. Modern deep learning techniques have the potential to provide a more reliable, fully-automated solution.

## Dataset
- 13434 pairs of 94x93 images (grey+labels) stored in a numpy array saved in a hdf5 file.
- The dataset and description is available from [Kaggle](https://www.kaggle.com/jeanpat/overlapping-chromosomes)

## Preprocessing
- Image is padded to a size of 96x96 to make upsampling easier
- Pixel values are normalised by dividing by 255

## Model
The model uses an encoder-decoder architecture similar [SegNet](https://arxiv.org/pdf/1511.00561v2.pdf).
- Conv-BatchNormalisation-ReLU-Maxpooling x 3 --> Upsampling-Conv-BatchNormalisation-ReLU x 3 --> 1x1 Conv
