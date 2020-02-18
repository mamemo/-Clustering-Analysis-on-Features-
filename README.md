# Clustering Analysis on Features

This repo contains the code for perform clustering analysis on image-based features extracted from a CNN. 

## Prerequisites

This repo needs the csv file created with the feature values from the last convolutional layer of a trained CNN. The file has to contain a column and value for every feature plus the following columns:

* img_ids: Unique ID for every sample.
* predictions: The prediction given from the model.
* labels: The real label of the sample.
* phase: Train, if it was used on training the model, Test, otherwise.

## Running the tests

The project is divided on the two datasets used on the paper. But both folders have similar files.

### Files and purpose

* acc_all.py/err_all.py: Calculates the accuracy of error of the whole dataset.
* fuzzy_kmeans.py: Defines the clustering algorithm used for the paper. Same structure has to be followed if the algorithm wants to be changed.
* metrics.py: Definition of some unsupervised clustering metrics to decide which algorithm is the best.
* pca_data.py: Create the principal components to be fed into the clustering.
* main.py: Entry point of the program. It uses different sample sizes and k-fold to validate experiments.

## Authors

* **Mauro Mendez** - *Code and First Author* - [Github](https://github.com/mamemo/)
* **Saul Calderon**
* **Pascal Tyrrell**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
