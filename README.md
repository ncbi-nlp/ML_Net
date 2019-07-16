# ML-NET
Reference codes repository for paper "ML-NET: multi-label classification of biomedical texts with deep neural networks". This repository demonstrates how to train and test
ML-NET on task3: diagnosis codes assignment in the paper.

ML-Net is a novel end-to-end deep learning framework for multi-label classification of biomedical tasks.
ML-Net combines the label prediction network with a label count prediction network,
which can determine the output labels based on both label confidence scores
and document context in an end-to-end manner.

## Key features
* Good performance on small to large scale multi-label biomedical text classification tasks
* Determine the output labels based on both label confidence scores and document context in an end-to-end manner
* Use [Threading and Queues](https://www.tensorflow.org/api_guides/python/threading_and_queues) in Tensorflow to faciliate faster training


## Requirements
ML-NET relies on Python 3.6, TensorFlow 1.8+.

## Scripts overview

```
aux_data_generation.py                  #generate auxiliary data for model training and testing
write_tf.py                             #generate training and test data (in both TFRecords and pickle format)
ML_Net.py                               #the model of ML_Net
ML_Net_components.py                    #the components of ML_Net
metrics.py                              #the evaluation metrics
data_utils.py                           #utilities functions of ML_Net
ML_Net_label_prediction_train.py        #the training of label prediction network
ML_Net_label_count_prediction_train     #the training of label count prediction network
```

## Data preparation
### Download dataset
The dataset is available here at: https://physionet.org/works/ICD9CodingofDischargeSummaries.
Please note that you have to acquire the access for MIMIC II Clinical Database project first. Please cite the
following paper when using the dataset:
```
@article{perotte2013diagnosis,
  title={Diagnosis code assignment: models and evaluation metrics},
  author={Perotte, Adler and Pivovarov, Rimma and Natarajan, Karthik and Weiskopf, Nicole and Wood, Frank and Elhadad, No{\'e}mie},
  journal={Journal of the American Medical Informatics Association},
  volume={21},
  number={2},
  pages={231--237},
  year={2013},
  publisher={BMJ Publishing Group}
}
```

### Clean and construct dataset
Please follow the readme file in the dataset folder and run construct_datasets.py. (Note: please run the script using Python 2).
The following files will be used in the training and evaluation of ML-NET:
```
MIMIC_FILTERED_DSUMS    #raw text of discharge summaries
testing_codes.data      #the labels (after augmentation) of testing set
training_codes.data     #the labels (after augmentation) of training set
```

### Generate auxiliary data
The training and testing of the ML-Net relies on the auxiliary data. Please
run aux_data_generation.py to generate the auxiliary data

### Generate traininig and test data
To gerenate the training and test dataset in tfrecords and pickle format, please run write_tf.py 

## Training
There are two training steps. We first train the label prediction network.
During training, the label prediction as well as the hierarchical attention network are updated through back propagation.
Then, we train the label count prediction network. However, different from the training label prediction network,
only the MLP part is updated as gradient descent stops at the layer of the document vector.

* first to run "ML_Net_label_prediction_training.py"
* then to run "ML_Net_label_count_prediction_train.py"

## Contact

Please contact Jingcheng Du: Jingcheng.du@uth.tmc.edu, if you have any questions

## Cite
Please cite the following [article](https://academic.oup.com/jamia/advance-article/doi/10.1093/jamia/ocz085/5522430), if the codes are useful for your project.
```
@article{10.1093/jamia/ocz085,
    author = {Du, Jingcheng and Chen, Qingyu and Peng, Yifan and Xiang, Yang and Tao, Cui and Lu, Zhiyong},
    title = "{ML-Net: multi-label classification of biomedical texts with deep neural networks}",
    journal = {Journal of the American Medical Informatics Association},
    year = {2019},
    month = {06},
    issn = {1527-974X},
    doi = {10.1093/jamia/ocz085},
    url = {https://doi.org/10.1093/jamia/ocz085},
    eprint = {http://oup.prod.sis.lan/jamia/advance-article-pdf/doi/10.1093/jamia/ocz085/28858839/ocz085.pdf},
}
```

