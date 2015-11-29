Hidden-Unit Conditional Random Fields
=====================================

This package contains Matlab implementations of various training algorithms for linear-chain CRFs and hidden-unit CRFs. It contains code to test these algorithms on four different data sets. Installation instructions are below. This code belongs to the paper:
- L.J.P. van der Maaten, M. Welling, and L.K. Saul. Hidden-Unit Conditional Random Fields. To appear in Proceedings of the International Conference on Artificial Intelligence & Statistics (AI-STATS), 2011.


Version information
-------------------
Version: 0.1b
Release date: March 9, 2011
Requirements: Matlab 7.8 or newer
Webpage: http://cseweb.ucsd.edu/~lvdmaaten/hucrf
Copyright: Laurens van der Maaten, University of California San Diego / Delft University of Technology, 2011


Legal
-----
Code provided by Laurens van der Maaten, 2011. The author of this code do not take any responsibility for damage that is the result from bugs in the provided code. This code can be used for non-commercial purposes only. Please contact the author if you would like to use this code commercially.


Installation
------------
1) Download the ZIP-file with Matlab code from the website.
2) Extract the ZIP-file with Matlab code.
3) Download the OCR data set ZIP-file from the website. Extract the ZIP-file.
4) Copy the OCR data file into the /data folder inside the Matlab code folder.
5) Run the OCR_EXPERIMENT function to train and test the hidden-unit conditional random field using 10-fold cross-validation.


Example
-------
The general syntaxis of the OCR_EXPERIMENT function is:

	err = ocr_experiment(type, no_hidden, lambda, rho); 

Herein, type indicates the CRF type, no_hidden indicates the number of hidden units (if any), lambda indicates the L2-regularization parameter, and rho indicates the margin value for perceptron training. Possible values for type are:

 - 'linear_crf':	Trains linear-chain CRF using L-BFGS; no rho; no no_hidden
 - 'linear_crf_sgd':	Trains linear-chain CRF using SGD; no rho; no no_hidden
 - 'perceptron':	Trains linear-chain CRF using (large-margin) perceptron training; no lambda; no no_hidden
 - 'hidden_crf':	Trains hidden-unit CRF using L-BFGS; no rho
 - 'hidden_crf_sgd':	Trains hidden-unit CRF using SGD; no rho
 - 'hidden_perceptron':	Trains hidden-unit CRF using perceptron training; no lambda

To perform a 10-fold cross-validation experiment with a hidden-unit CRF with 50 hidden units, using large-margin perceptron training with rho = 0.05, execute the following command:

	err = ocr_experiment('hidden_perceptron', 50, 0, 0.05); 

The same syntaxis applies to the FAQ_EXPERIMENT, CB513_EXPERIMENT, and TREEBANK_EXPERIMENT functions.


Tricks
------
- For discrete features, it is faster to use the MEX-implementations that come with the Matlab code. By running the MEXALL function, these MEX-implementations can be used. For continuous features, however, the Matlab code is usually faster, and it is thus better to remove all *.mex* files.


Bugs / Problems / questions
---------------------------
Feel free to drop me a line at: lvdmaaten@gmail.com

