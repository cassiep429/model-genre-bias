# model-genre-bias
COMP 599 Final Project

## Getting Masked Language Model (MLM) outputs
### mlm_outputs.ipynb

To obtain examples of the outputs from the MLM task for each model, we built the mlm_outputs.ipynb script. It was initially run in a Jupyter notebook. It allows to easily test the models and obtain their outputs for different MLM task items. Some of our test items were taken from the dev.json file. 

When running the mlm_outputs.ipynb script in a Jupyter notebook, it will require you to indicate the local directory of the model. To change the masked sentence the model is tested on, you simply need to change the string given as input to _fill_mask()_; make sure the masked word is replaced by _<mask>_. 

  
