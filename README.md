# Sugar_Extraction
# Predicted Efficiency: Beer mash extraction efficiency

Module :
1. main_run.py -- > Train/Test using the SVR model
2. eda_testsheet.ipynb --> EDA ,Feature selection , Model comparison / best Model selection


Steps to follow :
1. clone the repo
2. pip install -r requirements.txt
3. for training : python main_run.py -check "train" -data_path "<csv file path>"
   or
   for testing  : python main_run.py -model_path "model" -check "test" -data_path "<csv file path>"
