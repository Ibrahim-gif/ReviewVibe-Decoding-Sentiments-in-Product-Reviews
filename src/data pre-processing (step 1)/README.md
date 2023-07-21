The code written in main.py will generate 12 csv files 
|_ out/test/train/val for dataset with stop words
|_ out_ns/train_ns/val_ns/test_ns for dataset without stop words
|_ out.label/train.label/test.label/val.label contains the positive or negative label for their corresponding files

The main code first reads the text dataset and tokenizes the data while also removing the special characters and then stores the processed data into CSV files. 

