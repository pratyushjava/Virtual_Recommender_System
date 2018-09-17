To run the chatbot application you will require following tools
1. tensorflow 0.12.01
2. Anaconda 3.4.2
3. Python 2.7

To test our chatbot on trained model for amazon review data set:
Make sure in the seq2seq.ini, mode = test
Run the myMain.py 

To Train our sequence to sequence model:
Make sure in the seq2seq.ini, mode = trained
Run the myMain.py

Preprocessed data set is in the "data" directory 
Previously saved chekpoints are found in the working directory

This module has recomendation system algorithm.
1. To run this module independently one need to give the value of constant data_path at the top of each file clustering.py and collaborative_filtering.py
2. data_path should have the value of complete path of data file.
3. complete dataset can be downloaded from below link:
   review_data = http://snap.stanford.edu/data/amazon/productGraph/user_dedup.json.gz
   meta_data = http://snap.stanford.edu/data/amazon/productGraph/metadata.json.gz
   
   the clustering.py file need meta_data dataset.
   the collaborative_filtering.py file need review_data dataset.
   note: the meta_data data set is not clean there are erregular singel quotes('), double quotes("), simicolon(;)
   so we have preprocessed these files to some extent which can be found in the data folder here: 
	https://drive.google.com/drive/folders/0ByF97F7OFu5IWGdhTG05WWJoMzQ?usp=sharing
	https://drive.google.com/drive/folders/0ByF97F7OFu5IUkhjZE5lVkxhOTA?usp=sharing
   give the path of metadata.json.gz file found in data folder in clustering.py file
   give the path of reviews_200k.json.gz file found in data folder in collaborative_filtering.py
   
4. In addition to above files to run the code one also need python3x, Networkx Lib, and python community module.
   these packages can be found at winpython.
5. clustering.py has implementation of community detection technique and Hierarchical clustering techniques which uses metadata dataset
6. collaborative_filtering.py has implementation of collaborative filtering technique which uses review_data dataset

