# IR-Project-Wikipedia-
In this project we created a retrieval engine on Wikipedia documents, saved in my GCP bucket.

You will find a few files in this repository:

1.  Project Report.word - a report in Hebrew, documenting our product and course of action.
2. Build_Index_GCP.ipynb - This file is creating an Inverted index using PySpark abilities for parallel calculation. In order to use, you'll need to upload the file "Inverted_index_gcp", also found here. You will need to initiate your cluster using "init_file.sh", also found here.
3.  Inverted_index_gcp.py - A class to build, read and write the inverted index.
4.  init_file.sh - Initialization file for the cluster, containing the relevant installs.
5.  Project_Retrieval_Development.ipynb - A colab notebook used to develop our retrieval method. Workprocesses are documented in it, as well as different experiments done along the way. Have fun! Upload to it the Inverted_index_gcp, queries_train files.
6.  backend.py - A clean and organized code file where our retrieval app is implemented. The functions in this file are used by the Search() function in search_frontend.py file, while having the indexes loaded.
7.  queries_train.json - A Json file with a few test cases. Te Colab notebook will use it in order to conduct the tests in it.
8.  search_frontend.py - Flask app for search engine frontend. A wrapper that activates our backend file.
9.  run_frontend_in_colab.ipynb - notebook running the search engine's frontend in Colab for development purposes. The notebook also provides instructions for querying/testing the engine.
10.  run_frontend_in_gcp.sh - command-line instructions for deploying the search engine to GCP.
11.  startup_script_gcp.sh - A shell script that sets up the Compute Engine instance.
12.  Search_results_synonims / no_synonims.pdf - 2 optimization files used in order to create our ranking in te retrieval, using grid search.
13.  Synonim_Comparison.xlsx - weights optimization based on the 2 retrieval methods (with / without synonims)
