# IR-Project-Wikipedia-
In this project we created a retrieval engine on Wikipedia documents, saved in my GCP bucket.

You will find a few files in this repository:

1.  Project Report.word - a report in Hebrew, documenting our product and course of action.
2. Build_Index_GCP.ipynb - This file is creating an Inverted index using PySpark abilities for parallel calculation. In order to use, you'll need to upload the file "Inverted_index_gcp", also found here. You will need to initiate your cluster using "init_file.sh", also found here.
3.  Inverted_index_gcp.py - A class to build, read and write the inverted index.
4.  init_file.sh - Initialization file for the cluster, containing the relevant installs.
5.  Project_Retrieval_Development.ipynb - A colab notebook used to develop our retrieval method. Workprocesses are documented in it, as well as different experiments done along the way. Have fun! Upload to it the Inverted_index_gcp, queries_train files.
6.  queries_train.json - A Json file with a few test cases. Te Colab notebook will use it in order to conduct the tests in it.
7.  search_frontend.py
8.  run_frontend_in_colab.ipynb
9.  run_frontend_in_gcp.sh
10.  startup_script_gcp.sh
11.  Search_results_synonims / no_synonims.pdf - 2 optimization files used in order to create our ranking in te retrieval, using grid search.
12.  Synonim_Comparison.xlsx - weights optimization based on the 2 retrieval methods (with / without synonims)
