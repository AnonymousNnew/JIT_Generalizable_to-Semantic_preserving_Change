# Just In Time Defect Prediction Models: Are They Generalizable to Semantic-preserving Change?

## I'm grateful that you're interested in my work! :)

![Alt Text](https://c.tenor.com/vqeev_89AP0AAAAC/excited-adorable.gif)

I hope the following guide will make it easier for you to understand what we did.

## Requirements

Install all requirements in the [requirements.txt](https://github.com/AnonymousNnew/JIT_Generalizable_to_Semantic_Preservation_Change/blob/main/requirements.txt) file

## Step 1 - Extract Data:

1. Download the folder named bic and javadiff.
2. Requirements: 
    * Python 3.9 - then run:  
    
    ```
      python -m pip install --upgrade pip
      pip install pytest 
      pip install gitpython
      pip install jira
      pip install termcolor 
      pip install openpyxl  
      pip install javalang
      pip install pathlib
      pip install junitparser
      pip install pandas
      pip install numpy
      pip install pydriller
      pip install dataclasses
      pip install jsons
      python ./javadiff/setup.py install
     ```
    * java version 11 and 8
3. Checkout to directory name "./local_repo" the repository. For example:

   ```
   cd local_repo
   git clone https://github.com/apache/deltaspike.git
   ```
4. Execute: 

```
python Main.py [0]
```
  * Note: [0] - indicate  the number of commit extract in windown size of 50.  For large project set [0, 1, 2.....200].
 5. After the run the file save in "./results/" folder. 
 6. To merge all file Run:
 
  ```
    import pandas as pd
    import glob
    all_files = glob.glob("./results/*.csv")
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)
    pd.concat(li, axis=0, ignore_index=True).to_csv('./results/all.csv', index=False
   ```
 7. All data save in './results/all.csv'.
    

## Step 2 - Create Tabluar Data

This program runs the SZZ algorithm and performs pre-processing on tabular data.

1. Open data folder named  "Data".
2. In Data folder open for each project directory with the name NAME_PROJECT and put the file "all.csv" in the directory. For example  ("Data/knox/all.csv"). 
3. In "variable.py" add the NAME_PROJECT and the key_issue (according to JIRA) to function get_key_issue(). 
4. Update projects varibale in file CreateData.py and run:

   ```
   python main_create_data.py
   ```
5. After this run several file created:
   - "Data/{NAME_PROJECT}/all_after_preprocess.csv" - this file is the tabular data for RF and LR after pre-processing.
   - "Data/{NAME_PROJECT}/blame/pydriller_{NAME_PROJECT}_bugfixes_bic.csv" - the results of the SZZ algorithm.


## Step 3 - Create Raw Textual Data 

In parallel to step 2, you can generate raw textual data.

1. In main.py update the name_projects and url_projects varible. For example:

   ```
   name_projects = ['zeppelin'] # NAME_PROJECT
   url_projects = ['https://github.com/apache/zeppelin'] # link to github
   ```
2. Execute: 

  ```
   python main.py 0
   ```
   * Note: 0 - indicate to genrate the raw textual data for the real data.
3. After this run several file created:
   - for each cross 'k' from 1 to 5:
      - "Data/{NAME_PROJECT}/{k}/{NAME_PROJECT}_train.pkl" - train data set. 
      - "Data/{NAME_PROJECT}/{k}/{NAME_PROJECT}_test.pkl" - test data set.
      - "Data/{NAME_PROJECT}/{k}/{NAME_PROJECT}_dict.pkl" - dictionary for tha data.
      - "Files/{NAME_PROJECT}/{k}/{NAME_PROJECT}_methods.csv" - the methods that the modification changed are recorded for each file in the test set.
      - "Files/{NAME_PROJECT}/{k}/{commit.hash}_after_{file.filename}" - the test file after the modification was written for the transformations phase. 
      - "Files/{NAME_PROJECT}/{k}/{commit.hash}_before_{file.filename}" - the test file before the modification was written for the transformations phase. 


## Step 4 - Transformation 
  
 
This step activates the transformations on the written test files.

1.  Place the JavaTransformerWorkflows.ja in a directory separate from this project's directory (putside form this project's directory).
2. Execute: 

   ```
   java -jar JavaTransformerWorkflows.jar {k} {NAME_PROJECT} {index}
   ```
  * Note: 
    - k - indicate the cross number.
    - NAME_PROJECT - the name project
    - index - indicate  the number of file extract in windown size of 200. For large project set [0, 1, 2.....20].
    
## Step 5 - Create Transformation Data

1. In main.py update the name_projects and url_projects varible. For example:

   ```
   name_projects = ['zeppelin'] # NAME_PROJECT
   url_projects = ['https://github.com/apache/zeppelin'] # link to github
   ```
2. Execute: 

  ```
   python main.py 1
   ```
   * Note: 1 - indicate to genrate the raw textual data for the transform data.
3. After this run several file created:
   - for each cross 'k' from 1 to 5:
      - "Data/{NAME_PROJECT}/{k}/{NAME_PROJECT}_train.pkl" - train data set. 
      - "Data/{NAME_PROJECT}/{k}/{NAME_PROJECT}_test.pkl" - test data set.
      - "Data/{NAME_PROJECT}/{k}/{NAME_PROJECT}_dict.pkl" - dictionary for tha data.
      - "Files/{NAME_PROJECT}/{k}/{NAME_PROJECT}_methods.csv" - the methods that the modification changed are recorded for each file in the test set.
      - "Files/{NAME_PROJECT}/{k}/{commit.hash}_after_{file.filename}" - the test file after the modification was written for the transformations phase. 
      - "Files/{NAME_PROJECT}/{k}/{commit.hash}_before_{file.filename}" - the test file before the modification was written for the transformations phase. 

## Step 6 - Run Machine Learning Models (RF and LR)

1. In tabular_data.py projects varible. For example:

   ```
   projects = ['knox'] # NAME_PROJECT
   ```
2. Execute: 

  ```
   python tabular_data.py
  ```
3. This process consisted of several steps:
   - read the tabluar data from step 2 and split to train test accroding the raw textual data. for each cross 'k' from 1 to 5:
      - "Data/{NAME_PROJECT}/{k}/{NAME_PROJECT}_test_tabular.csv" - train data set. 
      - "Data/{NAME_PROJECT}/{k}/{NAME_PROJECT}_test_tabular.csv" - test data set.
   - Tuning for RF and RF completed
   - The results written to:
      - "Data/{NAME_PROJECT}/{model}/metrics_avg.csv" - results for baseline performance
      - "Data/{NAME_PROJECT}/{model}/evel_avg.csv" - results for RQ1 and RQ2.

## Step 6 - Run [DeepJIT](https://github.com/hvdthong/DeepJIT_updated) and Run [CC2Vec](https://github.com/CC2Vec/CC2Vec)
   
Run the models with the data created in step 3 according to the readme file in GitHub.

* Take at note that maybe you will be need change the relative path in the project.

## Good luck !!! 
