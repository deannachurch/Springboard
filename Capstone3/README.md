# Capstone 3 project

## Overview
An integral task when evaluating new technology is to understand how this technology fits into the larger corpus of scientific work. However, peer-reviewed publications tend to be a lagging indicator of scientific progress due to the long time required to publish a manuscript (often 1-2 after the research is finished and the manuscript written by the authors). Preprint servers, such as ArXiv and BioRxiv, are often a much better leading indicator of trends in science. 

For Capstone 3, I propose to build a system that will allow me to:

* Classify papers into a predefined list of paper types. 
* Analyze trends in ‘hot topics’ of research over time. 
* Establish networks of authors to identify potential Key Opinion Leaders (KOLs) in a given field. 

## Notebooks

I submitted the README because there are three notebooks as part of this Capstone project, not just one. I have included links to each of the notebooks below.

Paper classification: https://github.com/deannachurch/Springboard/blob/main/Capstone3/notebooks/BuildModels.ipynb
Trend analysis: https://github.com/deannachurch/Springboard/blob/main/Capstone3/notebooks/TrendAnalysis.ipynb
Network analysis: https://github.com/deannachurch/Springboard/blob/main/Capstone3/notebooks/NetworkAnalysis.ipynb

EDA: https://github.com/deannachurch/Springboard/blob/main/Capstone3/notebooks/EDA.ipynb

## Criteria for Success
* Paper classification using a single term (even though many papers cover multiple terms) with an accuracy of greater than 90%. 
* Insights into how research trends have changed over the past 10 years. 
* Ability to identify high value KOLs for follow-up. These will likely be individuals with many connections within the network and also likely a large number of published manuscripts. 

## Source Data

https://www.kaggle.com/datasets/sumitm004/arxiv-scientific-research-papers-dataset

## Directory Structure
```
Capstone3   
├── README.md
├── data
│   ├── raw
│   │   ├── arXiv-scientific_dataset.csv        <--- immutable source data>
│   ├── processed                               <--- any artefacts I produce to clean the data>
├── models                                      <--- models
├── notebooks                                   <--- notebooks used for EDA and model development
|  ├── EDA.ipynb                                <--- EDA notebook
├── scripts                                     <--- scripts to modularize analysis and run models (if needed)
├── conda_env.yml                               <--- conda environment definition
├── .gitignore                                  <--- files to ignore
```

## Analysis plan/To Do List

1. Introduction & Data Exploration
    - [X] Load data
    - [X] Explore data
    - [X] Clean data
    - [X] Generate basic statistics 
2. Paper Categorization
    - [X] Obtain sciBert model (https://github.com/allenai/scibert), we will use this pre-trained model to classify papers into a predefined list of paper types.
    - [X] Fine-tune the model on a subset of the data.
    - [X] Evaluate the model on a test set.
    - [X] Generate a classification report to summarize the performance of the model.
3. Trend Analysis
    - [ ] Topic Modeling 
    - [ ] Generate a list of the top 10 topics in the data set.
    - [ ] Generate a list of the top 10 topics in the data set over time.
4. Author Network Analysis
    - [ ] Generate a list of the top 10 authors in the data set.
    - [ ] Generate a list of the top 10 authors in the data set over time.
    - [ ] Generate a network graph of the authors in the data set.
    - [ ] Generate a list of the top 10 authors in the network graph.