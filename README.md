# COVID-Net USPro: Prototypical Network for COVID Ultrasound Image Classification 

Implementation of a prototypical network for few-shot learning on COVIDx-US, an ultrasound image dataset that contains image types: Normal, COVID, Pneumonia and Other. 

## Requirements 
TBA 

## COVID-Net USPro Analysis workflow 
The analysis workflow for COVID-Net USPro is outlined below. 
![COVID-Net USPro Analysis workflow](./paper_figs/analysis_overview.png?raw=true "COVID-Net USPro Analysis workflow")

The prototypical network structure of COVID-Net USPro is also illustrated below. 
![COVID-Net USPro](./paper_figs/concept_flow.png?raw=true "COVID-Net USPro Prototypical Architecture")

## COVID-US Dataset 
The COVIDx-US dataset is an open-access benchmark dataset of ultrasound imaging data for AI-driven COVID-19 analytics. The github repo can be accessed here: https://github.com/nrc-cnrc/COVID-US. For the complete description, please refer to [paper](https://pubmed.ncbi.nlm.nih.gov/35866396/). 

## Results 
Full test results are detailed in experiment_outputs folder. The following plots illustrate the testing results of two encoders ResNet18L1 and ResNet50L4. 
![Performance with shots](./paper_figs/outputs_model1and4_4classes.png?raw=true "Performance results")