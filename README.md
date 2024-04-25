# Default repository template

## Title
Create summary of tv show episode without spoilers from scripts

## Abstract
_A max 150-word description of the project question or idea, goals, dataset used. What story you would like to tell and why? What's the motivation behind your project?_
This project aims to create summaries of tv show episodes from available scripts of the episodes. Our goal is that these summaries do not contain spoilers for the story in the episode. 


## Research questions
_A list of research questions you would like to address during the project._
* How can we train a model to recognize the climax/resolution stage of a story?
* How can we train a summarisation model to ignore this stage in it's summary to avoid spoilers?
* What's the effect of stage directions on TV script summarisation results?

## Dataset
_List the dataset(s) you want to use, and some ideas on how do you expect to get, manage, process and enrich it/them. Show you've read the docs and are familiar with some examples, and you've a clear idea on what to expect. Discuss data size and format if relevant._
We use scripts of sitcom-like TV shows available on Kaggle.
[Seinfield](https://www.kaggle.com/datasets/thec03u5/seinfeld-chronicles)
[Friends](https://www.kaggle.com/datasets/blessondensil294/friends-tv-series-screenplay-script)
[The Office](https://www.kaggle.com/code/washingtongold/load-the-office-scripts/output)
[South Park](https://www.kaggle.com/datasets/thedevastator/south-park-scripts-dataset)


## A tentative list of milestones for the project
Add here a sketch of your planning for the coming weeks. Please mention who does what.

* Create a dataset by combining our input (scripts) and expected output (synopsis sentences)
* Clean up dataset to a usable format
* Create an NLP pipeline for preprocessing the datasets
* Try first attempt at training the model
* Look into the results and find room for improvement
* Modify model and try again


## Documentation
This can be added as the project unfolds. You should describe, in particular, what your repo contains and how to reproduce your results.
