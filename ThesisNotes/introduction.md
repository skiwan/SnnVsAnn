# Introduction

### What are BCIs and why do we need them

motor controlled devices translate physical motion into computer commands like mouse and keyboard. peripheral control devices that interact directly with human motion. BCI focus on interpreting brain activity into digital command signals for computers. Many application purposes, socially most important is control of computers for people who are not able to produces motoric input for other peripheral devices + extend our physical limitations

### What sub-task does this thesis focus on (Decoding of EEG and Motor Imagery classification)

Multiple types of BCIs
This thesis focus on decoding of EEG signal within the domain of motor imagery classification
In short, given an eeg signal, determine when and if a subject is imagining a specific movement (hence the name)

### What would be optimal solution for task (setup for real life)
Full loop system that non invasive records brain activity, recognizes when a person is imagining a specific movement, wanting to execute a specific command and classifying the command correctly in real time with minimal hardware setup required. If a BCI can be described as a natural extension of the human body

### Why are ML Methods suitable here (Generalization and pattern recognition)
Assumption that specific brain intentions have similar patters. hard to define specific rules or heuristics by hand, need to be specifically tuned to each subject. Generalization is required, noise resistance and pattern recognition.. ML has proven to be very efficient in these tasks, neural networks have proven to be really good at generalization, handling unknown input and even utilize online learning & learning during application

### Starting Point of thesis (BCI Contest) + current state of the art approaches + no comparison (transition to comparison in EEG motor imagery classification)
BCI competition over multiple years. Relevant and difficult tasks to help move the field forward. Many different approaches and methods which show there is no best practice yet. From simple classifiers over Gen 2 ANNs to Gen3 Snns have been applied.   

Short listing of some of the different methods and models applied

Further investigation has shown there seems to be a general lack of information about differences of SNNs and ANNs in BCIs and Motor imagery classification. How to decide on model to use based on task?

### General Scientific question we are aiming at
What differences are there between SNNs and ANNs if I want to do Motor imagery classification as a researcher\for my application? Are there any significant differences in the first place?