## Bachelor Thesis main directory

The adjencent directories in the parent directory consist of matlab, R code and previous conducted expriment from a Fgaprojekt conduct by another group of students at DTU.

This directory atempts a complete restructuring and refactoring, but with code borrowed from the `fagprojet directory.
___

The `project_main` directory have the following directory structure

    ├── data                    # datasets and labels
    ├── distributions           # Probability distribution classes
    ├── experiments             # Script for running experiments/training
        ├── results
    ├── utils
        ├── preproccesing


### In code to-do
* Logsoftmax, Logsoftplus
* logkummer, addiction in kummer-> logkummer?,
* log pdf ACG
* surface area konstant? log?
* Softplus conasaint 0 wich is not  pos definite
* all n! to Gamma?
* 
* High likelihood values
* Change ACG matmul to log domain
* Change nn.functional to nn. for correct backprop and device
* Viterbi & forward matmul/state prob weight

### To-do
1. Training-loop/loading, splits etc.
2. Watson EM traning
2. Evaluate optimal number of clusters/state for modelling
4. Ground truth task signal comparison/evaluation - General Linear model?
5. Report Content plan