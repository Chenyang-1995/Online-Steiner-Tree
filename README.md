# Learning-Augmented Algorithms for Online Steiner Tree
This is the code for the paper "Learning-Augmented Algorithms for Online Steiner Tree".

# Requirements

Python >= 3.6.11

scipy >= 1.5.4

matplotlib >= 3.3.1




# Random graphs
To do the robustness experiments, run
```
python robustness_random_graph.py
```
To obtain the learnability performance of uniform distribution and two-class distribution, run
```
python learnability_random_graph_uniform_distri.py
```
and 
```
python learnability_random_graph_twoclass_distri.py
```
respectively.

# Road graphs

We give 4 text files in ``road_graph/``, each corresponding to a road graph. In each file, a row (u,v,w) respresents the edge (u,v) with cost w.

To do the robustness experiments, run
```
python robustness_road_graph.py
```

To obtain the learnability performance of cluster distribution, run
```
python learnability_road_graph_cluster_distri.py
```
The number of terminals sampled per cluster is 100 by default.

Some functions are provided in ``draw.py``, which could be useful when drawing figures for the experiments.

# Acknowledgement

We thank Mirko Giacchini for the discussion on some implementation details of the proposed algorithms.
