# Embedded chimera states in recurrent neural networks

Set of codes that you need in order to reproduce the results presented in *Embedded chimeras in recurrent neural networks* (https://www.nature.com/articles/s42005-022-00984-2).

#### Structure
In this project you have different programs that are used for three different objectives. (1) Generate the supervisor (chimera). (2) Train a RNN using the FORCE method to mimic the chimera state previously generated. (3) Plot and analize the RNN (basically the figures in the article). 

The chimera state was produced with the C++ code. Depending on the number of nodes (n = 3 or n = 25) the initial conditions change (see code to choose accordingly).
The training of the RNN was done with the python codes. Different codes are used for different constrains (see title). Finally, you have the scripts to produce the different figures in the paper.
