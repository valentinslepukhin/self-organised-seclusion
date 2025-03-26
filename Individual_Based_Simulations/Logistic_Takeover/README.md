The code included here performs the simulations that produce the effect of logistic takeover, shown in Fig. 3C in the manuscript.

The initial fraction of advantageous cells is controlled by the parameter "initial_fraction_advantageous" on line646 and the relative growth rate advantage is set by the parameter "s" on line640.

The code needs one "out.$ID/" folder for every process that is run in parallel using multiprocessing, where $ID is the number of the current process.
The code takes in as initial configuration the output of some previous simulation. This configuration is found in the folder "initial_condition/". Its contents must be copied to every one of the "out.$ID/" 
folders and renamed to include the $ID of the process, like so: filename.dat --> filename.$ID.dat

Outputs of simulations shown in Fig. 3C in the manuscript are not inlcuded as they are extremely heavy, but are available upon request, as well as the code that produces the Figure from them. Please write
to victor.peris@phys.ens.fr if interested.
