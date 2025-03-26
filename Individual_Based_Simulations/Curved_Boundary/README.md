The code included here (main-curved_boundary.py)  runs the simulations used to produce Fig. 4D,E in the manuscript.

The code is meant to be used to run many simulations in parallel. There needs to be an "out.$ID" folder for every process. The code will output there the last 20 recordings of the inter-strain boundary
in each simulation, in a file named "contour_list.$ID.dat" in the corresponding folder. The code takes in as initial configuration the output of some previous simulation. This configuration is found in the folder "initial_condition/". Its contents must be copied to every one of the "out.$ID/" folders and renamed to include the $ID of the process, like so: filename.dat --> filename.$ID.dat

Outputs of simulations shown in Fig. 4D,E in the manuscript are not inlcuded as they are extremely heavy, but are available upon request, as well as the code that produces the Figure from them. Please write to victor.peris@phys.ens.fr if interested.
