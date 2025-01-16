The repository for the paper "Self-Organized Colonization Resistance without Physical Barriers".

In the folder "Panflute-design" there are files to create the design of the whole  mask or design of the single panflute for COMSOL simulation. Run the file mask-design.  to get both of them. They generate scr. files that are used in AutoCAD as script (in Autocad, choose from command line script and choose the script file earlier generated). 
To use the file for COMSOL simulation, expot it from AutoCAD as dxf, in COMSOL start new 3D geometry (Stationary Laminar Flow), create there Work plane and import dxf file. Then extrude it, assign inlet and outlet and run simultion.
