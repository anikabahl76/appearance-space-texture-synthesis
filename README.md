# appearance-space-texture-synthesis
CSCI 1290 Final Project

# usage

- **lefebvre_hoppe_synthesis.py** contains the code for running Lefebvre & Hoppe's appearance-space texture synthesis algorithm. 
To run, enter the simply call `python lefebvre_hoppe_synthesis.py` from the command line, within the **code** directory.
For more information on specifying parameters beyond the defaults, run `python lefebvre_hoppe_synthesis.py --help`.

- **harrison_synthesis.py** contains the code for running Harrison's texture synthesis algorithm with appearance-space preprocessing. 
To run, enter the simply call `python harrison_synthesis.py` from the command line, within the **code** directory.

This runs the texture synthesis algorithm on all textures contained within the **data** directory. All the toggle-able hyperparameters are contained at the top of the file and can be changed there. 

- **efros_freeman_synthesis.py** contains the code for running Efros & Freeman's image quilting texture synthesis algorithm with appearance-space preprocessing. 
To run, enter the simply call `python efros_freeman_synthesis.py` from the command line, within the **code** directory.

This runs the texture synthesis algorithm on all textures contained within the **patch_data** directory.