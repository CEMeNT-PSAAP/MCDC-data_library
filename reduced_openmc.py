import h5py
import os
import shutil

import reaction_list

# ======================================================================================
# Setup
# ======================================================================================

supported_reactions = reaction_list.captures + [reaction_list.elastic_scattering] + [reaction_list.fission]

# Directories
OPENMC_DIR = "/Users/ilhamvariansyah/nuclear_data/endfb-viii.0-hdf5/neutron/"
output_dir = "/Users/ilhamvariansyah/nuclear_data/endfb-viii.0-hdf5/neutron-reduced/"

# ======================================================================================
# Preparation
# ======================================================================================

# Create output directory if needed
os.makedirs(output_dir, exist_ok=True)

# ======================================================================================
# Main
# ======================================================================================

# Loop over all data
for file_name in os.listdir(OPENMC_DIR):
    # Skip unsupported data file
    if file_name[-3:] != ".h5" or file_name[-6:] == "_m1.h5" or file_name[:2] == "c_":
        continue
    
    print(f"Creating reduced {file_name}...")
    
    # Copy the original file for reduced OpenMC data
    shutil.copy(f"{OPENMC_DIR}/{file_name}", f"{output_dir}/{file_name}")
    openmc_file = h5py.File(f"{output_dir}/{file_name}", "a")
    nuclide_name = file_name[:-3]

    # Delete non-relevant subgroup
    for subgroup in openmc_file[f'{nuclide_name}'].keys():
        if subgroup not in ["reactions", "kTs", "energy"]:
            del openmc_file[f'{nuclide_name}/{subgroup}']

    # Loop over reactions and delete unsupported ones
    openmc_reactions = openmc_file[f'{nuclide_name}/reactions']
    for reaction in openmc_reactions.keys():
        if reaction not in supported_reactions:
            del openmc_reactions[reaction]

    openmc_file.close()
