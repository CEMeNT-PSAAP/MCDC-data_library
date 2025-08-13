import h5py
import os
import shutil

import numpy as np

# ======================================================================================
# Setups
# ======================================================================================

# Directories
openmc_dir = "/Users/ilhamvariansyah/nuclear_data/endfb-viii.0-hdf5/neutron/"
output_dir = "mcdc_data_library"

# Parameters
temperature = "294K"

# ======================================================================================
# Preparation
# ======================================================================================

# Create output directory if needed
os.makedirs(output_dir, exist_ok=True)

# ======================================================================================
# Capture reactions
# ======================================================================================
#   Capture reactions consist of all reactions not producing neutrons.
#   The selected MTs are based on [https://t2.lanl.gov/nis/endf/mts.html].

capture_reactions = [
    "reaction_102",
    "reaction_103",
    "reaction_104",
    "reaction_105",
    "reaction_106",
    "reaction_107",
    "reaction_108",
    "reaction_109",
    "reaction_111",
    "reaction_112",
    "reaction_113",
    "reaction_114",
    "reaction_115",
    "reaction_116",
    "reaction_117",
]


def set_capture_reactions(openmc_reactions, mcdc_reactions):
    # XS accumulator
    xs_capture = np.zeros_like(energy_grid)

    # Go over all reactions, accumulate capture reactions
    for reaction in capture_reactions:
        if reaction in openmc_reactions.keys():
            openmc_xs = openmc_reactions[f"{reaction}/{temperature}/xs"]
            idx_start = openmc_xs.attrs["threshold_idx"]
            xs_capture[idx_start:] += openmc_xs[()]

    # Create dataset
    mcdc_reactions.create_dataset("capture/xs", data=xs_capture)


# ======================================================================================
# Main
# ======================================================================================

# Loop over all data
for file_name in os.listdir(openmc_dir):
    # Skip unsupported data file
    if file_name[-3:] != ".h5" or file_name[-6:] == "_m1.h5" or file_name[:2] == "c_":
        continue

    print(f"Generating {file_name}...")

    # Open OpenMC file and create the corresponding MC/DC file
    openmc_file = h5py.File(f"{openmc_dir}/{file_name}", "r")
    mcdc_file = h5py.File(f"{output_dir}/{file_name}", "w")

    # Basic nuclide information
    nuclide_name = file_name[:-3]
    openmc_nuclide = openmc_file[nuclide_name]
    atomic_weight_ratio = openmc_nuclide.attrs["atomic_weight_ratio"]
    mcdc_file.create_dataset("nuclide_name", data=f"{nuclide_name}")
    mcdc_file.create_dataset("atomic_weight_ratio", data=atomic_weight_ratio)

    # Reactions data group
    mcdc_reactions = mcdc_file.create_group("neutron_reactions")
    openmc_reactions = openmc_nuclide["reactions"]

    # Reaction XS energy grid
    energy_grid = openmc_file[f"{nuclide_name}/energy/{temperature}"][()]
    mcdc_reactions.create_dataset("xs_energy_grid", data=energy_grid)

    # Set the supported reactions
    set_capture_reactions(openmc_reactions, mcdc_reactions)

    # Close the files
    openmc_file.close()
    mcdc_file.close()
