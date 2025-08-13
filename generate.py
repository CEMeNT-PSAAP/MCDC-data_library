import h5py
import os
import shutil

import numpy as np

import check_support

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


def set_capture(openmc_reactions, mcdc_reactions):
    print("  Capture")

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
# Elastic scattering
# ======================================================================================
#   Supported physics:
#     - Single distribution
#     - Uncorrelated angle-energy
#     - Angle
#         - COM
#         - Linear-linear interpolations

elastic_scattering = "reaction_002"


def set_elastic_scattering(openmc_reactions, mcdc_reactions):
    print("  Elastic scattering")

    openmc_secondary = openmc_reactions[f"{elastic_scattering}/product_0"]

    # Check supported physics
    check_support.single_distribution(openmc_secondary)
    check_support.uncorrelated_distribution(openmc_secondary)
    check_support.COM(openmc_reactions[elastic_scattering])
    check_support.linear_interpolation(openmc_secondary["distribution_0/angle/mu"])

    mcdc_elastic = mcdc_reactions.create_group("elastic_scattering")

    # XS
    xs = openmc_reactions[f"{elastic_scattering}/{temperature}/xs"][()]
    mcdc_elastic.create_dataset("xs", data=xs)

    # Scattering cosine distribution
    openmc_angle = openmc_secondary[f"distribution_0/angle"]
    mcdc_cosine = mcdc_elastic.create_group("scattering_cosine")
    #
    mcdc_cosine.create_dataset("energy_grid", data=openmc_angle["energy"][()])
    mcdc_cosine.create_dataset("energy_offset", data=openmc_angle["mu"].attrs["offsets"])
    mcdc_cosine.create_dataset("value", data=openmc_angle["mu"][0])
    mcdc_cosine.create_dataset("PDF", data=openmc_angle["mu"][1])


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
    set_capture(openmc_reactions, mcdc_reactions)
    set_elastic_scattering(openmc_reactions, mcdc_reactions)

    # Close the files
    openmc_file.close()
    mcdc_file.close()
