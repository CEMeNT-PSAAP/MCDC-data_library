from io import text_encoding
import h5py
import os
import shutil

import numpy as np

import check_support
import reaction_list

# ======================================================================================
# Setup
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
#   The selected reaction MTs (see reaction_list.py) are based on
#   [https://t2.lanl.gov/nis/endf/mts.html].


def set_capture(openmc_reactions, mcdc_reactions):
    print("  Capture")

    # XS accumulator
    xs_capture = np.zeros_like(energy_grid)

    # Go over all reactions, accumulate capture reactions
    for reaction in reaction_list.captures:
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


def set_elastic_scattering(openmc_reactions, mcdc_reactions):
    print("  Elastic scattering")
    elastic_scattering = reaction_list.elastic_scattering

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
    mcdc_cosine.create_dataset(
        "energy_offset", data=openmc_angle["mu"].attrs["offsets"]
    )
    mcdc_cosine.create_dataset("value", data=openmc_angle["mu"][0])
    mcdc_cosine.create_dataset("PDF", data=openmc_angle["mu"][1])


# ======================================================================================
# Fission
# ======================================================================================
#   Supported physics:
#     - Only total fission (sum of first-, second-, third-, and fourth-chance)
#     - One prompt product, arbitrary number of delayed products
#       - If no products at all, transfer xs to capture
#     - Single distribution on all products
#     - Yield (in inducing energy)
#       - If total yield is given, it is assumed all-prompt
#       - Only linearly-interpolable tabulated-1D or polynomial
#     - Angle: Isotropic (even if correlated distributions are given)
#     - Energy
#         - Only continuous or Maxwell distribution
#         - Linear interpolation (assumed linear if histogram)


nuclist = {}
nuclist['Fission total yield casted as all-prompt'] = []
nuclist['Fission XS added to capture'] = []
nuclist['Fission product correlated distribution casted as uncorrelated-isotropic'] = []
nuclist['Fission spectrum constant interpolation casted as linear'] = []

def set_fission(nuclide_name, openmc_reactions, mcdc_reactions):
    print("  Fission")

    mcdc_fission = mcdc_reactions.create_group("fission")

    # XS
    xs = openmc_reactions[f"{reaction_list.fission}/{temperature}/xs"][()]
    mcdc_fission.create_dataset("xs", data=xs)

    # Get all the neutron products
    product_names = [
        x
        for x in list(openmc_reactions[f"{reaction_list.fission}"].keys())
        if (x[:7] == "product")
    ]
    product_names = [
        x
        for x in product_names
        if openmc_reactions[f"{reaction_list.fission}/{x}"].attrs["particle"].decode()
        == "neutron"
    ]
    products = [openmc_reactions[f"{reaction_list.fission}/{x}"] for x in product_names]

    # ==================================================================================
    # Check numbers of products
    # ==================================================================================

    # Get number of prompt and delayed products
    n_prompt = 0
    n_delayed = 0
    for i, product in enumerate(products):
        emission_mode = product.attrs["emission_mode"].decode()
        if emission_mode == "prompt":
            n_prompt += 1
        if emission_mode == "delayed":
            n_delayed += 1
        if emission_mode == "total":
            n_prompt += 1
            print(
                f"    [WARNING] fission product {product_names[i]} yield is given in total, which is assumed as all prompt"
            )
            nuclist['Fission total yield casted as all-prompt'].append(nuclide_name)
    
    # Check number of prompt and delayed products
    if n_prompt != 1 and n_delayed > 0:
        print(f"    [ERROR] fission has {n_prompt} prompt products")
        exit()
    if n_delayed == 0 and n_prompt == 0:
        print("    [WARNING] No products, fission XS is transferred to capture")
        nuclist['Fission XS added to capture'].append(nuclide_name)
        xs_capture = mcdc_reactions["capture/xs"][()]
        mcdc_reactions["capture/xs"][:] = xs_capture + xs
        del mcdc_reactions["fission"]
        return

    # ==================================================================================
    # Check other supported physics
    # ==================================================================================

    for i, product in enumerate(products):
        product_name = product_names[i]

        # Single distribution
        check_support.single_distribution(product)

        # Yield
        yield_type = product["yield"].attrs["type"].decode()
        if yield_type not in ["Tabulated1D", "Polynomial"]:
            print(
                f"    [ERROR] fission {product_name} yield not Tabulated1D or Polynomial: {yield_type}"
            )
            exit()

        if yield_type == "Tabulated1D" and product["yield"].attrs["interpolation"] != 2:
            print(f"    [ERROR] fission {product_name} yield not linearly interpolable")
            exit()

        # Check if distribution is not uncorrelated
        distribution_type = product["distribution_0"].attrs["type"].decode()
        uncorrelated = distribution_type == "uncorrelated"
        if not uncorrelated:
            print(
                f"    [WARNING] Casting correlated distribution in fission {product_name} into uncorrelated"
            )
            nuclist['Fission product correlated distribution casted as uncorrelated-isotropic'].append(nuclide_name+'/'+product_name)

            # Check if not linear interpolation
            interpolation_linear = all(
                [
                    x
                    for x in product["distribution_0/energy_out"].attrs["interpolation"]
                    == 2
                ]
            )
            if not interpolation_linear:
                print(
                    f"    [ERROR] Not linear interpolation energy distribution in fission {product_name}"
                )
                exit()

        if uncorrelated:
            # Check if energy distribution is not continuous or Maxwell
            distribution_type = product["distribution_0/energy"].attrs["type"].decode()
            continuous = distribution_type == "continuous"
            maxwell = distribution_type == "maxwell"
            if not (continuous or maxwell):
                print(
                    f"    [ERROR] Not continuous or Maxwell energy distribution in fission {product_name}: {distribution_type}"
                )
                exit()

            # If continuous distribution, check if NOT linear or constant interpolation
            if continuous:
                interpolation_linear = all(
                    [
                        x
                        for x in product["distribution_0/energy/distribution"].attrs[
                            "interpolation"
                        ]
                        == 2
                    ]
                )
                interpolation_constant = all(
                    [
                        x
                        for x in product["distribution_0/energy/distribution"].attrs[
                            "interpolation"
                        ]
                        == 1
                    ]
                )
                if not (interpolation_linear or interpolation_constant):
                    print(
                        f"    [ERROR] Not linear or constant interpolation energy distribution in fission {product_name}"
                    )
                    exit()
                if interpolation_constant:
                    print(
                        f"    [WARNING] Casting constant interpolation in fission spectrum of {product_name} into linear"
                    )
                    nuclist['Fission spectrum constant interpolation casted as linear'].append(nuclide_name+'/'+product_name)

            # If maxwell distribution, check if NOT linear interpolation
            if maxwell and not all(
                [
                    x
                    for x in product["distribution_0/energy/theta"].attrs[
                        "interpolation"
                    ]
                    == 2
                ]
            ):
                print(
                    f"    [ERROR] Not linear-linear interpolation energy distribution in fission {product_name}"
                )
                exit()
    
    # ==================================================================================
    # Set the rest of the data
    # ==================================================================================
    
    # Product groups
    mcdc_products = mcdc_fission.create_group("products")
    prompt_product = mcdc_products.create_group("prompt_neutron")
    delayed_products = mcdc_products.create_group("delayed_neutrons")
    
    # Loop over all products
    i_delayed = 0
    for i, product in enumerate(products):
        product_name = product_names[i]

        # Determine kind
        emission_mode = product.attrs["emission_mode"].decode()
        if (emission_mode == "prompt" or emission_mode == "total"):
            the_product = prompt_product
        if emission_mode == "delayed":
            i_delayed += 1
            the_product = delayed_products.create_group(f"group_{i_delayed}")
            the_product.create_dataset(
                "mean_emission_time", data=1.0 / product.attrs["decay_rate"]
            )

        # Yield
        the_yield = the_product.create_group("yield")
        yield_type = product["yield"].attrs["type"].decode()
        if yield_type == "Polynomial":
            the_yield.attrs['type'] = 'polynomial'
            the_yield.create_dataset(
                "polynomial_coefficient", data=product["yield"][()]
            )
        if yield_type == "Tabulated1D":
            the_yield.attrs['type'] = 'table'
            the_yield.create_dataset("energy_grid", data=product["yield"][0])
            the_yield.create_dataset("value", data=product["yield"][1])

        # Energy spectrum
        the_spectrum = the_product.create_group("energy_out")
        distribution_type = product['distribution_0'].attrs["type"].decode()
        uncorrelated = distribution_type == "uncorrelated"
        if uncorrelated:
            distribution_type = product['distribution_0/energy'].attrs["type"].decode()
            continuous = distribution_type == "continuous"
            maxwell = distribution_type == 'maxwell'
            if maxwell:
                the_spectrum.attrs['type'] = 'maxwellian'
                the_spectrum.create_dataset(
                    "maxwell_restriction_energy", data=product["distribution_0/energy"].attrs['u']
                )
                nuclear_temperature = the_spectrum.create_group('maxwell_nuclear_temperature')
                nuclear_temperature.create_dataset(
                    "energy_grid", data=product["distribution_0/energy/theta"][()][0]
                )
                nuclear_temperature.create_dataset(
                    "value", data=product["distribution_0/energy/theta"][()][1]
                )
            if continuous:
                the_spectrum.attrs['type'] = 'multi_pdf'
                the_spectrum.create_dataset("energy_grid", data=product["distribution_0/energy/energy"][()])
                the_spectrum.create_dataset("energy_offset", data=product["distribution_0/energy/distribution"].attrs['offsets'])
                the_spectrum.create_dataset("value", data=product["distribution_0/energy/distribution"][0])
                the_spectrum.create_dataset("PDF", data=product["distribution_0/energy/distribution"][1])
        else:
            the_spectrum.attrs['type'] = 'multi_pdf'
            the_spectrum.create_dataset("energy_grid", data=product["distribution_0/energy"][()])
            the_spectrum.create_dataset("energy_offset", data=product["distribution_0/energy_out/"].attrs['offsets'])
            the_spectrum.create_dataset("value", data=product["distribution_0/energy_out"][0])
            the_spectrum.create_dataset("PDF", data=product["distribution_0/energy_out"][1])

    # Cancel delayed products?
    if n_delayed == 0:
        del mcdc_products["delayed_neutrons"]


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
    if reaction_list.fission in openmc_reactions.keys():
        set_fission(nuclide_name, openmc_reactions, mcdc_reactions)

    # Close the files
    openmc_file.close()
    mcdc_file.close()

# Report approximated data
text = ''
for key in nuclist.keys():
    text += f'{key}\n'
    for item in nuclist[key]:
        text += f'  - {item}\n'
with open('approximation_notes.txt', 'w') as text_file:
    text_file.write(text)
