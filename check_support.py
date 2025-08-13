def single_distribution(product):
    if product.attrs["n_distribution"] != 1:
        print("    [ERROR] Not single distribution")
        exit()


def uncorrelated_distribution(product):
    distribution_type = product["distribution_0"].attrs["type"].decode()
    if distribution_type != "uncorrelated":
        print(f"    [ERROR] Not uncorrelated distribution: {distribution_type}")
        exit()


def COM(reaction):
    if reaction.attrs["center_of_mass"] != 1:
        print("    [ERROR] Distribution is not in COM")
        exit()


def linear_interpolation(distribution):
    if not all([x for x in distribution.attrs["interpolation"] == 2]):
        print("    [ERROR] Not linear interpolation")
        exit()
