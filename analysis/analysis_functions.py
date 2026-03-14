# Necessary imports
import pypsa
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import geopandas as gpd
import xarray as xr
from scipy.stats import norm, lognorm, rayleigh, gamma, t as student_t, johnsonsu, kstest
from scipy.optimize import fsolve
from math import erf, exp, log

# Load network
# run_name = "Elec-only_test" # Replace with correct run name
# results_directory = "results/{}".format(run_name)
# n = pypsa.Network(f"{results_directory}/networks/base_s_1_elec_EP.nc")  # Replace with correct network name
# scenario = 'no_cfd'   # Replace with correct scenario name if needed ('cap','no_cfd')


# Color theme
mcgreen = (0.8, 0.898, 0.8) 
mcorange = (1.0, 0.9294, 0.8)
mcblue = (0.8, 0.8, 1.0)
mcred = (1.0, 0.8, 0.8)    


# Club-Mate Granat color palette
cm_1 = (179/255, 27/255, 35/255) # granat_red
cm_2 = (249/255, 207/255, 61/255) # yellow
cm_3 = (0/255, 87/255, 56/255) # green
cm_4 = (164/255, 21/255, 27/255) # dark_red
cm_5 = (21/255, 21/255, 21/255) # black
cm_6 = (245/255, 245/255, 245/255) # white


def color_theme(carrier):
    if carrier in ['onwind', 'biomass']:
        return cm_3  # green
    elif carrier in ['solar', 'solar rooftop', 'solar-hsat']:
        return cm_2  # yellow
    elif carrier in ['offwind', 'offwind-ac', 'offwind-dc', 'offwind-float']:
        return cm_1  # granat_red
    elif carrier in ['ror', 'PHS', 'H2']:
        return mcblue
    elif carrier in ['CCGT', 'OCGT', 'coal', 'lignite', 'oil']:
        return cm_5  # black
    else:
        return cm_6  # white


# ___ Define network ___
def get_network(run_name):
    # Check parent directory first (assuming we are in 'analysis'), then current directory
    possible_paths = [
        "../results/{}".format(run_name),
        "results/{}".format(run_name)
    ]
    
    results_directory = possible_paths[0]
    for path in possible_paths:
        if os.path.isdir(os.path.join(path, "networks")):
            results_directory = path
            break

    net_dir = os.path.join(results_directory, "networks")
    if not os.path.isdir(net_dir):
        raise FileNotFoundError(f"Networks directory not found. Checked: {possible_paths}. cwd: {os.getcwd()}")
    files = [f for f in os.listdir(net_dir) if os.path.isfile(os.path.join(net_dir, f)) and not f.startswith('.')]
    if len(files) == 0:
        raise FileNotFoundError(f"No files found in {net_dir}")
    if len(files) > 1:
        raise ValueError(f"Expected exactly one file in {net_dir}, found: {files}")
    network_file = os.path.join(net_dir, files[0])
    n = pypsa.Network(network_file)
    return n

# ___ Naming conventions ___
def carrier_full_name(carrier):
    names = {
        'onwind': 'Onshore Wind',
        'offwind-ac': 'Offshore Wind (AC)',
        'offwind-dc': 'Offshore Wind (DC)',
        'offwind-float': 'Offshore Wind (Floating)',
        'solar': 'Ground-Mounted Solar PV',
        'solar rooftop': 'Rooftop Solar PV',
        'solar-hsat': 'Rotating Solar PV',
        'ror': 'Run-of-River Hydro',
        'geothermal': 'Geothermal',
        'biomass': 'Biomass',
        'CCGT': 'Combined Cycle Gas Turbine',
        'OCGT': 'Open Cycle Gas Turbine',
        'coal': 'Hard Coal',
        'lignite': 'Lignite',
        'oil': 'Oil-fired Power Plant',
        'nuclear': 'Nuclear Power Plant'
    }
    return names.get(carrier, carrier)

def colormap_for_carrier(carrier):
    colormaps = {
        'onwind': 'Greens',
        'offwind-ac': 'Blues',
        'offwind-dc': 'Blues',
        'offwind-float': 'Blues',
        'solar': 'Oranges',
        'solar rooftop': 'Oranges',
        'solar-hsat': 'Oranges',
        'ror': 'Blues',
        'geothermal': 'Reds',
        'biomass': 'Greens',
        'CCGT': 'Greys',
        'OCGT': 'Greys',
        'coal': 'Greys',
        'lignite': 'Greys',
        'oil': 'Greys',
        'nuclear': 'Greys'
    }
    return colormaps.get(carrier, 'viridis')


# ___ Get electricity generation time series ___

# Function to get hourly generation of any carrier
def get_hourly_generation(carrier, n, rc=0):
    if carrier == 'offwind':
        hourly_generation = n.generators_t.p[f"DE0 0 0 offwind-float"]
        for c in ["offwind-ac", "offwind-dc"]:    
            rc_offwind = n.generators.index[n.generators.carrier == c].size
            for l in range(rc_offwind):
                hg = n.generators_t.p[f"DE0 0 {l} {c}"]
                hourly_generation = hourly_generation.add(hg, fill_value=0)
    elif carrier == 'onwind' or carrier == 'solar' or carrier == 'solar-hsat':
        hourly_generation = n.generators_t.p[f"DE0 0 {rc} {carrier}"]
    else:
        # Check if the generator exists with resource class 0 (for special carriers not in elif)
        if f"DE0 0 {rc} {carrier}" in n.generators_t.p:
            hourly_generation = n.generators_t.p[f"DE0 0 {rc} {carrier}"]
        elif f"DE0 0 0 {carrier}" in n.generators_t.p:
            hourly_generation = n.generators_t.p[f"DE0 0 0 {carrier}"]
        else:
            # Fallback for conventional carriers (e.g., CCGT, biomass)
            hourly_generation = n.generators_t.p[f"DE0 0 {carrier}"]
    return hourly_generation


# Function to get hourly generation per MW of any carrier (essentially calculates average capacity factors)
def get_hourly_generation_per_MW(carrier, n, rc=0):
    if carrier == 'offwind':
        hg = get_hourly_generation(carrier, n)
        p_nom = 0
        for c in ["offwind-ac", "offwind-dc", "offwind-float"]:    
            rc_offwind = n.generators.index[n.generators.carrier == c].size
            for l in range(rc_offwind):
                p_nom = p_nom + n.generators.p_nom_opt[f"DE0 0 {l} {c}"]
        # Align indices and sum row-wise across the three DataFrames, return as single-column DataFrame
        # Convert to single-column DataFrame and divide by total optimized capacity
        hourly_generation_per_MW = hg / p_nom
    elif carrier == 'onwind' or carrier == 'solar' or carrier == 'solar-hsat':
        hg = get_hourly_generation(carrier, n, rc)
        p_nom = n.generators.p_nom_opt[f"DE0 0 {rc} {carrier}"]
        hourly_generation_per_MW = hg / p_nom
    else:
        hg = get_hourly_generation(carrier, n, rc)
        if f"DE0 0 {rc} {carrier}" in n.generators.p_nom_opt:
            p_nom = n.generators.p_nom_opt[f"DE0 0 {rc} {carrier}"]
        elif f"DE0 0 0 {carrier}" in n.generators.p_nom_opt:
            p_nom = n.generators.p_nom_opt[f"DE0 0 0 {carrier}"]
        else:
            p_nom = n.generators.p_nom_opt[f"DE0 0 {carrier}"]
        hourly_generation_per_MW = hg / p_nom
    return hourly_generation_per_MW



# ___ Get CO2 emission data (still needs some fixing!) ___

# Function to get CO2 intensity of all carriers
def CO2_intensity_carriers(run_name):
    n_temp = pypsa.Network(f"resources/{run_name}/networks/base_s_1_elec_EP.nc")
    co2_intensity = n_temp.carriers.co2_emissions
    return co2_intensity


# Function to calculate annual CO2 emissions
def annual_CO2_emissions(run_name, year):
    co2_intensity = CO2_intensity_carriers(run_name)
    n = get_network(run_name)
    total_emissions = 0
    for carrier in co2_intensity.index:
        if carrier in n.generators.carrier.unique():
            hourly_generation = get_hourly_generation(carrier, n)
            annual_generation = hourly_generation[hourly_generation.index.year == year]
            emissions = (annual_generation.values.flatten() * co2_intensity[carrier]).sum() / 1e6  # Convert to MtCO2
            total_emissions += emissions
        else:
            print(f"Carrier {carrier} not found in the network generators.")
    return total_emissions


# Function to calculate total CO2 emissions
def total_CO2_emissions(run_name_or_n):
    """Calculate total CO2 emissions.

    Accepts either a run name (str) or a pypsa.Network object. If a network
    is provided the calculation runs directly on it; if a run name is given
    the function will load the network from the expected results directory.
    Returns total emissions (in MtCO2 as in the original code division by 1e6).
    """
    if isinstance(run_name_or_n, str):
        run_name = run_name_or_n
        co2_intensity = CO2_intensity_carriers(run_name)
        n = get_network(run_name)
    else:
        n = run_name_or_n
        if hasattr(n, "carriers") and "co2_emissions" in n.carriers.columns:
            co2_intensity = n.carriers.co2_emissions
        else:
            # No intensity data available
            return 0.0

    total_emissions = 0.0
    for carrier in co2_intensity.index:
        if carrier in n.generators.carrier.unique():
            hourly_generation = get_hourly_generation(carrier, n)
            emissions = (hourly_generation.values.flatten() * co2_intensity[carrier]).sum() / 1e6  # Convert to MtCO2
            total_emissions += emissions
    return total_emissions



# ___ Price statistics ___

# Get hourly marginal price at each node for any carrier
def get_hourly_marginal_price(n, bus="DE0 0"):
    price_t = n.buses_t.marginal_price[bus]
    return price_t


# Statistical quantities of marginal prices at node 'DE0 0'
def price_statistics(n, bus="DE0 0"):
    price_t = get_hourly_marginal_price(n, bus)
    stats = {
        "mean": price_t.mean(),
        "median": price_t.median(),
        "std_dev": price_t.std(),
        "min": price_t.min(),
        "max": price_t.max()
    }
    return stats


# Number of hours with prices above 5000 €/MWh at node 'DE0 0'
def hours_above_threshold(n, bus="DE0 0", threshold=5000):
    price_t = get_hourly_marginal_price(n, bus)
    hours = (price_t > threshold).sum()
    return hours


# Values in those hours
def prices_above_threshold(n, bus="DE0 0", threshold=5000):
    price_t = get_hourly_marginal_price(n, bus)
    prices = price_t[price_t > threshold]
    return prices
    

# Set values in those hours to threshold value
def cap_prices_above_threshold(n, bus="DE0 0", threshold=5000):
    price_t = get_hourly_marginal_price(n, bus)
    price_t[price_t > threshold] = threshold
    return price_t



# ___ Plots of price statistics ___

# Histogram of marginal prices at node 'DE0 0'
def plot_price_histogram(run_name, bus="DE0 0", bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, np.inf], save_to_file=True):
    n = get_network(run_name)
    price_t = get_hourly_marginal_price(n, bus)
    plt.figure(figsize=(10, 6))
    plt.hist(price_t, bins=bins, color=cm_2, alpha=0.7)
    plt.title(f'Histogram of Marginal Prices at {bus}')
    plt.xlabel('Marginal Price [€/MWh]')
    plt.ylabel('Frequency')
    plt.grid(True)
    if save_to_file:
        results_directory = f'../results/{run_name}'
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/price_histogram.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


# Price duration curve at node 'DE0 0'
def plot_price_duration_curve(run_name, bus="DE0 0", save_to_file=True):
    n = get_network(run_name)
    price_t = get_hourly_marginal_price(n, bus)
    sorted_prices = np.sort(price_t)[::-1]
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_prices, color=cm_3, linewidth=2) 
    # plt.title(f'Price Duration Curve at {bus}')
    plt.xlabel('Sorted Hours')
    plt.ylabel('Marginal Price [€/MWh]')
    plt.ylim(0, 1.5*sorted_prices[int(np.ceil(0.1*len(sorted_prices)))])  # Cap y-axis at 100 €/MWh
    plt.xlim(0, len(sorted_prices))
    plt.grid(True)
    if save_to_file:
        results_directory = f'../results/{run_name}'
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/price_duration_curve.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()



# ___ General statistics ___

# General statistics
def statistics(n, bus="DE0 0"):
    # Define boolean masks for carriers
    vre_i = n.generators.carrier.isin(['onwind', 'offwind-ac', 'offwind-dc', 'offwind-float', 'solar', 'solar rooftop', 'solar-hsat', 'ror', 'geothermal', 'biomass'])
    onwind_i = n.generators.carrier == 'onwind'
    offwind_i = n.generators.carrier.isin(['offwind-ac', 'offwind-dc', 'offwind-float'])
    solar_i = n.generators.carrier.isin(['solar', 'solar rooftop', 'solar-hsat'])
    gas_i = n.generators.carrier.isin(['CCGT', 'OCGT'])
    coal_i = n.generators.carrier.isin(['coal', 'lignite'])

    stats = {
        "Average Price": price_statistics(n, bus)["mean"],
        "Price Std Dev": price_statistics(n, bus)["std_dev"],
        "Generation Total": n.generators_t.p.sum().sum(),
        "Generation VRE": n.generators_t.p.loc[:, vre_i].sum().sum(),
        "Generation Onshore Wind": n.generators_t.p.loc[:, onwind_i].sum().sum(),
        "Generation Offshore Wind": n.generators_t.p.loc[:, offwind_i].sum().sum(),
        "Generation Solar": n.generators_t.p.loc[:, solar_i].sum().sum(),
        "Generation Hydro": n.generators_t.p.loc[:, n.generators.carrier == 'ror'].sum().sum(),
        "Generation Bio": n.generators_t.p.loc[:, n.generators.carrier == 'biomass'].sum().sum(),
        "Generation Geothermal": n.generators_t.p.loc[:, n.generators.carrier == 'geothermal'].sum().sum(),
        "Generation Gas": n.generators_t.p.loc[:, gas_i].sum().sum(),
        "Generation Coal": n.generators_t.p.loc[:, coal_i].sum().sum(),
        "Generation Oil": n.generators_t.p.loc[:, n.generators.carrier == 'oil'].sum().sum(),
        "Generation Nuclear": n.generators_t.p.loc[:, n.generators.carrier == 'nuclear'].sum().sum(),
        "VRE Generation Share": n.generators_t.p.loc[:, vre_i].sum().sum() / n.generators_t.p.sum().sum(),
        "Capacity Total": n.generators.p_nom_opt.sum(),
        "Capacity Onshore Wind": n.generators.p_nom_opt[onwind_i].sum(),
        "Capacity Offshore Wind": n.generators.p_nom_opt[offwind_i].sum(),
        "Capacity Solar": n.generators.p_nom_opt[solar_i].sum(),
        "Capacity Hydro": n.generators.p_nom_opt[n.generators.carrier == 'ror'].sum(),
        "Capacity Bio": n.generators.p_nom_opt[n.generators.carrier == 'biomass'].sum(),
        "Capacity Geothermal": n.generators.p_nom_opt[n.generators.carrier == 'geothermal'].sum(),
        "Capacity Gas": n.generators.p_nom_opt[gas_i].sum(),
        "Capacity Coal": n.generators.p_nom_opt[coal_i].sum(),
        "Capacity Oil": n.generators.p_nom_opt[n.generators.carrier == 'oil'].sum(),
        "Capacity Nuclear": n.generators.p_nom_opt[n.generators.carrier == 'nuclear'].sum(),
        "VRE Capacity Share": n.generators.p_nom_opt[vre_i].sum() / n.generators.p_nom_opt.sum(),
        "CO2 Emissions": total_CO2_emissions(n)
    }
    return stats



# ___ Plots of general statistics ___

# Plot comparison of key statistics for different scenarios
# This has to be adapted with the final run names
def compare_statistics_plot(stat, save_to_file=True):
    run_name_nocfd = "2030_without_CfD_v5"
    results_directory_nocfd = "../results/{}".format(run_name_nocfd)
    n_nocfd = pypsa.Network(f"{results_directory_nocfd}/networks/base_s_1_EP__2030.nc")
    stats_nocfd = statistics(n=n_nocfd)
    stat_nocfd = stats_nocfd[stat]
    run_name_cfd = "2030_with_CfD_v4"
    results_directory_cfd = "../results/{}".format(run_name_cfd)
    n_cfd = pypsa.Network(f"{results_directory_cfd}/networks/base_s_1_EP__2030.nc")
    stats_cfd = statistics(n=n_cfd)
    stat_cfd = stats_cfd[stat]
    run_name_cap = "2030_withouth_CfD_v6"
    results_directory_cap = "../results/{}".format(run_name_cap)
    n_cap = pypsa.Network(f"{results_directory_cap}/networks/base_s_1_EP__2030.nc")
    stats_cap = statistics(n=n_cap)
    stat_cap = stats_cap[stat]
    plt.figure(figsize=(8, 5))
    plt.bar(['No CfD', 'With CfD', 'With CM'], [stat_nocfd, stat_cfd, stat_cap], color=[cm_1, cm_2, cm_3])
    plt.ylabel(stat)
    plt.title(f'Comparison of {stat} across Scenarios')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_to_file:
        output_dir = "../results/my_plots"
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/comparison_{stat}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


# Plot installed capacities of all carriers
def plot_installed_capacities(run_name, save_to_file=True):
    n = get_network(run_name)
    # Define boolean masks for carriers
    onwind = n.generators.carrier == 'onwind'
    offwind = n.generators.carrier.isin(['offwind-ac', 'offwind-dc', 'offwind-float'])
    solar = n.generators.carrier.isin(['solar', 'solar rooftop', 'solar-hsat'])
    hydro = n.generators.carrier == 'ror'
    bio = n.generators.carrier == 'biomass'
    # geothermal = n.generators.carrier == 'geothermal'
    gas = n.generators.carrier.isin(['CCGT', 'OCGT'])
    # coal = n.generators.carrier.isin(['coal', 'lignite'])
    # oil = n.generators.carrier == 'oil'
    # nuclear = n.generators.carrier == 'nuclear'
    batteries = n.storage_units.carrier == 'battery'
    phs = n.storage_units.carrier == 'PHS'
    # h2 = n.storage_units.carrier == 'H2'
    # load = n.generators.carrier == 'load'
    carriers = [onwind, offwind, solar, hydro, bio, # geothermal, 
                gas, # coal, oil, nuclear, 
                batteries, phs] #, h2]
    labels = [
        'Onshore',
        'Offshore',
        'Solar',
        'Hydro',
        'Bio',
        # 'Geothermal',
        'Gas',
        # 'Coal',
        # 'Oil',
        # 'Nuclear',
        'Batteries',
        'PHS',
        # 'H2',
        # 'Lost Load'
    ]
    capacities = []
    for carrier in carriers:
        if carrier is batteries or carrier is phs: # or carrier is h2:
            cap = n.storage_units.p_nom_opt[carrier].sum() / 1e3  # Convert to GW
        else:
            cap = n.generators.p_nom_opt[carrier].sum() / 1e3  # Convert to GW
        capacities.append(cap)
    plt.figure(figsize=(8, 5))
    # use labels (strings) as x values so matplotlib can align heights correctly
    plt.bar(labels, capacities, color=cm_1)
    plt.title('Installed Capacities by Carrier [GW]')
    plt.ylabel('Installed Capacity [GW]')
    plt.xticks(fontsize='small')
    plt.grid(True)
    if save_to_file:
        results_directory = f'../results/{run_name}'
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/installed_capacities_all_carriers.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


# Plot total generation of all carriers
def plot_avg_annual_generation(run_name, save_to_file=True):
    n = get_network(run_name)
    # Define boolean masks for carriers
    onwind = n.generators.carrier == 'onwind'
    offwind = n.generators.carrier.isin(['offwind-ac', 'offwind-dc', 'offwind-float'])
    solar = n.generators.carrier.isin(['solar', 'solar rooftop', 'solar-hsat'])
    hydro = n.generators.carrier == 'ror'
    bio = n.generators.carrier == 'biomass'
    # geothermal = n.generators.carrier == 'geothermal'
    gas = n.generators.carrier.isin(['CCGT', 'OCGT'])
    # coal = n.generators.carrier.isin(['coal', 'lignite'])
    # oil = n.generators.carrier == 'oil'
    # nuclear = n.generators.carrier == 'nuclear'
    batteries = n.storage_units.carrier == 'battery'
    phs = n.storage_units.carrier == 'PHS'
    # h2 = n.storage_units.carrier == 'H2'
    load = n.generators.carrier == 'load'
    carriers = [onwind, offwind, solar, hydro, bio, # geothermal, 
                gas, # coal, oil, nuclear, 
                batteries, phs, #, h2, 
                load]
    generations = []
    for carrier in carriers:
        if carrier is batteries or carrier is phs: # or carrier is h2:
            gen = n.storage_units_t.p.loc[:, carrier].clip(lower=0).sum().sum() / 1e6  # Convert to TWh
        else:
            gen = n.generators_t.p.loc[:, carrier].sum().sum() / 1e6  # Convert to TWh
        gen /= (n.snapshot_weightings.objective.sum() / 8760.0)  # Average annual generation
        generations.append(gen)
    plt.figure(figsize=(8, 5))
    labels = [
        'Onshore',
        'Offshore',
        'Solar',
        'Hydro',
        'Bio',
        # 'Geothermal',
        'Gas',
        # 'Coal',
        # 'Oil',
        # 'Nuclear',
        'Batteries',
        'PHS',
        # 'H2',
        'Lost Load'
    ]
    plt.bar(labels, generations, color=cm_2)
    plt.title('Average Annual Generation by Carrier [TWh]')
    plt.ylabel('Average Annual Generation [TWh]')
    plt.xticks(fontsize='small')
    plt.grid(True)
    if save_to_file:
        results_directory = f'../results/{run_name}'
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/average_annual_generation_all_carriers.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show() 


# Plot installed capacities of a carrier across different model iterations
# This has to be adapted with the final run names
def plot_installed_capacities_across_scenarios(carrier, save_to_file=True): 
    run_name_nocfd = "2030_without_CfD_v5"
    results_directory_nocfd = "../results/{}".format(run_name_nocfd)
    n_nocfd = pypsa.Network(f"{results_directory_nocfd}/networks/base_s_1_EP__2030.nc")
    c_inst_nocfd = n_nocfd.generators.p_nom_opt[n_nocfd.generators.carrier == carrier].sum()
    run_name_cfd = "2030_with_CfD_v4"
    results_directory_cfd = "../results/{}".format(run_name_cfd)
    n_cfd = pypsa.Network(f"{results_directory_cfd}/networks/base_s_1_EP__2030.nc")
    c_inst_cfd = n_cfd.generators.p_nom_opt[n_cfd.generators.carrier == carrier].sum()
    run_name_cap = "2030_withouth_CfD_v6"
    results_directory_cap = "../results/{}".format(run_name_cap)
    n_cap = pypsa.Network(f"{results_directory_cap}/networks/base_s_1_EP__2030.nc")
    c_inst_cap = n_cap.generators.p_nom_opt[n_cap.generators.carrier == carrier].sum()
    plt.figure(figsize=(8, 5))
    categories = ['Without CfD', 'With CfD', 'With Capacity Component']
    capacities = [c_inst_nocfd, c_inst_cfd, c_inst_cap]
    color = ''
    if carrier == 'offwind':
        color = mcblue
    elif carrier == 'onwind':
        color = mcgreen
    elif carrier == 'solar':
        color = mcorange
    elif carrier == 'solar rooftop':
        color = mcred
    plt.bar(categories, capacities, color=color)
    carrier_name = ""
    if carrier == 'offwind':
        carrier_name = "Offshore Wind"
    elif carrier == 'onwind':
        carrier_name = "Onshore Wind"
    elif carrier == 'solar':
        carrier_name = "Ground-Mounted Solar PV"
    elif carrier == 'solar rooftop':
        carrier_name = "Rooftop Solar PV"
    plt.title(f'Installed Capacities for {carrier_name} [MW]')
    plt.ylabel('Installed Capacity [MW]')
    plt.grid(True)
    if save_to_file:
        output_dir = '../results/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/installed_capacities_{carrier}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


# ___ Resource classes ___

# Plot map of resource classes for a given carrier
def plot_resource_classes_map(run_name, carrier, save_to_file=True):
    dir = f'../resources/{run_name}'
    file = f'{dir}/regions_by_class_1_{carrier}.geojson'
    gdf = gpd.read_file(file)
    
    # Check if 'bin' column exists
    if 'bin' not in gdf.columns:
        print("Error: 'bin' column not found in GeoJSON file.")
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(column='bin', ax=ax, legend=True, cmap=colormap_for_carrier(carrier), legend_kwds={'label': "Resource Class"})
    
    plt.title(f"Regional Resource Classes for {carrier_full_name(carrier)}")
    plt.axis('off')
    
    if save_to_file:
        output_dir = f'../results/{run_name}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/map_{os.path.basename(file).replace(".geojson", "")}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()
    


# ___ Revenue statistics ___

# Load cost data
def get_costs(run_name):
    cost_file = f"../resources/{run_name}/costs_2030.csv"
    costs = pd.read_csv(cost_file, index_col=[0, 1]).sort_index()
    costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
    costs.loc[costs.unit.str.contains("/GW"), "value"] /= 1e3
    
    # ERROR FIX: Unstack to turn parameters (lifetime, investment, etc.) into columns
    costs = costs.value.unstack(level=1).groupby("technology").sum(min_count=1)
    return costs


# Revenue of a carrier and rc for a given year
def annual_revenue(costs, carrier, year, n, rc, bus="DE0 0"):
    hg = get_hourly_generation(carrier, n, rc)
    hg_year = hg[hg.index.year == year]
    price = get_hourly_marginal_price(n, bus)
    price_year = price[price.index.year == year]

    # Defaults and helper
    if carrier == 'onwind':
        vom = costs.at[carrier, "VOM"]  # in €/MWh
    else: 
        vom = 0.0

    rev_year = (hg_year.values.flatten() * (price_year.values.flatten() - vom)).sum()
    return rev_year


# Annual revenue per MW of a carrier and rc for a given year
def annual_revenue_per_MW(costs, carrier, year, n, rc, bus="DE0 0"):
    ann_rev = annual_revenue(costs, carrier, year, n, rc, bus)
    p_nom_opt = n.generators.p_nom_opt[f"DE0 0 {rc} {carrier}"] if carrier != 'offwind' else sum(
        n.generators.p_nom_opt[f"DE0 0 {l} {c}"] 
        for c in ["offwind-ac", "offwind-dc", "offwind-float"] 
        for l in range(n.generators.index[n.generators.carrier == c].size)
    )
    rev_per_MW_year = ann_rev / p_nom_opt if p_nom_opt != 0 else np.nan
    return rev_per_MW_year


# Strike price of a carrier and rc in € per MWh for the production-based CfD
# -> averaged market value across all years for the specific rc
def strike_price_PB(costs, carrier, n, rc, bus="DE0 0"):
    gen_tot = n.generators_t.p[f"DE0 0 {rc} {carrier}"].sum()
    # capital_cost = n.generators.capital_cost[f"DE0 0 {rc} {carrier}"]
    # cap = n.generators.p_nom_opt[f"DE0 0 {rc} {carrier}"]
    # flh = gen_tot / cap if cap != 0 else 0
    # if carrier == 'onwind':
    #     vom = costs.at[carrier, "VOM"]  # in €/MWh
    # else:
    #     vom = 0.0
    # strike_price = capital_cost / flh - vom  # in €/MWh
    # return strike_price
    years = n.generators_t.p.index.year.unique()
    revenues = sum([annual_revenue(costs, carrier, year, n, rc, bus) for year in years])
    strike_price = revenues / gen_tot if gen_tot != 0 else np.nan  # in €/MWh
    return strike_price


# Strike price of a carrier and rc in € per MWh for the production-independent CfD
# -> averaged market value across all years for all rcs
def strike_price_PI(costs, carrier, n, rc, bus="DE0 0"):
    gen_tot = sum(sum(n.generators_t.p[f"DE0 0 {l} {carrier}"] for l in range(n.generators.index[n.generators.carrier == carrier].size)))
    years = n.generators_t.p.index.year.unique()
    revenues = sum([annual_revenue(costs, carrier, year, n, rc, bus) for year in years for rc in range(n.generators.index[n.generators.carrier == carrier].size)])
    strike_price = revenues / gen_tot if gen_tot != 0 else np.nan  # in €/MWh
    return strike_price


# Strike price of a carrier in €/MW/year for the capacity-based CfD
# -> carrier-wide average annual market revenue per MW of installed capacity
def strike_price_CB(costs, carrier, n, rc, bus="DE0 0"):
    years = n.generators_t.p.index.year.unique()
    n_years = len(years)
    n_rc = n.generators.index[n.generators.carrier == carrier].size
    revenues = sum(annual_revenue(costs, carrier, year, n, i, bus) for year in years for i in range(n_rc))
    p_tot = sum(n.generators.p_nom_opt[f"DE0 0 {i} {carrier}"] for i in range(n_rc))
    strike_price = revenues / (p_tot * n_years) if p_tot != 0 else np.nan  # in €/MW/year
    return strike_price


# Revenue of a carrier and rc for a given year under a conventional CfD
def annual_revenue_PB(costs, carrier, year, n, rc, strike_price=None, bus="DE0 0"):
    # Requires strike_price_PB as input for the specific carrier and rc
    hg = get_hourly_generation(carrier, n, rc)
    hg_year = hg[hg.index.year == year]
    if strike_price is None:
        strike_price = costs.at[carrier, "strike_price"]  # in €/MWh
    rev_year_cfd = hg_year.values.sum() * strike_price 
    return rev_year_cfd


# Revenue per MW of a carrier and rc for a given year under a conventional CfD
def annual_revenue_PB_per_MW(costs, carrier, year, n, rc, strike_price=None, bus="DE0 0"):
    ann_rev = annual_revenue_PB(costs, carrier, year, n, rc, strike_price, bus)
    p_nom_opt = n.generators.p_nom_opt[f"DE0 0 {rc} {carrier}"]
    rev_per_MW_year_cfd = ann_rev / p_nom_opt if p_nom_opt != 0 else np.nan
    return rev_per_MW_year_cfd


# Revenue of a carrier and rc for a given year under a financial CfD with average technology profile as reference profile
def annual_revenue_PI(costs, carrier, year, n, rc, strike_price=None, bus="DE0 0"):
    if strike_price is None:
        strike_price = costs.at[carrier, "strike_price"]  # in €/MWh
    hg_car = sum([get_hourly_generation(carrier, n, i) for i in range(n.generators.index[n.generators.carrier == carrier].size)])
    hg_car_year = hg_car[hg_car.index.year == year]
    gen_tot = hg_car_year.sum()
    rev_tot = sum(annual_revenue(costs, carrier, year, n, i, bus) for i in range(n.generators.index[n.generators.carrier == carrier].size))
    p_tot = sum(n.generators.p_nom_opt[f"DE0 0 {i} {carrier}"] for i in range(n.generators.index[n.generators.carrier == carrier].size))
    p = n.generators.p_nom_opt[f"DE0 0 {rc} {carrier}"]
    rev_year_cfd = annual_revenue(costs, carrier, year, n, rc, bus) + (strike_price * gen_tot - rev_tot) * p / p_tot
    return rev_year_cfd


# Revenue per MW of a carrier and rc for a given year under a financial CfD with average technology profile as reference profile
def annual_revenue_PI_per_MW(costs, carrier, year, n, rc, strike_price=None, bus="DE0 0"):
    ann_rev = annual_revenue_PI(costs, carrier, year, n, rc, strike_price, bus)
    p = n.generators.p_nom_opt[f"DE0 0 {rc} {carrier}"]
    rev_per_MW_year_cfd = ann_rev / p if p != 0 else np.nan
    return rev_per_MW_year_cfd


# Revenue of a carrier and rc for a given year under a capacity-based financial CfD with average technology profile as reference profile
def annual_revenue_CB(costs, carrier, year, n, rc, strike_price=None, bus="DE0 0"):
    if strike_price is None:
        strike_price = costs.at[carrier, "strike_price"]  # in €/MW
    rev_tot = sum(annual_revenue(costs, carrier, year, n, i, bus) for i in range(n.generators.index[n.generators.carrier == carrier].size))
    p = n.generators.p_nom_opt[f"DE0 0 {rc} {carrier}"]
    p_tot = sum(n.generators.p_nom_opt[f"DE0 0 {i} {carrier}"] for i in range(n.generators.index[n.generators.carrier == carrier].size))
    rev_year_cfd = annual_revenue(costs, carrier, year, n, rc, bus) + strike_price * p - rev_tot * p / p_tot
    return rev_year_cfd


# Revenue per MW of a carrier and rc for a given year under a capacity-based financial CfD with average technology profile as reference profile
def annual_revenue_CB_per_MW(costs, carrier, year, n, rc, strike_price=None, bus="DE0 0"):
    ann_rev = annual_revenue_CB(costs, carrier, year, n, rc, strike_price, bus)
    p = n.generators.p_nom_opt[f"DE0 0 {rc} {carrier}"]
    rev_per_MW_year_cfd = ann_rev / p if p != 0 else np.nan
    return rev_per_MW_year_cfd


# Annual revenue risk of a carrier and rc across all years
def annual_revenue_risk(costs, carrier, n, rc, bus="DE0 0"):
    years = n.generators_t.p.index.year.unique()
    revenues = [annual_revenue_per_MW(costs, carrier, year, n, rc, bus) for year in years]
    mean_revenue = np.mean(revenues)
    std_dev_revenue = np.std(revenues)
    risk = std_dev_revenue / mean_revenue if mean_revenue != 0 else np.nan
    return risk


# Annual Revenue Statistics for a given carrier and rc across all years
def annual_revenue_statistics(costs, carrier, n, rc, bus="DE0 0"):
    years = n.generators_t.p.index.year.unique()
    revenues = [annual_revenue_per_MW(costs, carrier, year, n, rc, bus) for year in years]
    stats = {
        "mean": np.mean(revenues),
        "median": np.median(revenues),
        "std_dev": np.std(revenues),
        "risk": np.std(revenues) / np.mean(revenues) if np.mean(revenues) != 0 else np.nan,
        "min": np.min(revenues),
        "max": np.max(revenues)
    }
    return stats


# Monte Carlo Simulation of price years based on a Markov Chain model
def years_mcs(run_name, n, T=30, N=10000):
    # Distribution of years into Low, Mid, High buckets
    price_year_dist = [0.2, 0.6, 0.2]
    
    # Transition Probability Matrix
    # Rows: Current State, Columns: Next State
    p_trans = pd.DataFrame(
        [
            [0.5, 0.5, 0.0],  # From Low
            [1/6, 2/3, 1/6],  # From Mid
            [0.0, 0.5, 0.5]   # From High
        ],
        index=['Low', 'Mid', 'High'],
        columns=['Low', 'Mid', 'High']
    )

    # Get available years from price data
    price = get_hourly_marginal_price(n)
    years = price.index.year.unique()
    n_years = len(years)
    
    # Define buckets based on distribution percentages
    cutoff_low = int(n_years * price_year_dist[0])
    cutoff_mid = cutoff_low + int(n_years * price_year_dist[1])
    
    # Sort years if needed, or assume chronological order implies price regime? 
    # Usually for price duration one sorts by price, but here we just slice the array of years.
    # If the years are sorted chronologically, this just takes the first 20% as 'low', etc. 
    # Note: If 'Low' implies low price years, you might want to sort `years` by average price first.
    # Assuming the user just wants to slice the existing list:
    years_low = years[:cutoff_low]
    years_mid = years[cutoff_low:cutoff_mid]
    years_high = years[cutoff_mid:]
    
    state_pools = {
        'Low': years_low,
        'Mid': years_mid,
        'High': years_high
    }
    states = ['Low', 'Mid', 'High']

    # Initialize results array (N simulations x T years)
    results = np.zeros((N, T), dtype=int)

    for i in range(N):
        # 1. Randomly choose starting year from ALL available years
        start_year = np.random.choice(years)
        results[i, 0] = start_year
        
        # Determine initial state based on the chosen start year
        if start_year in years_low:
            current_state = 'Low'
        elif start_year in years_mid:
            current_state = 'Mid'
        else:
            current_state = 'High'
            
        # 2. Generate subsequent years based on transition matrix
        for t in range(1, T):
            # Get probabilities for transitioning from current_state
            probs = p_trans.loc[current_state].values
            probs = probs / probs.sum() # Ensure sums to 1 to avoid float errors
            
            # Sample next state
            next_state = np.random.choice(states, p=probs)
            
            # Sample specific year from the next state's pool
            # (Assumes non-empty pools)
            next_year = np.random.choice(state_pools[next_state])
            
            results[i, t] = next_year
            current_state = next_state
    # Save to CSV
    df_results = pd.DataFrame(results)
    os.makedirs(f"../results/{run_name}/MCS", exist_ok=True)
    df_results.to_csv(f"../results/{run_name}/MCS/years_mcs_T={T}_N={N}.csv", index=False)

    return results


# Monte Carlo Simulation of revenue years based on a Markov Chain model
def revenue_mcs(costs, run_name, n, T=30, N=10000):
    vre_carriers = ['offwind-ac', 'offwind-dc', 'offwind-float', 'onwind', 'solar', 'solar-hsat'] #'offwind', 
    # Read the CSV file produced by years_mcs
    file_path = f"../results/{run_name}/MCS/years_mcs_T={T}_N={N}.csv"
    try:
        yrs_mcs = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}. Please run years_mcs first.")
        return None    
    # --- Pre-calculate revenues outside the main loop ---
    
    # 1. Identify all unique years that appear in our simulation
    unique_years = np.unique(yrs_mcs)

    # 2. Calculate the total number of streams and pre-compute strike prices
    total_streams = 0
    stream_info = []  # list of (carrier, rc) tuples matching the stream order
    for carrier in vre_carriers:
        if n.generators.index[n.generators.carrier == carrier].size > 0:
            n_rc = n.generators.index[n.generators.carrier == carrier].size
        else:
            n_rc = 1
        for rc in range(n_rc):
            stream_info.append((carrier, rc))
        total_streams += n_rc

    # Pre-compute strike prices once per (carrier, rc) — these are year-independent
    strike_prices = {}
    for carrier, rc in stream_info:
        strike_prices[(carrier, rc)] = {
            'pb': strike_price_PB(costs, carrier, n, rc),
            'pi': strike_price_PI(costs, carrier, n, rc),
            'cb': strike_price_CB(costs, carrier, n, rc),
        }
        
    # 3. Create a lookup table: year -> array of revenues for all streams
    revenue_lookup = {}
    for year in unique_years:
        revenues_mb = []
        revenues_pb = []
        revenues_pi = []
        revenues_cb = []
        for carrier, rc in stream_info:
            sp = strike_prices[(carrier, rc)]
            rev_mb = annual_revenue_per_MW(costs, carrier, year, n, rc)
            rev_pb = annual_revenue_PB_per_MW(costs, carrier, year, n, rc, strike_price=sp['pb'])
            rev_pi = annual_revenue_PI_per_MW(costs, carrier, year, n, rc, strike_price=sp['pi'])
            rev_cb = annual_revenue_CB_per_MW(costs, carrier, year, n, rc, strike_price=sp['cb'])
            revenues_mb.append(rev_mb)
            revenues_pb.append(rev_pb)
            revenues_pi.append(rev_pi)
            revenues_cb.append(rev_cb)
        revenue_lookup[year] = {
            'mb': np.array(revenues_mb),
            'pb': np.array(revenues_pb),
            'pi': np.array(revenues_pi),
            'cb': np.array(revenues_cb),
        }

    # 4. Initialize result arrays — one per revenue scenario
    # Shape: (N simulations, T years, num_streams)
    rev_mcs_mb = np.zeros((N, T, total_streams), dtype=float)
    rev_mcs_pb = np.zeros((N, T, total_streams), dtype=float)
    rev_mcs_pi = np.zeros((N, T, total_streams), dtype=float)
    rev_mcs_cb = np.zeros((N, T, total_streams), dtype=float)
    
    # 5. Broadcast the pre-calculated revenues into the result arrays
    for year in unique_years:
        mask = (yrs_mcs == year)
        if np.any(mask):
            rev_mcs_mb[mask] = revenue_lookup[year]['mb']
            rev_mcs_pb[mask] = revenue_lookup[year]['pb']
            rev_mcs_pi[mask] = revenue_lookup[year]['pi']
            rev_mcs_cb[mask] = revenue_lookup[year]['cb']
            
    # 6. Reshape to (N, T * total_streams) and save each to a separate CSV
    mcs_dir = f"../results/{run_name}/MCS"
    os.makedirs(mcs_dir, exist_ok=True)

    scenarios = {
        'mb': rev_mcs_mb,
        'pb': rev_mcs_pb,
        'pi': rev_mcs_pi,
        'cb': rev_mcs_cb,
    }
    for label, arr in scenarios.items():
        df = pd.DataFrame(arr.reshape(N, -1))
        df.to_csv(f"{mcs_dir}/revenue_mcs_{label}_T={T}_N={N}.csv", index=False)

    return scenarios


# Lifetime revenue per MW for a given carrier and resource class from MCS results
def lifetime_revenues_mcs_per_MW(run_name, n, carrier, rc, T=20, N=10000, scenario='mb'):
    # Read the CSV file produced by revenue_mcs
    file_path = f"../results/{run_name}/MCS/revenue_mcs_{scenario}_T={T}_N={N}.csv"
    try:
        df_rev_mcs = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}. Please run revenue_mcs first.")
        return None
    
    # Convert to numpy array
    rev_mcs_flat = df_rev_mcs.values # Shape (N, T * total_streams)
    
    # Determine the index for the requested carrier and resource class to extract the correct column
    vre_carriers = ['offwind-ac', 'offwind-dc', 'offwind-float', 'onwind', 'solar', 'solar-hsat']
    current_idx = 0
    target_idx = -1
    
    for c in vre_carriers:
        # Determine number of resource classes for this carrier
        # Must match logic in revenue_mcs exactly
        n_rc_actual = n.generators.index[n.generators.carrier == c].size
        n_rc = n_rc_actual if n_rc_actual > 0 else 1
        
        if c == carrier:
            if rc >= n_rc:
                raise ValueError(f"Resource class {rc} out of bounds for carrier {carrier} (max {n_rc-1})")
            target_idx = current_idx + rc
            
        current_idx += n_rc
        
    total_streams = current_idx
    
    if target_idx == -1:
         raise ValueError(f"Carrier {carrier} not found in VRE carriers list: {vre_carriers}")

    # Reshape the array: (N, T, total_streams)
    # The csv was flattened from (N, T, streams) to (N, T*streams)
    rev_mcs = rev_mcs_flat.reshape(N, T, total_streams)
    
    # Extract revenues for the specific carrier/rc: Shape (N, T)
    stream_revenues = rev_mcs[:, :, target_idx]
    
    # Calculate aggregated lifetime returns (sum over T years)
    lifetime_revenues = stream_revenues.sum(axis=1) # Shape (N,)
    
    return lifetime_revenues


# Average lifetime revenue per MW for a given carrier and resource class from MCS results
def avg_lifetime_revenues_mcs_per_MW(run_name, n, carrier, rc, T=0, N=10000, scenario='mb'):
    if T == 0:
        if carrier == 'solar' or carrier == 'solar-hsat':
            T = 40
        elif carrier == 'offwind' or carrier == 'offwind-ac' or carrier == 'offwind-dc' or carrier == 'onwind':
            T = 30
        elif carrier == 'offwind-float':
            T = 20
        else:
            T = 25
    lifetime_revenues = lifetime_revenues_mcs_per_MW(run_name, n, carrier, rc, T, N, scenario)
    if lifetime_revenues is None:
        return None
    avg_lifetime_revenue = lifetime_revenues.mean()
    return avg_lifetime_revenue


# Lifetime revenue per MW for a given carrier and resource class from MCS results
def avg_lifetime_revenues_mcs(run_name, n, carrier, rc, T=0, N=10000, scenario='mb'):
    if T == 0:
        if carrier == 'solar' or carrier == 'solar-hsat':
            T = 40
        elif carrier == 'offwind' or carrier == 'offwind-ac' or carrier == 'offwind-dc' or carrier == 'onwind':
            T = 30
        elif carrier == 'offwind-float':
            T = 20
        else:
            T = 25
    avg_rev = avg_lifetime_revenues_mcs_per_MW(run_name, n, carrier, rc, T, N, scenario)
    avg_lifetime_revenue = avg_rev * n.generators.p_nom_opt[f"DE0 0 {rc} {carrier}"]
    return avg_lifetime_revenue


# Standard deviation of lifetime revenue per MW for a given carrier and resource class from MCS results
def std_dev_lifetime_revenues_mcs_per_MW(run_name, n, carrier, rc, T=0, N=10000, scenario='mb'):
    if T == 0:
        if carrier == 'solar' or carrier == 'solar-hsat':
            T = 40
        elif carrier == 'offwind' or carrier == 'offwind-ac' or carrier == 'offwind-dc' or carrier == 'onwind':
            T = 30
        elif carrier == 'offwind-float':
            T = 20
        else:
            T = 25
    lifetime_revenues = lifetime_revenues_mcs_per_MW(run_name, n, carrier, rc, T, N, scenario)
    if lifetime_revenues is None:
        return None
    std_dev_lifetime_revenue = lifetime_revenues.std()
    return std_dev_lifetime_revenue


# Normalized risk (std dev / mean) of lifetime revenue per MW for a given carrier and resource class from MCS results
def risk_lifetime_revenues_mcs_per_MW(run_name, n, carrier, rc, T=0, N=10000, scenario='mb'):
    if T == 0:
        if carrier == 'solar' or carrier == 'solar-hsat':
            T = 40
        elif carrier == 'offwind' or carrier == 'offwind-ac' or carrier == 'offwind-dc' or carrier == 'onwind':
            T = 30
        elif carrier == 'offwind-float':
            T = 20
        else:
            T = 25
    lifetime_revenues = lifetime_revenues_mcs_per_MW(run_name, n, carrier, rc, T, N, scenario)
    if lifetime_revenues is None:
        return None
    mean_revenue = lifetime_revenues.mean()
    std_dev_revenue = lifetime_revenues.std()
    risk_lifetime_revenue = std_dev_revenue / mean_revenue if mean_revenue != 0 else np.nan
    return risk_lifetime_revenue


# Average lifetime revenue per MW for a given carrier and resource class from MCS results
def avg_lifetime_revenues_mcs_carrier_per_MW(run_name, n, carrier, T=0, N=10000, scenario='mb'):
    if T == 0:
        if carrier == 'solar' or carrier == 'solar-hsat':
            T = 40
        elif carrier == 'offwind' or carrier == 'offwind-ac' or carrier == 'offwind-dc' or carrier == 'onwind':
            T = 30
        elif carrier == 'offwind-float':
            T = 20
        else:
            T = 25
    lifetime_revenues = 0.0
    p_tot = 0.0
    for rc in range(n.generators.index[n.generators.carrier == carrier].size):
        lifetime_revenues_rc = avg_lifetime_revenues_mcs(run_name, n, carrier, rc, T, N, scenario)
        lifetime_revenues += lifetime_revenues_rc
        p_tot += n.generators.p_nom_opt[f"DE0 0 {rc} {carrier}"]
    if lifetime_revenues is None:
        return None
    avg_lifetime_revenue = lifetime_revenues / p_tot
    return avg_lifetime_revenue


# Annuity calculated by PyPSA-EUR Script with default assumptions
def annuity_old(run_name, carrier, costs):        
    # Fill missing values like in default config
    fill_values = {
        "FOM": 0,
        "VOM": 0,
        "efficiency": 1,
        "fuel": 0,
        "investment": 0,
        "lifetime": 25,
        "CO2 intensity": 0,
        "discount rate": 0.07,
        "standing losses": 0
    }
    costs = costs.fillna(fill_values)

    # Defaults and helper
    carrier_key = carrier
    if carrier in ['offwind-ac', 'offwind-dc']:
        carrier_key = 'offwind'
    elif carrier == 'solar-hsat':
        carrier_key = 'solar-utility single-axis tracking'
    elif carrier == 'solar':
        carrier_key = 'solar-utility' 
    
    # Handle missing keys gracefully
    if carrier_key not in costs.index:
        # print(f"Warning: {carrier_key} not found in costs. Using default lifetime 25.")
        t = 25.0
    else:
        t = costs.at[carrier_key, "lifetime"]
        
    r = costs.at[carrier_key, "discount rate"]

    annuity = r / (1.0 - 1.0 / (1.0 + r) ** t)

    return annuity


# Expected lifetime return on investment for a given carrier and resource class from MCS results
def fixed_costs(run_name, n, carrier, rc, costs):
    # ann_inv = 0.0
    # if f"DE0 0 {rc} {carrier}" in n.generators.capital_cost:
    #     ann_inv = n.generators.capital_cost[f"DE0 0 {rc} {carrier}"]
    # elif f"DE0 0 0 {carrier}" in n.generators.capital_cost:
    #     ann_inv = n.generators.capital_cost[f"DE0 0 0 {carrier}"]
    # else:
    #     # Fallback for conventional carriers (e.g., CCGT, biomass)
    #     ann_inv = n.generators.capital_cost[f"DE0 0 {carrier}"]
    # print(f"Annualized investment costs of DE0 0 {rc} {carrier}: {ann_inv} €/MW")
    # n = 0.0
    # if carrier in ['offwind', 'offwind-ac', 'offwind-dc', 'offwind-float', 'onwind']:
    #     n = 30.0 # From default PyPSA assumptions
    # elif carrier in ['solar', 'solar-hsat']:
    #     n = 40.0 # From default PyPSA assumptions
    # else:
    #     return print(f"Error: {carrier} lifetime not found.")
    # r = 0.07 # From default PyPSA assumptions
    # annuity = r / (1.0 - 1.0 / (1.0 + r) ** n)
    # print(f"Annuity of DE0 0 {rc} {carrier}: {annuity}")
    # invest = ann_inv / (annuity * n)
        
    # Fill missing values like in default config
    fill_values = {
        "FOM": 0,
        "VOM": 0,
        "efficiency": 1,
        "fuel": 0,
        "investment": 0,
        "lifetime": 25,
        "CO2 intensity": 0,
        "discount rate": 0.07,
        "standing losses": 0
    }
    costs = costs.fillna(fill_values)

    # Defaults and helper
    carrier_key = carrier
    if carrier in ['offwind-ac', 'offwind-dc']:
        carrier_key = 'offwind'
    elif carrier == 'solar-hsat':
        carrier_key = 'solar-utility single-axis tracking'
    elif carrier == 'solar':
        carrier_key = 'solar-utility' 
    
    # Handle missing carrier gracefully
    if carrier_key not in costs.index:
        # print(f"Warning: {carrier_key} not found in costs. Using defaults.")
        lifetime = 25.0
        invest = 0.0
        fom = 0.0
    else:
        lifetime = costs.at[carrier_key, "lifetime"]
        invest = costs.at[carrier_key, "investment"]
        fom = costs.at[carrier_key, "FOM"]

    # # Number of weather years in the model run
    # nyears = n.snapshot_weightings.objective.sum() / 8760.0 

    # # Annuity
    # annuity = annuity_old(run_name, carrier, costs=costs)

    # # Annualized investment + capital costs aggregated over nyears
    # cc_ann = n.generators.capital_cost[f"DE0 0 {rc} {carrier}"]

    # # Total investment + capital costs
    # cc_full = cc_ann * lifetime / nyears

    # # Total fixed costs (invest + FOM)
    # invest_fom = cc_full - (annuity * lifetime -1) * invest

    # # Annualized fixed costs (invest + FOM) for nyears 
    # invest_fom = cc_ann - (annuity - 1 / lifetime) * invest * nyears
    invest_fom = invest * (1 + fom / 100 * lifetime)
    
    # Default calculation of annualized fixed costs
    # invest_fom = invest * (annuity + fom / 100) * nyears # Default base calculation

    
    # Specific Offshore Logic with Connection Costs
    if "offwind" in carrier:
        try:
            # 1. Get distance from profile
            profile_file = f"../resources/{run_name}/profile_1_{carrier}.nc"
            if not os.path.exists(profile_file):
                 # Fallback/Try looking in generic resources if run-specific doesn't exist
                 profile_file = f"../resources/profile_1_{carrier}.nc"
            
            # Helper to match flattened index logic
            def flatten_idx(t): return " ".join(map(str, t))

            with xr.open_dataset(profile_file) as ds:
                if "year" in ds.indexes:
                    ds = ds.sel(year=ds.year.min(), drop=True)
                ds_stacked = ds.stack(bus_bin=["bus", "bin"])
                distance_k = ds_stacked["average_distance"].to_pandas()
                distance_k.index = distance_k.index.map(flatten_idx)
                
                # Construct lookup key: "DE0 0 {rc}" for bus+bin
                # Usually profile bus is "DE0 0", bin is integer rc
                lookup_key = f"DE0 0 {rc}" 
                if lookup_key in distance_k.index:
                    dist = distance_k[lookup_key]
                else:
                    print(f"Warning: Could not find distance for {lookup_key} in profile. Using 0.")
                    dist = 0
            
            # 2. Get Parameters (hardcoded defaults matching standard config)
            # Default line_length_factor in PyPSA-Eur config.default.yaml is 1.25
            # line_length_factor = 1.0
            line_length_factor = 1.25 
            # Default landfall_length is 0.0 unless specified in config["renewable"][carrier]["landfall_length"]
            # Checked iteration_zero.yaml and config.default.yaml, it is not set, so 0.0.
            landfall_length = 0.0
            if carrier == 'offwind-ac':
                landfall_length = 20.0
            if carrier == 'offwind-dc':
                landfall_length = 30.0
            if carrier == 'offwind-float':
                landfall_length = 40.0
            
            
            # 3. Calculate Connection Costs (Investment)
            # Keys based on add_electricity.py logic: car + "-connection-submarine"
            sub_key = f"{carrier}-connection-submarine"
            under_key = f"{carrier}-connection-underground"
            station_key = f"{carrier}-station"
            
            # Check if specific keys exist, otherwise fallback? 
            # add_electricity logic implies they must exist.
            
            check_keys = [sub_key, under_key, station_key]
            if all(k in costs.index for k in check_keys):
                inv_sub = costs.at[sub_key, "investment"]
                inv_under = costs.at[under_key, "investment"]
                inv_station = costs.at[station_key, "investment"]
                
                # # Formula
                # conn_inv = line_length_factor * (dist * inv_sub + landfall_length * inv_under)

                # # Total Investment Calculation               
                # total_invest_overnight = turb_invest + inv_station + conn_inv
                
                # # Station + Turbine
                # # Note: 'invest' variable already holds the turbine investment (from 'offwind' or 'offwind-float')
                # # But for AC/DC 'invest' came from 'offwind', which is correct.
                # # For float it should come from 'offwind-float'.
                
                # Recalculate base keys just to be safe
                if carrier == "offwind-float":
                    turb_invest = costs.at["offwind-float", "investment"]
                    turb_fom = costs.at["offwind-float", "FOM"]
                    turb_lifetime = costs.at["offwind-float", "lifetime"]
                else:
                    turb_invest = costs.at["offwind", "investment"]
                    turb_fom = costs.at["offwind", "FOM"]
                    turb_lifetime = costs.at["offwind", "lifetime"]
                
                # Turbine
                # Annualized capital + investment costs
                # tc_turbine = turb_invest * (annuity_old(run_name, carrier, costs=costs) + turb_fom/100) * nyears # * turb_lifetime)
                # Total fixed costs
                fc_turbine = turb_invest * (1 + turb_fom * turb_lifetime)
                
                fom_station = costs.at[station_key, "FOM"] # This is set to zero in default config
                life_station = costs.at[station_key, "lifetime"] # This is set to 25 years in default config
                # Annualized capital + investment costs 
                # tc_station = inv_station * (annuity_old(run_name, station_key, costs=costs) + fom_station/100) * nyears
                # Total fixed costs
                fc_station = inv_station * (1 + fom_station / 100 * life_station)

                fom_sub = costs.at[sub_key, "FOM"] # This is set to zero in default config
                life_sub = costs.at[sub_key, "lifetime"] # This is set to 25 years in default config
                # Annualized capital + investment costs 
                # tc_sub = (line_length_factor * dist * inv_sub) * (annuity_old(run_name, sub_key, costs=costs) + fom_sub/100) * nyears
                # Total fixed costs
                fc_sub = line_length_factor * dist * inv_sub * (1 + fom_sub / 100 * life_sub)
                
                fom_under = costs.at[under_key, "FOM"] # This is set to zero in default config
                life_under = costs.at[under_key, "lifetime"] # This is set to 25 years in default config
                # Annualized capital + investment costs
                # tc_under = (line_length_factor * landfall_length * inv_under) * (annuity_old(run_name, under_key, costs=costs) + fom_under/100) * nyears
                # Total fixed costs
                fc_under = line_length_factor * landfall_length * inv_under * (1 + fom_under / 100 * life_under)
                
                # Annualized capital + investment costs
                # invest_fom = tc_turbine + tc_station + tc_sub + tc_under
                # Total fixed costs
                invest_fom = fc_turbine + fc_station + fc_sub + fc_under
                
                # print(f"Calculated Offshore costs for {carrier} rc {rc}: Dist {dist:.1f}km, Total Life Cost {invest_fom:.0f}")

        except Exception as e:
            print(f"Error calculating connection costs for {carrier}: {e}. using base invest.")

    return invest_fom


# Transform CAGR (continuous compunded annual growth rates) to discrete annual growth rates (IRR)
def irr_to_cagr(irr, T):
    if irr == 0:
        return 0
    # Annuity factor (A/P) = irr / (1 - (1+irr)^-T)
    ann_of_invest = irr / (1 - (1 + irr) ** (-T))
    # Total Revenue / Project Cost = T * ann_of_invest
    # CAGR = ln(Total Revenue / Project Cost) / T
    cagr = log(T * ann_of_invest) / T
    return cagr


# Transform discrete annual growth rates (IRR) to CAGR (continuous compunded annual growth rates)
def cagr_to_irr(cagr, T):
    # Reverse the formula: Total Revenue / Project Cost = exp(CAGR * T)
    # Target (A/P) = (Total Revenue / Project Cost) / T
    target_ap = exp(cagr * T) / T
    
    # Solve for irr where: (r / (1 - (1+r)^-T)) - target_ap = 0
    func = lambda r: (r / (1 - (1+r)**-T)) - target_ap
    try:
        # Initial guess 7%
        irr = fsolve(func, 0.07)[0]
    except:
        irr = np.nan
    return irr


# Expected return on invest based on MCS revenues
def expected_roi_mcs(run_name, n, carrier, rc, N=10000, costs=None, scenario='mb'):
    if costs is None:
        cost_file = f"../resources/{run_name}/costs_2030.csv"
        costs = pd.read_csv(cost_file, index_col=[0, 1]).sort_index()
        costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
        costs.loc[costs.unit.str.contains("/GW"), "value"] /= 1e3
        costs = costs.value.unstack(level=1).groupby("technology").sum(min_count=1)
        # Fill missing values
        fill_values = { "FOM": 0, "investment": 0, "lifetime": 25 }
        costs = costs.fillna(fill_values)
    
    carrier_key = carrier
    if carrier in ['offwind-ac', 'offwind-dc']:
        carrier_key = 'offwind'
    elif carrier == 'solar-hsat':
        carrier_key = 'solar-utility single-axis tracking'
    elif carrier == 'solar':
        carrier_key = 'solar-utility'
    fom = costs.at[carrier_key, "FOM"] if carrier_key in costs.index else 0.0
    invest = costs.at[carrier_key, "investment"] if carrier_key in costs.index else 0.0
    lifetime = costs.at[carrier_key, "lifetime"] if carrier_key in costs.index else 25.0

    # invest_fom = fixed_costs(run_name, n, carrier, rc, costs)
    rev_T = avg_lifetime_revenues_mcs_per_MW(run_name, n, carrier, rc, int(lifetime), N, scenario)
    
    if rev_T is None:
        return np.nan

    # Discounting FOM from revenues as they are not part of initial investments
    rev_T_actual = rev_T - (fom / 100 * invest * lifetime)
    
    # Calculate ROI as compund annual growth rate
    roi = log(rev_T_actual / invest) / lifetime if invest != 0 and rev_T_actual > 0 else np.nan
    roi = cagr_to_irr(roi, lifetime) if not np.isnan(roi) else np.nan
    return roi


# Simplified closed formula calculation of WACC (CAGR) based on MM-Theorem
def wacc_cagr_simplified(f_cagr, T, sigma):
    wacc = f_cagr - log(1 - erf(sigma * 2**(-3/2))) / T
    return wacc


# Simplified closed formula calculation of WACC (IRR) based on MM-Theorem
def wacc_irr_simplified(f_irr, T, sigma):
    f_cagr = irr_to_cagr(f_irr, T)
    wacc_cagr = wacc_cagr_simplified(f_cagr, T, sigma)
    wacc_irr = cagr_to_irr(wacc_cagr, T)
    return wacc_irr


# Updated WACC for each scenario
def wacc_updated(run_name, n, carrier, rc, T=0, N=10000, scenario='mb'):
    if T == 0:
        if carrier == 'solar' or carrier == 'solar-hsat':
            T = 40
        elif carrier == 'offwind' or carrier == 'offwind-ac' or carrier == 'offwind-dc' or carrier == 'onwind':
            T = 30
        elif carrier == 'offwind-float':
            T = 20
        else:
            T = 25
    if carrier == 'solar' or carrier == 'solar-hsat':
        f_irr = 0.0486
    elif carrier == 'offwind' or carrier == 'offwind-ac' or carrier == 'offwind-dc' or carrier == 'offwind-float':
        f_irr = 0.0575
    elif carrier == 'onwind':
        f_irr = 0.0529
    else:
        print(f"Warning: Carrier {carrier} not recognized.")
        f_irr = np.nan
    sigma = risk_lifetime_revenues_mcs_per_MW(run_name, n, carrier, rc, T, N, scenario)
    wacc = wacc_irr_simplified(f_irr, T, sigma)
    return wacc


# # Needs fixing as numerical solver sometimes fails -> not relevant anymore
# # Numerical solver for debt return (CAGR) based on Mathematica script "Cost of Capital numerical clean.nb"
# def debt_cagr_numerical(s, I, f, T, sigma):
#     """
#     Numerically solves for the CAGR-formulation return on debt d.
    
#     Parameters:
#     s (float): Debt share (0 to 1)
#     I (float): Expected return on investment (CAGR)
#     f (float): Risk-free rate (CAGR)
#     T (float): Lifetime in years
#     sigma (float): Normalized standard deviation of returns
    
#     Returns:
#     float: The return on debt d
#     """
    
#     # Guard against zero volatility (risk-free case)
#     if sigma <= 1e-9:
#         return f

#     # Helper functions equivalent to Mathematica script
#     def F(omega, sigma):
#         # F[omega, sigma] := 1/2 * (1 + Erf[(Log[omega] + sigma^2/2)/(sigma * Sqrt[2])])
#         # This is equivalent to standard normal CDF of (ln(omega) + sigma^2/2)/sigma
#         # Ensure positive omega for log
#         omega = np.maximum(omega, 1e-10)
#         arg = (np.log(omega) + (sigma**2)/2) / sigma
#         return norm.cdf(arg)

#     def Fc(omega, sigma):
#         # Fc[omega, sigma] := 1/2 * (1 + Erf[(Log[omega] - sigma^2/2)/(sigma * Sqrt[2])])
#         # This is equivalent to standard normal CDF of (ln(omega) - sigma^2/2)/sigma
#         omega = np.maximum(omega, 1e-10)
#         arg = (np.log(omega) - (sigma**2)/2) / sigma
#         return norm.cdf(arg)

#     def rhs1(s, I, d, T, sigma):
#         # s * Exp[d * T] * F[s * Exp[-(I - d) * T], sigma]
#         omega = s * np.exp(-(I - d) * T)
#         return s * np.exp(d * T) * F(omega, sigma)

#     def rhs2(s, I, d, T, sigma):
#         # Exp[I * T] * Fc[s * Exp[-(I - d) * T], sigma]
#         omega = s * np.exp(-(I - d) * T)
#         return np.exp(I * T) * Fc(omega, sigma)

#     def lhs(s, f, T):
#         # s * Exp[f * T]
#         return s * np.exp(f * T)

#     # The equation to solve: -lhs + rhs1 + rhs2 = 0
#     def equation(d):
#         return -lhs(s, f, T) + rhs1(s, I, d, T, sigma) + rhs2(s, I, d, T, sigma)

#     # Initial guess: assume d is close to f
#     d_guess = f 
    
#     # Solve
#     d_solution = fsolve(equation, d_guess)
    
#     return d_solution[0]


# Still needs fixing from here onwards
# Annual revenue per MW for a given carrier and year under a standard financial CfD 
# (using the average market profile as the reference profile - equivalent to a conventional CfD in this case)
# def annual_revenue_per_MW_with_CfD(carrier, year, n, use_avg_rev_per_MWh=True, strike_price=0, bus="DE0 0"):
    if carrier == 'offwind':
        # For 'offwind-ac'
        hg_ac_per_mw = get_hourly_generation_per_MW('offwind-ac', n)
        hg_ac_year_per_mw = hg_ac_per_mw[hg_ac_per_mw.index.year == year]
        if use_avg_rev_per_MWh:
            stats_ac = annual_revenue_statistics('offwind-ac', n, bus)
            hg_per_mw_ac = get_hourly_generation_per_MW('offwind-ac', n)
            avg_annual_gen_per_MW_ac = hg_per_mw_ac.sum().values[0] / 9  # Total generation per MW divided by number of years
            strike_price_ac = stats_ac["mean"] / avg_annual_gen_per_MW_ac  # Average revenue per MWh
            # print(f"Using average revenue per MWh as AC strike price: {strike_price_ac:.2f} €/MWh")
        rev_year_per_MW_with_CfD_ac = (strike_price_ac * hg_ac_year_per_mw.values.flatten()).sum()
        # For 'offwind-dc'
        hg_dc_per_mw = get_hourly_generation_per_MW('offwind-dc', n)
        hg_dc_year_per_mw = hg_dc_per_mw[hg_dc_per_mw.index.year == year]
        if use_avg_rev_per_MWh:
            stats_dc = annual_revenue_statistics('offwind-dc', n, bus)
            hg_per_mw_dc = get_hourly_generation_per_MW('offwind-dc', n)
            avg_annual_gen_per_MW_dc = hg_per_mw_dc.sum().values[0] / 9  # Total generation per MW divided by number of years
            strike_price_dc = stats_dc["mean"] / avg_annual_gen_per_MW_dc  # Average revenue per MWh
            # print(f"Using average revenue per MWh as DC strike price: {strike_price_dc:.2f} €/MWh")
        rev_year_per_MW_with_CfD_dc = (strike_price_dc * hg_dc_year_per_mw.values.flatten()).sum()
        # For 'offwind-float'
        hg_float_per_mw = get_hourly_generation_per_MW('offwind-float', n)
        hg_float_year_per_mw = hg_float_per_mw[hg_float_per_mw.index.year == year]
        if use_avg_rev_per_MWh:
            stats_float = annual_revenue_statistics('offwind-float', n, bus)
            hg_per_mw_float = get_hourly_generation_per_MW('offwind-float', n)
            avg_annual_gen_per_MW_float = hg_per_mw_float.sum().values[0] / 9  # Total generation per MW divided by number of years
            strike_price_float = stats_float["mean"] / avg_annual_gen_per_MW_float  # Average revenue per MWh
            # print(f"Using average revenue per MWh as Float strike price: {strike_price_float:.2f} €/MWh")
        rev_year_per_MW_with_CfD_float = (strike_price_float * hg_float_year_per_mw.values.flatten()).sum()
        # Total revenue per MW with CfD
        p_nom_opt_ac = n.generators.p_nom_opt[n.generators.carrier == 'offwind-ac'].sum()
        p_nom_opt_dc = n.generators.p_nom_opt[n.generators.carrier == 'offwind-dc'].sum()
        p_nom_opt_float = n.generators.p_nom_opt[n.generators.carrier == 'offwind-float'].sum()
        rev_year_per_MW_with_CfD = (rev_year_per_MW_with_CfD_ac * p_nom_opt_ac + rev_year_per_MW_with_CfD_dc * p_nom_opt_dc + rev_year_per_MW_with_CfD_float * p_nom_opt_float) / (p_nom_opt_ac + p_nom_opt_dc + p_nom_opt_float)
    else:
        hourly_generation_per_MW = get_hourly_generation_per_MW(carrier, n)
        hourly_generation_year_per_MW = hourly_generation_per_MW[hourly_generation_per_MW.index.year == year]
        if use_avg_rev_per_MWh:
            stats = annual_revenue_statistics(carrier, n, bus)
            hg_per_mw = get_hourly_generation_per_MW(carrier, n)
            avg_annual_gen_per_MW = hg_per_mw.sum().values[0] / 9  # Total generation per MW divided by number of years
            strike_price = stats["mean"] / avg_annual_gen_per_MW  # Average revenue per MWh
            # print(f"Using average revenue per MWh as strike price: {strike_price:.2f} €/MWh")
        # Revenue calculation with CfD
        rev_year_per_MW_with_CfD = (strike_price * hourly_generation_year_per_MW.values.flatten()).sum()
    return rev_year_per_MW_with_CfD


# Annual Revenue Statistics with CfD for a given carrier and rc across all years
def annual_revenue_statistics_mcs(run_name, carrier, n, rc, T=0, N=10000, scenario='pb', bus="DE0 0"):
    if T == 0:
        if carrier == 'solar' or carrier == 'solar-hsat':
            T = 40
        elif carrier == 'offwind' or carrier == 'offwind-ac' or carrier == 'offwind-dc' or carrier == 'onwind':
            T = 30
        elif carrier == 'offwind-float':
            T = 20
        else:
            T = 25
    if scenario in ['mb', 'pb', 'pi', 'cb']:
        revenues = lifetime_revenues_mcs_per_MW(run_name, n, carrier, rc, T=T, N=N, scenario=scenario)
        wacc = wacc_updated(run_name, n, carrier, rc, T=T, N=N, scenario=scenario)
    else:
        print(f"Error: Scenario {scenario} not recognized.")
        revenues = np.array([])
    stats = {
        "mean": np.mean(revenues),
        "median": np.median(revenues),
        "std_dev": np.std(revenues),
        "risk": np.std(revenues) / np.mean(revenues) if np.mean(revenues) != 0 else np.nan,
        "min": np.min(revenues),
        "max": np.max(revenues),
        "wacc": wacc
    }
    return stats


# Table of revenue statistics with and without CfD for a given carrier and rc
def revenue_statistics_comparison(run_name, carrier, n, rc, T=0, N=10000, bus="DE0 0"):
    stats_mb = annual_revenue_statistics_mcs(run_name, carrier, n, rc, T=T, N=N, scenario='mb', bus=bus)
    stats_pb = annual_revenue_statistics_mcs(run_name, carrier, n, rc, T=T, N=N, scenario='pb', bus=bus)
    stats_pi = annual_revenue_statistics_mcs(run_name, carrier, n, rc, T=T, N=N, scenario='pi', bus=bus)
    stats_cb = annual_revenue_statistics_mcs(run_name, carrier, n, rc, T=T, N=N, scenario='cb', bus=bus)
    carrier_name = carrier_full_name(carrier)
    print(f"Revenue Statistics for {carrier_name}, Resource Class {rc}:")
    print(f"{'Statistic':<10} {'Market (€)':<18} {'PB CfD (€)':<18} {'PI CfD (€)':<18} {'CB CfD (€)':<18}")
    for key in stats_mb.keys():
        print(f"{key:<10} {stats_mb[key]:<18.4f} {stats_pb[key]:<18.4f} {stats_pi[key]:<18.4f} {stats_cb[key]:<18.4f}")
    return {'mb': stats_mb, 'pb': stats_pb, 'pi': stats_pi, 'cb': stats_cb}



# ___ Plots of revenue statistics ___

# Plot WACC (IRR) vs. revenue volatility
def plot_wacc_vs_volatility(sigma_max=0.5, save_to_file=True):
    sigma_values = np.linspace(0, sigma_max, 100)
    waccs_solar = [wacc_irr_simplified(f_irr=0.0486, T=40, sigma=sigma) for sigma in sigma_values]
    plt.plot(sigma_values, waccs_solar, label='Solar', color=color_theme("solar"))
    waccs_offwind = [wacc_irr_simplified(f_irr=0.0575, T=30, sigma=sigma) for sigma in sigma_values]
    plt.plot(sigma_values, waccs_offwind, label='Offshore Wind', color=color_theme("offwind"))
    waccs_onwind = [wacc_irr_simplified(f_irr=0.0529, T=30, sigma=sigma) for sigma in sigma_values]
    plt.plot(sigma_values, waccs_onwind, label='Onshore Wind', color=color_theme("onwind"))
    plt.xlabel('Revenue volatility ($\sigma$)')
    plt.ylabel('WACC')
    plt.xlim(0, sigma_max)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    plt.title('WACC vs. Revenue Volatility')
    plt.legend()
    plt.grid(True)
    if save_to_file:
        results_directory = f'../results'
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/wacc_vs_volatility.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


# Plot annual revenue of a carrier for all years
def plot_annual_revenue_per_MW(carrier, run_name, scenario, bus="DE0 0", save_to_file=True):
    n = get_network(run_name)
    years = n.generators_t.p.index.year.unique()
    revenues = [annual_revenue_per_MW(carrier, year, n, bus) for year in years]
    plt.figure(figsize=(10, 6))
    color = colormap_for_carrier(carrier)
    # Convert 'Blues', 'Greens' etc. to a single color for points, or just use a default
    # For points we probably want a specific color intensity, not a colormap (unless by value)
    # Mapping old custom colors to simple standard ones if colormap_for_carrier returns strings
    c_map_val = 'blue'
    if carrier == 'offwind': c_map_val = 'blue'
    elif carrier == 'onwind': c_map_val = 'green'
    elif carrier == 'solar': c_map_val = 'orange'
    elif carrier == 'solar rooftop': c_map_val = 'red'
    
    plt.plot(years, revenues, 'o', color=c_map_val) # Points
    plt.axhline(y=np.mean(revenues), color='black', linestyle='--')
    carrier_name = carrier_full_name(carrier)
    plt.title(f'Annual Revenue of {carrier_name} per MW installed capacity')
    plt.xlabel('Weather Year')
    plt.ylabel('Annual Revenue [€] per MW installed capacity')
    plt.ylim(0, 1.02*np.max(revenues))  # Set y-axis limit slightly above max revenue for better visualization
    plt.grid(True)
    if save_to_file:
        results_directory = f'results/{run_name}'
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/annual_revenue_per_MW_{carrier}_{scenario}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


# Plot histogram of normalized lifetime revenues across all MCS runs
def lifetime_revenue_mcs_dist_plot(run_name, scenario, carrier, N=10000, save_to_file = True):
    
    n = get_network(run_name)
    revenue_mcs = []
    # Determine number of resource classes (matching logic in analysis_functions)
    n_rc_actual = n.generators.index[n.generators.carrier == carrier].size
    n_rc = n_rc_actual if n_rc_actual > 0 else 1

    if carrier == 'solar' or carrier == 'solar-hsat':
        t = 40
    elif carrier == 'offwind' or carrier == 'offwind-ac' or carrier == 'offwind-dc' or carrier == 'onwind':
        t = 30
    elif carrier == 'offwind-float':
        t = 20
    
    for rc in range(n_rc):
        # Skip resource classes with zero installed capacity
        gen_label = f"DE0 0 {rc} {carrier}"
        if gen_label in n.generators.index and n.generators.p_nom_opt[gen_label] == 0:
            continue
        rev_mcs = lifetime_revenues_mcs_per_MW(run_name, n, carrier, rc, T=t, N=N, scenario=scenario)
        if rev_mcs is not None:
            mean_val = np.nanmean(rev_mcs)
            if mean_val == 0 or not np.isfinite(mean_val):
                continue
            rev_mcs_norm = rev_mcs / mean_val
            revenue_mcs.append(rev_mcs_norm)
            
    if not revenue_mcs:
        print(f"No valid results found for {carrier} (all resource classes have zero capacity or non-finite revenues)")
        return
        
    # Create DataFrame (rows=resource_classes, cols=simulations)
    rev_mcs_df = pd.DataFrame(revenue_mcs)
    flat_data = rev_mcs_df.values.flatten()
    flat_data = flat_data[np.isfinite(flat_data)]  # Remove any remaining NaN/Inf
    
    if len(flat_data) == 0:
        print(f"No finite data remaining for {carrier} after filtering")
        return
    
    gen_name = carrier_full_name(carrier)
    
    plt.figure(figsize=(10,6))
    
    # Plot histogram with density=True to assist with fitting
    plt.hist(flat_data, bins=50, color=color_theme(carrier), edgecolor='black', density=True, alpha=0.6, label='Histogram ($\sigma$={:.4f})'.format(np.std(flat_data)))
    
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    
    fit_stats = {}  # dict of {name: ks_stat} for best-fit comparison
    
    if scenario == 'mb':
        # --- Log-Normal Distribution (market-based scenario) ---
        shape, loc, scale = lognorm.fit(flat_data)
        p_lognorm = lognorm.pdf(x, shape, loc, scale)
        ks_stat_lognorm, p_val_lognorm = kstest(flat_data, 'lognorm', args=(shape, loc, scale))
        plt.plot(x, p_lognorm, 'r--', linewidth=2, label=f'Log-Normal (KS = {ks_stat_lognorm:.4f})')
        fit_stats['Log-Normal'] = ks_stat_lognorm
    else:
        # --- Student's t Distribution (CfD scenarios) ---
        df_t, loc_t, scale_t = student_t.fit(flat_data)
        p_t = student_t.pdf(x, df_t, loc_t, scale_t)
        ks_stat_t, p_val_t = kstest(flat_data, 't', args=(df_t, loc_t, scale_t))
        plt.plot(x, p_t, 'm:', linewidth=2, label=f"Student's t (KS = {ks_stat_t:.4f}, df = {df_t:.1f})")
        fit_stats["Student's t"] = ks_stat_t

        # --- Johnson SU Distribution (CfD scenarios) ---
        a_jsu, b_jsu, loc_jsu, scale_jsu = johnsonsu.fit(flat_data)
        p_jsu = johnsonsu.pdf(x, a_jsu, b_jsu, loc_jsu, scale_jsu)
        ks_stat_jsu, p_val_jsu = kstest(flat_data, 'johnsonsu', args=(a_jsu, b_jsu, loc_jsu, scale_jsu))
        plt.plot(x, p_jsu, 'c-', linewidth=2, label=f'Johnson SU (KS = {ks_stat_jsu:.4f})')
        fit_stats['Johnson SU'] = ks_stat_jsu

    plt.title(f'Lifetime Normalized Revenue Distribution for {gen_name} over {t} years')
    plt.xlabel('Lifetime Normalized Revenue')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.legend()
    if save_to_file:
        results_directory = f'../results/{run_name}'
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/norm_revenue_dist_MCS_{carrier}_{scenario}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()
    
    # Print Test Results
    print(f"--- Goodness of Fit Results for {gen_name} ---")
    if scenario == 'mb':
        print(f"Log-Normal:  KS Stat = {ks_stat_lognorm:.4f}, p-value = {p_val_lognorm:.4e}")
    else:
        print(f"Student's t: KS Stat = {ks_stat_t:.4f}, p-value = {p_val_t:.4e} (df = {df_t:.1f})")
        print(f"Johnson SU:  KS Stat = {ks_stat_jsu:.4f}, p-value = {p_val_jsu:.4e}")
    
    best_fit = min(fit_stats, key=fit_stats.get)
    print(f"-> The {best_fit} distribution fits the data best (lowest KS statistic).\n")
    
    return rev_mcs_df


# Plot ROI for all resource classes of a carrier
def plot_roi_per_rc(run_name, carrier, N=10000, save_to_file=True):
    n = get_network(run_name)
    n_rc_actual = n.generators.index[n.generators.carrier == carrier].size
    n_rc = n_rc_actual if n_rc_actual > 0 else 1
    rois = []
    for rc in range(n_rc):
        roi = expected_roi_mcs(run_name, n, carrier, rc, N)
        rois.append(roi * 100)  # Convert to percentage
    plt.figure(figsize=(10, 6))
    color = color_theme(carrier)
    plt.plot(range(n_rc), rois, 'o', color=color, markeredgecolor='black')
    # plt.axhline(y=np.mean(rois), color='black', linestyle='--')
    carrier_name = carrier_full_name(carrier)
    plt.title(rf'Expected ROI of {carrier_name} per Resource Class')
    plt.xlabel('Resource Class')
    plt.xticks(range(n_rc))
    plt.ylabel('Expected ROI [%]')
    valid_rois = [r for r in rois if np.isfinite(r)]
    if valid_rois:
        plt.ylim(0.0, 1.2 * np.max(valid_rois))  # Set y-axis limit slightly above max ROI for better visualization
    plt.grid(True)
    if save_to_file:
        results_directory = f'../results/{run_name}'
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/expected_roi_per_resource_class_{carrier}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


# Plot lifetime risk for all resource classes of a carrier
def plot_lifetime_risk_per_rc(run_name, carrier, N=10000, save_to_file=True, scenario='mb'):
    n = get_network(run_name)
    n_rc_actual = n.generators.index[n.generators.carrier == carrier].size
    n_rc = n_rc_actual if n_rc_actual > 0 else 1
    risks = []
    for rc in range(n_rc):
        rev_mcs = lifetime_revenues_mcs_per_MW(run_name, n, carrier, rc, N=N, scenario=scenario)
        if rev_mcs is not None:
            risk = np.std(rev_mcs) / np.mean(rev_mcs) if np.mean(rev_mcs) != 0 else np.nan
            risks.append(risk * 100)  # Convert to percentage
        else:
            risks.append(np.nan)
    plt.figure(figsize=(10, 6))
    color = color_theme(carrier)
    plt.plot(range(n_rc), risks, 'o', color=color, markeredgecolor='black')
    # plt.axhline(y=np.mean(risks), color='black', linestyle='--')
    carrier_name = carrier_full_name(carrier)
    plt.title(rf'Lifetime Revenue Risk $\frac{{\sigma}}{{\mu}}$ of {carrier_name} per Resource Class')
    plt.xlabel('Resource Class')
    plt.xticks(range(n_rc))
    plt.ylabel('Lifetime Revenue Risk [%]')
    valid_risks = [r for r in risks if np.isfinite(r)]
    if valid_risks:
        plt.ylim(0.0, 1.2 * np.max(valid_risks))  # Set y-axis limit slightly above max risk for better visualization
    plt.grid(True)
    if save_to_file:
        results_directory = f'../results/{run_name}'
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/lifetime_risk_per_resource_class_{carrier}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


# Plot annual revenue of a carrier for all years under a standard financial CfD
def plot_annual_revenue_per_MW_with_CfD(carrier, run_name, scenario, use_avg_rev_per_MWh=True, strike_price=0, bus="DE0 0", save_to_file=True): 
    n = get_network(run_name)
    years = n.generators_t.p.index.year.unique()
    revenues = [annual_revenue_per_MW_with_CfD(carrier, year, n, use_avg_rev_per_MWh, strike_price, bus) for year in years]
    rev_0 = [annual_revenue_per_MW(carrier, year, n, bus) for year in years]
    plt.figure(figsize=(10, 6))
    
    c_map_val = 'blue'
    if carrier == 'offwind': c_map_val = 'blue'
    elif carrier == 'onwind': c_map_val = 'green'
    elif carrier == 'solar': c_map_val = 'orange'
    elif carrier == 'solar rooftop': c_map_val = 'red'

    plt.plot(years, revenues, 'o', color=c_map_val) # Points
    plt.axhline(y=np.mean(revenues), color='black', linestyle='--')
    carrier_name = carrier_full_name(carrier)
    plt.ylim(0, 1.02*np.max(rev_0))  # Set y-axis limit slightly above max revenue without CfD for better comparison
    plt.title(f'Annual Revenue of {carrier_name} per MW installed capacity with a financial CfD')
    plt.xlabel('Weather Year')
    plt.ylabel('Annual Revenue [€] per MW installed capacity with a financial CfD')
    plt.grid(True)
    if save_to_file:
        results_directory = f'../results/{run_name}'
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/annual_revenue_per_MW_with_CfD_{carrier}_{scenario}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


# Plot histogram of normalized distribution of annual revenues of all carriers around their average
def plot_normalized_annual_revenue_distribution(run_name, carriers=['offwind', 'onwind', 'solar', 'solar rooftop'], bus="DE0 0", save_to_file=True):
    n = get_network(run_name)
    years = n.generators_t.p.index.year.unique()
    plt.figure(figsize=(10, 6))
    normalized_revenues = []
    for carrier in carriers:
        revenues = [annual_revenue_per_MW(carrier, year, n, bus) for year in years]
        avg_revenue = np.mean(revenues)
        normalized_revenues_carrier = [rev / avg_revenue for rev in revenues]
        normalized_revenues.append(normalized_revenues_carrier)
    normalized_revenues = np.array(normalized_revenues).flatten()
    plt.hist(normalized_revenues, bins=10, alpha=0.5, label='Normalized annual revenue for all VRE', color='purple', width=0.01)
    # Add fitted log-linear probability density
    # count, bins, ignored = plt.hist(normalized_revenues, bins=10, density=True, alpha=0)
    # bin_centers = 0.5 * (bins[1:] + bins[:-1])
    # log_counts = np.log(count + 1e-10)  # Avoid log(0)
    # fitted = np.exp(np.polyval(coeffs, bin_centers))
    # plt.plot(bin_centers, fitted, 'r--', label='Fitted log-linear density')
    # Fit a log-normal distribution to the normalized revenues
    shape, loc, scale = lognorm.fit(normalized_revenues, floc=0)
    x = np.linspace(min(normalized_revenues), max(normalized_revenues), 100)
    pdf_fitted = lognorm.pdf(x, shape, loc=loc, scale=scale)
    plt.plot(x, pdf_fitted, 'r--', label='Fitted log-normal density')
    # Fit a Rayleigh distribution to the normalized revenues
    # param = rayleigh.fit(normalized_revenues, floc=0)
    # pdf_fitted_rayleigh = rayleigh.pdf(x, *param)
    # plt.plot(x, pdf_fitted_rayleigh, 'g--', label='Fitted Rayleigh density')
    plt.title('Normalized Distribution of Annual Revenues per MW installed capacity')
    plt.xlabel('Normalized Annual Revenue (Revenue / Average Revenue)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    if save_to_file:    
        results_directory = f'../results/{run_name}'
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/normalized_annual_revenue_distribution.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


# Violin plot for annual revenues for different CfD instruments
def plot_revenue_violin(carrier, run_name, bus="DE0 0", save_to_file=True):
    n = get_network(run_name)
    years = n.generators_t.p.index.year.unique()
    all_revenues = []
    carrier_names_map = {
        'offwind': "Offshore Wind",
        'onwind': "Onshore Wind",
        'solar': "Ground-Mounted Solar PV",
        'solar rooftop': "Rooftop Solar PV"}
    carrier_name = carrier_names_map.get(carrier, carrier)
    rev_no_cfd = [annual_revenue_per_MW(carrier, year, n, bus) for year in years]
    rev_cfd = [annual_revenue_per_MW_with_CfD(carrier, year, n, bus) for year in years]
    # rev_cap = annual_revenue_per_MW_with_Cap(carrier, year, bus) for year in years]
    # Define data for each scenario
    scenarios = {
        "No CfD": rev_no_cfd,
        "Financial CfD": rev_cfd 
        # "Capacity Component": rev_cap # Uncomment if you have this network
    }
    for name, revenues in scenarios.items():
        for revenue in revenues:
            all_revenues.append({'run_name': name, 'revenue': revenue})
    df = pd.DataFrame(all_revenues)
    plt.figure(figsize=(10, 6))
    carrier_colors = {
        "Offshore Wind": mcblue,
        "Onshore Wind": mcgreen,
        "Ground-Mounted Solar PV": mcorange,
        "Rooftop Solar PV": mcred}
    color = carrier_colors.get(carrier_name, (0.5, 0.5, 0.5))  # Default to grey if not found
    palette_colors = {name: color for name in scenarios.keys()}
    sns.violinplot(x='run_name', y='revenue', data=df, palette=palette_colors)
    plt.title(f'Distribution of Annual Revenues per MW installed capacity of {carrier_name}', fontsize=12)
    plt.xlabel(None)
    plt.ylabel('Annual Revenue [€] per MW installed capacity', fontsize=12)
    plt.grid(True)
    plt.ylim(0, None)  # Start y-axis at 0
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if save_to_file:
        results_directory = f'../results/{run_name}'
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        # The filename should probably be more specific to the carrier
        filename = f'{output_dir}/revenue_violin_plot_{carrier}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()



# # ___ Cost of capital calculations ___

# # Updated discount rate (i.e. WACC) assuming (simplified!) linear relation between WACC and revenue variability
# def updated_discount_rate(carrier, n, bus="DE0 0"):
#     base_wacc = 0.07  # Base WACC
#     base_wacc_solar_rooftop = 0.04  # Base WACC for solar (avg. of grounded (0.07) and rooftop (0.04))
#     risk_free_rate = 0.02  # Risk-free rate
#     stats_without_CfD = annual_revenue_statistics(carrier, n, bus)
#     revenue_variability_without_CfD = stats_without_CfD["std_dev"]
#     stats_with_CfD = annual_revenue_statistics_with_CfD(carrier, n, True, 0, bus)
#     revenue_variability_with_CfD = stats_with_CfD["std_dev"]
#     if carrier == 'solar rooftop':
#         updated_wacc = risk_free_rate + (base_wacc_solar_rooftop - risk_free_rate) * (revenue_variability_with_CfD / revenue_variability_without_CfD)
#     else:
#         updated_wacc = risk_free_rate + (base_wacc - risk_free_rate) * (revenue_variability_with_CfD / revenue_variability_without_CfD)
#     return updated_wacc


# # Updated annuity factor based on updated discount rate
# def updated_capital_cost(carrier, n, bus="DE0 0"):
#     lifetime = 0.0
#     if carrier == 'offwind':
#         lifetime = 30.0 # Taken from the costs_2030.csv file for offshore wind
#     else:
#         lifetime = n.generators.lifetime[n.generators.carrier == carrier]
#     if carrier == 'solar rooftop':
#         base_wacc = 0.04  # Base WACC for solar (avg. of grounded (0.07) and rooftop (0.04))
#     else:
#         base_wacc = 0.07  # Base WACC
#     annuity_factor_old = (1 - (1 + base_wacc) ** -lifetime) / base_wacc
#     wacc = updated_discount_rate(carrier, n, bus)
#     annuity_factor_new = (1 - (1 + wacc) ** -lifetime) / wacc
#     capital_cost = 0.0
#     if carrier == 'offwind':
#         capital_cost = 1682122.6 * annuity_factor_old / annuity_factor_new # Taken from the costs_2030.csv file for offshore wind
#     else:
#         capital_cost = n.generators.capital_cost[n.generators.carrier == carrier] * annuity_factor_old / annuity_factor_new
#     return capital_cost



# ___ Statistics on public CfD payments ___

# Annual state CfD payments for a given carrier and year
def annual_CfD_payments_per_MW(carrier, year, n, use_avg_rev_per_MWh=True, strike_price=0, bus="DE0 0"):
    rev_without_CfD = annual_revenue_per_MW(carrier, year, n, bus)
    rev_with_CfD = annual_revenue_per_MW_with_CfD(carrier, year, n, use_avg_rev_per_MWh, strike_price, bus)
    payments = rev_with_CfD - rev_without_CfD
    return payments



# ___ Plots of public CfD payments statistics ___

# Plot annual state CfD payments for a given carrier across all years
def plot_annual_CfD_payments(carrier, run_name, use_avg_rev_per_MWh=True, strike_price=0, bus="DE0 0", save_to_file=True):
    n = get_network(run_name)
    years = n.generators_t.p.index.year.unique()
    payments = [annual_CfD_payments_per_MW(carrier, year, n, use_avg_rev_per_MWh, strike_price, bus) for year in years]
    plt.figure(figsize=(10, 6))
    color = ''
    if carrier == 'offwind':
        color = mcblue
    elif carrier == 'onwind':
        color = mcgreen
    elif carrier == 'solar':
        color = mcorange
    elif carrier == 'solar rooftop':
        color = mcred
    plt.bar(years, payments, color=color, alpha=0.7)
    carrier_name = ""
    if carrier == 'offwind':
        carrier_name = "Offshore Wind"
    elif carrier == 'onwind':
        carrier_name = "Onshore Wind"
    elif carrier == 'solar':
        carrier_name = "Ground-Mounted Solar PV"
    elif carrier == 'solar rooftop':
        carrier_name = "Rooftop Solar PV"
    plt.title(f'Annual State CfD Payments for {carrier_name} per MW installed capacity')
    plt.xlabel('Weather Year')
    plt.ylabel('Annual State CfD Payments [€] per MW installed capacity')
    plt.grid(True)
    if save_to_file:
        results_directory = f'../results/{run_name}'
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/annual_CfD_payments_{carrier}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()









