# Necessary imports
import pypsa
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import lognorm
import seaborn as sns
import pandas as pd
import geopandas as gpd

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


# ___ Define network ___
def get_network(run_name):
    results_directory = "results/{}".format(run_name)
    net_dir = os.path.join(results_directory, "networks")
    if not os.path.isdir(net_dir):
        raise FileNotFoundError(f"Networks directory not found: {net_dir}")
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
        'ror': 'Purples',
        'geothermal': 'Reds',
        'biomass': 'YlGn',
        'CCGT': 'YlOrBr',
        'OCGT': 'YlOrBr',
        'coal': 'Greys',
        'lignite': 'Greys',
        'oil': 'PuRd',
        'nuclear': 'BuPu'
    }
    return colormaps.get(carrier, 'viridis')


# ___ Get electricity generation time series ___

# Function to get hourly generation of any carrier
def get_hourly_generation(carrier, n):
    if carrier == 'offwind':
        gen_ac = n.generators[n.generators.carrier == 'offwind-ac']
        hg_ac = n.generators_t.p[gen_ac.index]
        gen_dc = n.generators[n.generators.carrier == 'offwind-dc']
        hg_dc = n.generators_t.p[gen_dc.index]
        gen_float = n.generators[n.generators.carrier == 'offwind-float']
        hg_float = n.generators_t.p[gen_float.index]
        # Align indices and sum row-wise across the three DataFrames, return as single-column DataFrame
        summed = (hg_ac.add(hg_dc, fill_value=0)
                .add(hg_float, fill_value=0))
        # Convert to single-column DataFrame
        hourly_generation = summed.sum(axis=1).to_frame(name='generation')
    else:
        generators = n.generators[n.generators.carrier == carrier]
        hourly_generation = n.generators_t.p[generators.index]
    return hourly_generation


# Function to get hourly generation per MW of any carrier (essentially calculates average capacity factors)
def get_hourly_generation_per_MW(carrier, n):
    if carrier == 'offwind':
        gen_ac = n.generators[n.generators.carrier == 'offwind-ac']
        hg_ac = n.generators_t.p[gen_ac.index]
        p_nom_opt_ac = n.generators.p_nom_opt[gen_ac.index].sum()
        gen_dc = n.generators[n.generators.carrier == 'offwind-dc']
        hg_dc = n.generators_t.p[gen_dc.index]
        p_nom_opt_dc = n.generators.p_nom_opt[gen_dc.index].sum()
        gen_float = n.generators[n.generators.carrier == 'offwind-float']
        hg_float = n.generators_t.p[gen_float.index]
        p_nom_opt_float = n.generators.p_nom_opt[gen_float.index].sum()
        # Align indices and sum row-wise across the three DataFrames, return as single-column DataFrame
        summed = (hg_ac.add(hg_dc, fill_value=0)
                .add(hg_float, fill_value=0))
        # Convert to single-column DataFrame and divide by total optimized capacity
        hourly_generation_per_MW = (summed.sum(axis=1) / (p_nom_opt_ac + p_nom_opt_dc + p_nom_opt_float)).to_frame(name='generation_per_MW')
    else:
        generators = n.generators[n.generators.carrier == carrier]
        hourly_generation = n.generators_t.p[generators.index]
        p_nom_opt = n.generators.p_nom_opt[generators.index].sum()
        hourly_generation_per_MW = hourly_generation / p_nom_opt
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
    plt.hist(price_t, bins=bins, color=mcblue, alpha=0.7)
    plt.title(f'Histogram of Marginal Prices at {bus}')
    plt.xlabel('Marginal Price [€/MWh]')
    plt.ylabel('Frequency')
    plt.grid(True)
    if save_to_file:
        results_directory = f'results/{run_name}'
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
    plt.plot(sorted_prices, color=mcblue, linewidth=2) 
    # plt.title(f'Price Duration Curve at {bus}')
    plt.xlabel('SortedHours')
    plt.ylabel('Marginal Price [€/MWh]')
    plt.ylim(0, 1.1*sorted_prices[int(np.ceil(0.1*len(sorted_prices)))])  # Cap y-axis at 100 €/MWh
    plt.grid(True)
    if save_to_file:
        results_directory = f'results/{run_name}'
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
    vre_i = n.generators.carrier.isin(['onwind', 'offwind-ac', 'offwind-dc', 'offwind-float', 'solar', 'solar rooftop', 'ror', 'geothermal', 'biomass'])
    onwind_i = n.generators.carrier == 'onwind'
    offwind_i = n.generators.carrier.isin(['offwind-ac', 'offwind-dc', 'offwind-float'])
    solar_i = n.generators.carrier.isin(['solar', 'solar rooftop'])
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
    results_directory_nocfd = "results/{}".format(run_name_nocfd)
    n_nocfd = pypsa.Network(f"{results_directory_nocfd}/networks/base_s_1_EP__2030.nc")
    stats_nocfd = statistics(n=n_nocfd)
    stat_nocfd = stats_nocfd[stat]
    run_name_cfd = "2030_with_CfD_v4"
    results_directory_cfd = "results/{}".format(run_name_cfd)
    n_cfd = pypsa.Network(f"{results_directory_cfd}/networks/base_s_1_EP__2030.nc")
    stats_cfd = statistics(n=n_cfd)
    stat_cfd = stats_cfd[stat]
    run_name_cap = "2030_withouth_CfD_v6"
    results_directory_cap = "results/{}".format(run_name_cap)
    n_cap = pypsa.Network(f"{results_directory_cap}/networks/base_s_1_EP__2030.nc")
    stats_cap = statistics(n=n_cap)
    stat_cap = stats_cap[stat]
    plt.figure(figsize=(8, 5))
    plt.bar(['No CfD', 'With CfD', 'With CM'], [stat_nocfd, stat_cfd, stat_cap], color=[mcred, mcgreen, mcorange])
    plt.ylabel(stat)
    plt.title(f'Comparison of {stat} across Scenarios')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_to_file:
        output_dir = "results/my_plots"
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
    geothermal = n.generators.carrier == 'geothermal'
    gas = n.generators.carrier.isin(['CCGT', 'OCGT'])
    coal = n.generators.carrier.isin(['coal', 'lignite'])
    oil = n.generators.carrier == 'oil'
    nuclear = n.generators.carrier == 'nuclear'
    batteries = n.storage_units.carrier == 'battery'
    carriers = [onwind, offwind, solar, hydro, bio, geothermal, gas, coal, oil, nuclear, batteries]
    labels = [
        'Onshore',
        'Offshore',
        'Solar',
        'Hydro',
        'Bio',
        'Geothermal',
        'Gas',
        'Coal',
        'Oil',
        'Nuclear',
        'Batteries'
    ]
    capacities = []
    for carrier in carriers:
        if carrier is batteries:
            cap = n.storage_units.p_nom_opt[carrier].sum()
        else:
            cap = n.generators.p_nom_opt[carrier].sum()
        capacities.append(cap)
    plt.figure(figsize=(8, 5))
    # use labels (strings) as x values so matplotlib can align heights correctly
    plt.bar(labels, capacities, color=mcred)
    plt.title('Installed Capacities by Carrier [MW]')
    plt.ylabel('Installed Capacity [MW]')
    plt.grid(True)
    if save_to_file:
        results_directory = f'results/{run_name}'
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/installed_capacities_all_carriers.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


# Plot total generation of all carriers
def plot_total_generation(run_name, save_to_file=True):
    n = get_network(run_name)
    # Define boolean masks for carriers
    onwind = n.generators.carrier == 'onwind'
    offwind = n.generators.carrier.isin(['offwind-ac', 'offwind-dc', 'offwind-float'])
    solar = n.generators.carrier.isin(['solar', 'solar rooftop', 'solar-hsat'])
    hydro = n.generators.carrier == 'ror'
    bio = n.generators.carrier == 'biomass'
    geothermal = n.generators.carrier == 'geothermal'
    gas = n.generators.carrier.isin(['CCGT', 'OCGT'])
    coal = n.generators.carrier.isin(['coal', 'lignite'])
    oil = n.generators.carrier == 'oil'
    nuclear = n.generators.carrier == 'nuclear'
    batteries = n.storage_units.carrier == 'battery'
    carriers = [onwind, offwind, solar, hydro, bio, geothermal, gas, coal, oil, nuclear, batteries]
    generations = []
    for carrier in carriers:
        if carrier is batteries:
            gen = n.storage_units_t.p.loc[:, carrier].clip(lower=0).sum().sum()
        else:
            gen = n.generators_t.p.loc[:, carrier].sum().sum()
        generations.append(gen)
    plt.figure(figsize=(8, 5))
    labels = [
        'Onshore',
        'Offshore',
        'Solar',
        'Hydro',
        'Bio',
        'Geothermal',
        'Gas',
        'Coal',
        'Oil',
        'Nuclear',
        'Batteries'
    ]
    plt.bar(labels, generations, color=mcblue)
    plt.title('Total Generation by Carrier [MWh]')
    plt.ylabel('Total Generation [MWh]')
    plt.grid(True)
    if save_to_file:
        results_directory = f'results/{run_name}'
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/total_generation_all_carriers.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show() 


# Plot installed capacities of a carrier across different model iterations
# This has to be adapted with the final run names
def plot_installed_capacities_across_scenarios(carrier, save_to_file=True): 
    run_name_nocfd = "2030_without_CfD_v5"
    results_directory_nocfd = "results/{}".format(run_name_nocfd)
    n_nocfd = pypsa.Network(f"{results_directory_nocfd}/networks/base_s_1_EP__2030.nc")
    c_inst_nocfd = n_nocfd.generators.p_nom_opt[n_nocfd.generators.carrier == carrier].sum()
    run_name_cfd = "2030_with_CfD_v4"
    results_directory_cfd = "results/{}".format(run_name_cfd)
    n_cfd = pypsa.Network(f"{results_directory_cfd}/networks/base_s_1_EP__2030.nc")
    c_inst_cfd = n_cfd.generators.p_nom_opt[n_cfd.generators.carrier == carrier].sum()
    run_name_cap = "2030_withouth_CfD_v6"
    results_directory_cap = "results/{}".format(run_name_cap)
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
        output_dir = 'results/my_plots'
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
    dir = f'resources/{run_name}'
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
        output_dir = f'results/{run_name}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/map_{os.path.basename(file).replace(".geojson", "")}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()
    


# ___ Revenue statistics ___

# Annual revenue per MW of a carrier for a given year
def annual_revenue_per_MW(carrier, year, n, bus="DE0 0"):
    hourly_generation_per_MW = get_hourly_generation_per_MW(carrier, n)
    hourly_generation_year_per_MW = hourly_generation_per_MW[hourly_generation_per_MW.index.year == year]
    price = get_hourly_marginal_price(n, bus)
    price_year = price[price.index.year == year]
    rev_per_MW_year = (hourly_generation_year_per_MW.values.flatten() * price_year.values.flatten()).sum()
    return rev_per_MW_year


# Annual Revenue Statistics for a given carrier across all years
def annual_revenue_statistics(carrier, n, bus="DE0 0"):
    years = n.generators_t.p.index.year.unique()
    revenues = [annual_revenue_per_MW(carrier, year, n, bus) for year in years]
    stats = {
        "mean": np.mean(revenues),
        "median": np.median(revenues),
        "std_dev": np.std(revenues),
        "risk": np.std(revenues) / np.mean(revenues) if np.mean(revenues) != 0 else np.nan,
        "min": np.min(revenues),
        "max": np.max(revenues)
    }
    return stats 


# Annual revenue per MW for a given carrier and year under a standard financial CfD 
# (using the average market profile as the reference profile - equivalent to a conventional CfD in this case)
def annual_revenue_per_MW_with_CfD(carrier, year, n, use_avg_rev_per_MWh=True, strike_price=0, bus="DE0 0"):
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


# Annual Revenue Statistics with CfD for a given carrier across all years
def annual_revenue_statistics_with_CfD(carrier, n, use_avg_rev_per_MWh=True, strike_price=0, bus="DE0 0"):
    years = n.generators_t.p.index.year.unique()
    revenues = [annual_revenue_per_MW_with_CfD(carrier, year, n, use_avg_rev_per_MWh, strike_price, bus) for year in years]
    stats = {
        "mean": np.mean(revenues),
        "median": np.median(revenues),
        "std_dev": np.std(revenues),
        "risk": np.std(revenues) / np.mean(revenues) if np.mean(revenues) != 0 else np.nan,
        "min": np.min(revenues),
        "max": np.max(revenues)
    }
    return stats


# Table of revenue statistics with and without CfD for a given carrier
def revenue_statistics_comparison(carrier, n, use_avg_rev_per_MWh=True, strike_price=0, bus="DE0 0"):
    stats_without_CfD = annual_revenue_statistics(carrier, n, bus)
    stats_with_CfD = annual_revenue_statistics_with_CfD(carrier, n, use_avg_rev_per_MWh, strike_price, bus)
    print(f"Revenue Statistics for {carrier}:")
    print(f"{'Statistic':<10} {'Without CfD (€)':<20} {'With CfD (€)':<20}")
    for key in stats_without_CfD.keys():
        print(f"{key:<10} {stats_without_CfD[key]:<20.5f} {stats_with_CfD[key]:<20.5f}")
    return stats_without_CfD, stats_with_CfD



# ___ Plots of revenue statistics ___

# Plot annual revenue of a carrier for all years
def plot_annual_revenue_per_MW(carrier, run_name, scenario, bus="DE0 0", save_to_file=True):
    n = get_network(run_name)
    years = n.generators_t.p.index.year.unique()
    revenues = [annual_revenue_per_MW(carrier, year, n, bus) for year in years]
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
    plt.bar(years, revenues, color=color)
    plt.axhline(y=np.mean(revenues), color='black', linestyle='--')
    carrier_name = ""
    if carrier == 'offwind':
        carrier_name = "Offshore Wind"
    elif carrier == 'onwind':
        carrier_name = "Onshore Wind"
    elif carrier == 'solar':
        carrier_name = "Ground-Mounted Solar PV"
    elif carrier == 'solar rooftop':
        carrier_name = "Rooftop Solar PV"
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


# Plot annual revenue of a carrier for all years under a standard financial CfD
def plot_annual_revenue_per_MW_with_CfD(carrier, run_name, scenario, use_avg_rev_per_MWh=True, strike_price=0, bus="DE0 0", save_to_file=True): 
    n = get_network(run_name)
    years = n.generators_t.p.index.year.unique()
    revenues = [annual_revenue_per_MW_with_CfD(carrier, year, n, use_avg_rev_per_MWh, strike_price, bus) for year in years]
    rev_0 = [annual_revenue_per_MW(carrier, year, n, bus) for year in years]
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
    plt.bar(years, revenues, color=color)
    plt.axhline(y=np.mean(revenues), color='black', linestyle='--')
    carrier_name = ""
    if carrier == 'offwind':
        carrier_name = "Offshore Wind"
    elif carrier == 'onwind':
        carrier_name = "Onshore Wind"
    elif carrier == 'solar':
        carrier_name = "Ground-Mounted Solar PV"
    elif carrier == 'solar rooftop':
        carrier_name = "Rooftop Solar PV"
    plt.ylim(0, 1.02*np.max(rev_0))  # Set y-axis limit slightly above max revenue without CfD for better comparison
    plt.title(f'Annual Revenue of {carrier_name} per MW installed capacity with a financial CfD')
    plt.xlabel('Weather Year')
    plt.ylabel('Annual Revenue [€] per MW installed capacity with a financial CfD')
    plt.grid(True)
    if save_to_file:
        results_directory = f'results/{run_name}'
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
        results_directory = f'results/{run_name}'
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
        results_directory = f'results/{run_name}'
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        # The filename should probably be more specific to the carrier
        filename = f'{output_dir}/revenue_violin_plot_{carrier}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()



# ___ Cost of capital calculations ___

# Updated discount rate (i.e. WACC) assuming (simplified!) linear relation between WACC and revenue variability
def updated_discount_rate(carrier, n, bus="DE0 0"):
    base_wacc = 0.07  # Base WACC
    base_wacc_solar_rooftop = 0.04  # Base WACC for solar (avg. of grounded (0.07) and rooftop (0.04))
    risk_free_rate = 0.02  # Risk-free rate
    stats_without_CfD = annual_revenue_statistics(carrier, n, bus)
    revenue_variability_without_CfD = stats_without_CfD["std_dev"]
    stats_with_CfD = annual_revenue_statistics_with_CfD(carrier, n, True, 0, bus)
    revenue_variability_with_CfD = stats_with_CfD["std_dev"]
    if carrier == 'solar rooftop':
        updated_wacc = risk_free_rate + (base_wacc_solar_rooftop - risk_free_rate) * (revenue_variability_with_CfD / revenue_variability_without_CfD)
    else:
        updated_wacc = risk_free_rate + (base_wacc - risk_free_rate) * (revenue_variability_with_CfD / revenue_variability_without_CfD)
    return updated_wacc


# Updated annuity factor based on updated discount rate
def updated_capital_cost(carrier, n, bus="DE0 0"):
    lifetime = 0.0
    if carrier == 'offwind':
        lifetime = 30.0 # Taken from the costs_2030.csv file for offshore wind
    else:
        lifetime = n.generators.lifetime[n.generators.carrier == carrier]
    if carrier == 'solar rooftop':
        base_wacc = 0.04  # Base WACC for solar (avg. of grounded (0.07) and rooftop (0.04))
    else:
        base_wacc = 0.07  # Base WACC
    annuity_factor_old = (1 - (1 + base_wacc) ** -lifetime) / base_wacc
    wacc = updated_discount_rate(carrier, n, bus)
    annuity_factor_new = (1 - (1 + wacc) ** -lifetime) / wacc
    capital_cost = 0.0
    if carrier == 'offwind':
        capital_cost = 1682122.6 * annuity_factor_old / annuity_factor_new # Taken from the costs_2030.csv file for offshore wind
    else:
        capital_cost = n.generators.capital_cost[n.generators.carrier == carrier] * annuity_factor_old / annuity_factor_new
    return capital_cost



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
        results_directory = f'results/{run_name}'
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/annual_CfD_payments_{carrier}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()






