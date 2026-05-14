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


# ___ Price statistics ___

# Get hourly marginal price at each node for any carrier
def get_hourly_marginal_price(n, bus="DE0 0"):
    price_t = n.buses_t.marginal_price[bus]
    return price_t


# Average market value of a carrier 
def average_market_value_for_carrier(n, carrier, bus="DE0 0"):
    price_t = get_hourly_marginal_price(n, bus)
    generation_t = n.generators_t.p.loc[:, n.generators.carrier == carrier]
    total_generation = generation_t.sum().sum()
    if total_generation <= 0:
        return 0.0
    total_revenue = generation_t.mul(price_t, axis=0).sum().sum()
    return total_revenue / total_generation


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


# Average price of each MWh
def average_price_per_MWh(n, bus="DE0 0"):
    price_t = get_hourly_marginal_price(n, bus)
    total_hg = n.generators_t.p.sum().sum()
    total_costs = n.generators_t.p.mul(price_t, axis=0).sum().sum()
    avg_price_per_MWh = total_costs / total_hg if total_hg != 0 else np.nan
    return avg_price_per_MWh


# System costs
def system_costs(n, bus="DE0 0"):
    # As optimized by PyPSA, the objective function value is the total investment and fixed O&M costs.
    # total_costs = n.objective
    # As used here, total costs for offtakers
    gens_at_bus = n.generators.index[n.generators.bus == bus]
    if len(gens_at_bus) == 0:
        raise KeyError(f"No generators found at bus '{bus}'.")
    total_hg = n.generators_t.p.loc[:, gens_at_bus].sum(axis=1)
    price = get_hourly_marginal_price(n, bus)
    total_costs = (total_hg * price).sum()
    return total_costs



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

    def fmt(x):
        return f"{x:,.0f}"#.replace(",", ".")
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt(x)))

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

    n = get_network(run_name)
    profile_cf_by_bin = {}
    profile_candidates = [
        f"../resources/{run_name}/profile_1_{carrier}.nc",
        f"../resources/profile_1_{carrier}.nc",
    ]
    for profile_file in profile_candidates:
        if not os.path.exists(profile_file):
            continue
        try:
            with xr.open_dataset(profile_file) as ds_profile:
                if "profile" not in ds_profile:
                    continue

                cf_profile = ds_profile["profile"]
                for dim in ["time", "year"]:
                    if dim in cf_profile.dims:
                        cf_profile = cf_profile.mean(dim=dim)

                p_nom_max = ds_profile["p_nom_max"] if "p_nom_max" in ds_profile else None
                if "bin" not in cf_profile.dims:
                    continue

                for b in cf_profile["bin"].values:
                    rc = int(b)
                    cf_bin = cf_profile.sel(bin=b)
                    if "bus" in cf_bin.dims:
                        if p_nom_max is not None and "bus" in p_nom_max.dims and "bin" in p_nom_max.dims:
                            w = p_nom_max.sel(bin=b)
                            w_eff = w.where(np.isfinite(cf_bin), 0)
                            denom = float(w_eff.sum().values)
                            if denom > 0:
                                cf = float((cf_bin * w_eff).sum().values / denom)
                            else:
                                cf = float(cf_bin.mean().values)
                        else:
                            cf = float(cf_bin.mean().values)
                    else:
                        cf = float(cf_bin.values)

                    profile_cf_by_bin[rc] = cf
            break
        except Exception:
            profile_cf_by_bin = {}

    bins = sorted(gdf['bin'].dropna().unique())
    cf_labels = {}
    for b in bins:
        rc = int(b)
        try:
            if rc in profile_cf_by_bin:
                cf = profile_cf_by_bin[rc]
                cf_labels[rc] = f"Region {rc} ({cf:.1%})"
                continue

            gen_ids = n.generators.index[
                (n.generators.carrier == carrier)
                & (n.generators.index.str.contains(f" {rc} "))
            ]
            if len(gen_ids) == 0:
                gen_label = f"DE0 0 {rc} {carrier}"
                if gen_label in n.generators_t.p_max_pu.columns:
                    gen_ids = pd.Index([gen_label])

            if len(gen_ids) == 0:
                raise KeyError(f"No generator found for carrier {carrier}, rc {rc}")

            cf_t = n.generators_t.p_max_pu.loc[:, gen_ids]
            if hasattr(n.snapshot_weightings, "generators"):
                w = n.snapshot_weightings.generators
                cf = float(cf_t.mul(w, axis=0).sum().sum() / (w.sum() * cf_t.shape[1]))
            else:
                cf = float(cf_t.mean().mean())
            cf_labels[rc] = f"Region {rc} ({cf:.1%})"
        except Exception:
            cf_labels[rc] = f"Region {rc} (n/a)"

    gdf = gdf.copy()
    gdf['bin'] = gdf['bin'].astype(int)
    gdf['bin_label'] = gdf['bin'].map(cf_labels)

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(column='bin_label', ax=ax, legend=True, cmap=colormap_for_carrier(carrier), legend_kwds={'loc': 'lower left'}, edgecolor='black', linewidth=0.5)
    legend = ax.get_legend()
    # if legend is not None:
    #     legend.set_title("Potential Region\n(Avg. Capacity Factor)")
    
    plt.title(f"Potential Regions for {carrier_full_name(carrier)}\n(Average Capacity Factor in Parentheses)")
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
    

# Plot capacities of resource classes for a given carrier
def plot_resource_classes_capacities(run_name, carrier, save_to_file=True):
    n = get_network(run_name)
    carrier_gens = n.generators.index[n.generators.carrier == carrier]
    rc_tokens = carrier_gens.to_series().str.split().str[2]
    resource_classes = rc_tokens.unique()
    capacities = []
    for rc in resource_classes:
        gen_ids = rc_tokens.index[rc_tokens == rc]
        if len(gen_ids) == 0:
            gen_label = f"DE0 0 {rc} {carrier}"
            if gen_label in n.generators_t.p_max_pu.columns:
                gen_ids = pd.Index([gen_label])
        cap = n.generators.p_nom_opt[gen_ids].sum() / 1e3  # Convert to GW
        capacities.append(cap)
    plt.figure(figsize=(8, 5))
    plt.bar(resource_classes, capacities, color=color_theme(carrier))
    plt.title(f'Installed Capacities by Resource Class for {carrier_full_name(carrier)} [GW]')
    plt.ylabel('Installed Capacity [GW]')
    plt.xticks(fontsize='small')
    plt.grid(True)
    if save_to_file:
        results_directory = f'../results/{run_name}'
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/capacities_{carrier}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()



# ___ Annual revenue statistics based on optimization run ___

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


# Revenue of a carrier and rc for a given year with a price cap during high gas price years
def annual_revenue_with_price_cap(costs, carrier, year, n, rc, bus="DE0 0", cap=130):
    hg = get_hourly_generation(carrier, n, rc)
    hg_year = hg[hg.index.year == year]
    price = get_hourly_marginal_price(n, bus)
    years = n.generators_t.p.index.year.unique()
    price_year = price[price.index.year == year]
    if year in range((max(years) - min(years)) * 4 // 5 + min(years) + 1, max(years) + 1):
        price_year[price_year > cap] = cap

    # Defaults and helper
    if carrier == 'onwind':
        vom = costs.at[carrier, "VOM"]  # in €/MWh
    else: 
        vom = 0.0

    rev_year_capped = (hg_year.values.flatten() * (price_year.values.flatten() - vom)).sum()
    return rev_year_capped


# Annual revenue per MW of a carrier and rc for a given year with a price cap during high gas price years
def annual_revenue_per_MW_with_price_cap(costs, carrier, year, n, rc, bus="DE0 0", cap=130):
    ann_rev_capped = annual_revenue_with_price_cap(costs, carrier, year, n, rc, bus, cap)
    p_nom_opt = n.generators.p_nom_opt[f"DE0 0 {rc} {carrier}"] if carrier != 'offwind' else sum(
        n.generators.p_nom_opt[f"DE0 0 {l} {c}"] 
        for c in ["offwind-ac", "offwind-dc", "offwind-float"] 
        for l in range(n.generators.index[n.generators.carrier == c].size)
    )
    rev_per_MW_year_capped = ann_rev_capped / p_nom_opt if p_nom_opt != 0 else np.nan
    return rev_per_MW_year_capped


# Lost revenue due to price cap 
def lost_rev_pct_pc(run_name, carrier, n, rc, cap=130, bus="DE0 0"):
    costs = get_costs(run_name)
    years = n.generators_t.p.index.year.unique()
    rev_per_mw = sum([annual_revenue_per_MW(costs, carrier, year, n, rc, bus=bus) for year in years])
    rev_per_mw_capped = sum([annual_revenue_per_MW_with_price_cap(costs, carrier, year, n, rc, bus=bus, cap=cap) for year in years])
    lost_revenue_pct = ((rev_per_mw - rev_per_mw_capped) / rev_per_mw * 100 if rev_per_mw != 0 else np.nan)
    return lost_revenue_pct


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



# ___ Monte Carlo Simulation based Revenue Statistics ___

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
def revenue_mcs(costs, run_name, n, T=30, N=10000, cap=130, scenarios=['mb', 'pc', 'pb', 'pi', 'cb']):
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
        sp = {}
        if 'pb' in scenarios:
            sp['pb'] = strike_price_PB(costs, carrier, n, rc)
        if 'pi' in scenarios:
            sp['pi'] = strike_price_PI(costs, carrier, n, rc)
        if 'cb' in scenarios:
            sp['cb'] = strike_price_CB(costs, carrier, n, rc)
        strike_prices[(carrier, rc)] = sp
        
    # 3. Create a lookup table: year -> array of revenues for all streams
    revenue_lookup = {}
    for year in unique_years:
        revenues_by_scenario = {s: [] for s in scenarios}
        for carrier, rc in stream_info:
            sp = strike_prices[(carrier, rc)]
            if 'mb' in scenarios:
                revenues_by_scenario['mb'].append(annual_revenue_per_MW(costs, carrier, year, n, rc))
            if 'pc' in scenarios:
                revenues_by_scenario['pc'].append(annual_revenue_per_MW_with_price_cap(costs, carrier, year, n, rc, cap=cap))
            if 'pb' in scenarios:
                revenues_by_scenario['pb'].append(annual_revenue_PB_per_MW(costs, carrier, year, n, rc, strike_price=sp['pb']))
            if 'pi' in scenarios:
                revenues_by_scenario['pi'].append(annual_revenue_PI_per_MW(costs, carrier, year, n, rc, strike_price=sp['pi']))
            if 'cb' in scenarios:
                revenues_by_scenario['cb'].append(annual_revenue_CB_per_MW(costs, carrier, year, n, rc, strike_price=sp['cb']))
        revenue_lookup[year] = {s: np.array(revs) for s, revs in revenues_by_scenario.items()}

    # 4. Initialize result arrays — one per revenue scenario
    # Shape: (N simulations, T years, num_streams)
    rev_mcs_scenarios = {s: np.zeros((N, T, total_streams), dtype=float) for s in scenarios}
    
    # 5. Broadcast the pre-calculated revenues into the result arrays
    for year in unique_years:
        mask = (yrs_mcs == year)
        if np.any(mask):
            for s in scenarios:
                rev_mcs_scenarios[s][mask] = revenue_lookup[year][s]
            
    # 6. Reshape to (N, T * total_streams) and save each to a separate CSV
    mcs_dir = f"../results/{run_name}/MCS"
    os.makedirs(mcs_dir, exist_ok=True)

    for label, arr in rev_mcs_scenarios.items():
        df = pd.DataFrame(arr.reshape(N, -1))
        df.to_csv(f"{mcs_dir}/revenue_mcs_{label}_T={T}_N={N}.csv", index=False)

    return rev_mcs_scenarios


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


# Average lifetime revenue for a given carrier and resource class from MCS results
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


# Average lifetime revenue per MW for a given carrier averaged over all resource classes from MCS results
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


# Transform annuiity factor to WACC (IRR) by solving the formula: annuity = r / (1 - (1+r)^-T)
def annuity_to_wacc(annuity, T):
    # Solve for irr where: (r / (1 - (1+r)^-T)) - target_ap = 0
    func = lambda r: (r / (1 - (1+r)**-T)) - annuity
    try:
        # Initial guess 7%
        wacc = fsolve(func, 0.07)[0]
    except:
        wacc = np.nan
    return wacc


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


# Effective risk premium due to price cap (difference in WACC with and without price cap)
def effective_risk_premium_due_to_price_cap(run_name, carrier, n, rc, T=0, cap=130, bus="DE0 0"):
    if T == 0:
        if carrier == 'solar' or carrier == 'solar-hsat':
            T = 40
        elif carrier == 'offwind' or carrier == 'offwind-ac' or carrier == 'offwind-dc' or carrier == 'onwind':
            T = 30
        elif carrier == 'offwind-float':
            T = 20
        else:
            T = 25
    costs = get_costs(run_name)
    years = n.generators_t.p.index.year.unique()
    rev_per_mw = sum([annual_revenue_per_MW(costs, carrier, year, n, rc, bus=bus) for year in years])
    if carrier == 'offwind-ac' or carrier == 'offwind-dc':
        carkey = 'offwind'
    elif carrier == 'solar-hsat':
        carkey = 'solar-utility single-axis tracking'
    else:
        carkey = carrier
    ann_old = n.generators.capital_cost[f"DE0 0 {rc} {carrier}"] / (costs.at[carkey, "investment"] * T) - costs.at[carkey, "FOM"] / 100
    wacc_old = annuity_to_wacc(ann_old, T if T > 0 else costs.at[carkey, "lifetime"])
    rev_per_mw_capped = sum([annual_revenue_per_MW_with_price_cap(costs, carrier, year, n, rc, bus=bus, cap=cap) for year in years])
    pct_rev_loss = (rev_per_mw - rev_per_mw_capped) / rev_per_mw if rev_per_mw != 0 else np.nan
    ann_new = n.generators.capital_cost[f"DE0 0 {rc} {carrier}"] / ((1 - pct_rev_loss) * costs.at[carkey, "investment"] * T) - costs.at[carkey, "FOM"] / 100
    wacc_new = annuity_to_wacc(ann_new, T if T > 0 else costs.at[carkey, "lifetime"])
    effective_risk_premium = wacc_new - wacc_old
    return effective_risk_premium


# Updated WACC for each scenario
def wacc_updated(run_name, n, carrier, rc, T=0, N=10000, scenario='mb', cap=130):
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
    if scenario == "pc":
        effective_risk_premium = effective_risk_premium_due_to_price_cap(run_name, carrier, n, rc, T, cap=cap)
        wacc += effective_risk_premium
    return wacc


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
    if scenario in ['mb', 'pc', 'pb', 'pi', 'cb']:
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
    stats_pc = annual_revenue_statistics_mcs(run_name, carrier, n, rc, T=T, N=N, scenario='pc', bus=bus)
    stats_pb = annual_revenue_statistics_mcs(run_name, carrier, n, rc, T=T, N=N, scenario='pb', bus=bus)
    stats_pi = annual_revenue_statistics_mcs(run_name, carrier, n, rc, T=T, N=N, scenario='pi', bus=bus)
    stats_cb = annual_revenue_statistics_mcs(run_name, carrier, n, rc, T=T, N=N, scenario='cb', bus=bus)
    carrier_name = carrier_full_name(carrier)
    print(f"Revenue Statistics for {carrier_name}, Resource Class {rc}:")
    print(f"{'Statistic':<10} {'Market (€)':<18} {'Price Cap (€)':<18} {'PB CfD (€)':<18} {'PI CfD (€)':<18} {'CB CfD (€)':<18}")
    for key in stats_mb.keys():
        print(f"{key:<10} {stats_mb[key]:<18.4f} {stats_pc[key]:<18.4f} {stats_pb[key]:<18.4f} {stats_pi[key]:<18.4f} {stats_cb[key]:<18.4f}")
    return {'mb': stats_mb, 'pc': stats_pc, 'pb': stats_pb, 'pi': stats_pi, 'cb': stats_cb}



# ___ Plots of revenue statistics ___

# Plot lost revenue due to price cap over price cap levels for a given carrier and rc
def plot_lost_revenue_due_to_price_cap(run_name, carrier, n, rc, cap_min=80, cap_max=200, bus="DE0 0", save_to_file=True):
    costs = get_costs(run_name)
    years = n.generators_t.p.index.year.unique()
    rev_per_mw = sum([annual_revenue_per_MW(costs, carrier, year, n, rc, bus=bus) for year in years])
    cap_values = np.linspace(cap_min, cap_max, 13)
    lost_revenue_pct = []
    for cap in cap_values:
        rev_per_mw_capped = sum([annual_revenue_per_MW_with_price_cap(costs, carrier, year, n, rc, bus=bus, cap=cap) for year in years])
        lost_revenue_pct.append((rev_per_mw - rev_per_mw_capped) / rev_per_mw * 100 if rev_per_mw != 0 else np.nan)
    plt.figure(figsize=(10, 6))
    color = color_theme(carrier)
    plt.plot(cap_values, lost_revenue_pct, 'o-', color=color)
    plt.xlabel('Price Cap [€/MWh]')
    plt.ylabel('Lost Revenue (%)')
    plt.title(f'Lost Revenue Due to Price Cap for {carrier_full_name(carrier)}, RC {rc}')
    plt.grid(True)
    if save_to_file:
        results_directory = f'../results/{run_name}'
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/lost_revenue_due_to_price_cap_{carrier}_{rc}_{int(cap_min)}_{int(cap_max)}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


# Plot effective higher WACC due to price cap over price cap levels for a given carrier and rc
def plot_effective_risk_premium_due_to_price_cap(run_name, carrier, n, rc, cap_min=80, cap_max=200, T=0, bus="DE0 0", save_to_file=True):
    if T == 0:
        if carrier == 'solar' or carrier == 'solar-hsat':
            T = 40
        elif carrier == 'offwind' or carrier == 'offwind-ac' or carrier == 'offwind-dc' or carrier == 'onwind':
            T = 30
        elif carrier == 'offwind-float':
            T = 20
        else:
            T = 25
    costs = get_costs(run_name)
    years = n.generators_t.p.index.year.unique()
    rev_per_mw = sum([annual_revenue_per_MW(costs, carrier, year, n, rc, bus=bus) for year in years])
    cap_values = np.linspace(cap_min, cap_max, 13)
    if carrier == 'offwind-ac' or carrier == 'offwind-dc':
        carkey = 'offwind'
    elif carrier == 'solar-hsat':
        carkey = 'solar-utility single-axis tracking'
    else:
        carkey = carrier
    ann_old = n.generators.capital_cost[f"DE0 0 {rc} {carrier}"] / (costs.at[carkey, "investment"] * T) - costs.at[carkey, "FOM"] / 100
    wacc_old = annuity_to_wacc(ann_old, T if T > 0 else costs.at[carkey, "lifetime"])
    effective_risk_premium = []
    for cap in cap_values:
        rev_per_mw_capped = sum([annual_revenue_per_MW_with_price_cap(costs, carrier, year, n, rc, bus=bus, cap=cap) for year in years])
        pct_rev_loss = (rev_per_mw - rev_per_mw_capped) / rev_per_mw if rev_per_mw != 0 else np.nan
        ann_new = n.generators.capital_cost[f"DE0 0 {rc} {carrier}"] / ((1 - pct_rev_loss) * costs.at[carkey, "investment"] * T) - costs.at[carkey, "FOM"] / 100
        wacc_new = annuity_to_wacc(ann_new, T if T > 0 else costs.at[carkey, "lifetime"])
        effective_risk_premium.append(wacc_new - wacc_old)
    plt.figure(figsize=(10, 6))
    color = color_theme(carrier)
    plt.plot(cap_values, effective_risk_premium, 'o-', color=color)
    plt.xlabel('Price Cap [€/MWh]')
    plt.ylabel('Effective Risk Premium')
    plt.title(f'Effective Risk Premium Due to Price Cap for {carrier_full_name(carrier)}, RC {rc}')
    plt.grid(True)
    if save_to_file:
        results_directory = f'../results/{run_name}'
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/effective_risk_premium_due_to_price_cap_{carrier}_{rc}_{int(cap_min)}_{int(cap_max)}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


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


# Scenario Name Mapping for Plot Titles
def scenario_name_mapping(scenario):
    mapping = {
        'mb': 'Direct Marketing',
        'pb': 'a Production-Based CfD',
        'pi': 'a Production-Independent CfD',
        'cb': 'a Capacity-Based CfD'
    }
    return mapping.get(scenario, scenario)


# Plot annual revenue of a carrier for all years
def plot_annual_revenue_per_MW(run_name, scenario, carrier, rc, bus="DE0 0", save_to_file=True):
    n = get_network(run_name)
    years = n.generators_t.p.index.year.unique()
    costs = get_costs(run_name)
    if scenario == 'mb':
        revenues = [annual_revenue_per_MW(costs, carrier, year, n, rc) for year in years]
    elif scenario == 'pb':
        sp = strike_price_PB(costs, carrier, n, rc)
        revenues = [annual_revenue_PB_per_MW(costs, carrier, year, n, rc, strike_price=sp, bus=bus) for year in years]
    elif scenario == 'pi':
        sp = strike_price_PI(costs, carrier, n, rc)
        revenues = [annual_revenue_PI_per_MW(costs, carrier, year, n, rc, strike_price=sp, bus=bus) for year in years]
    elif scenario == 'cb':
        sp = strike_price_CB(costs, carrier, n, rc)
        revenues = [annual_revenue_CB_per_MW(costs, carrier, year, n, rc, strike_price=sp, bus=bus) for year in years]
    else:
        print(f"Error: Scenario {scenario} not recognized.")
        return
    
    # Divide into quintiles
    n_years = len(years)
    q_size = n_years // 5
    
    # First quintile (high prices)
    low_idx = list(range(0, q_size))
    low_revenues = [revenues[i] for i in low_idx]
    low_years = [years[i] for i in low_idx]
    low_mean = np.mean(low_revenues)
    
    # Middle three quintiles (average)
    avg_idx = list(range(q_size, 4 * q_size))
    avg_revenues = [revenues[i] for i in avg_idx]
    avg_years = [years[i] for i in avg_idx]
    avg_mean = np.mean(avg_revenues)
    
    # Fifth quintile (low prices)
    high_idx = list(range(4 * q_size, n_years))
    high_revenues = [revenues[i] for i in high_idx]
    high_years = [years[i] for i in high_idx]
    high_mean = np.mean(high_revenues)
    
    plt.figure(figsize=(10, 6))
    color = color_theme(carrier)    
    plt.plot(years, revenues, 'o', color=color) # Points
    
    # Draw mean lines only in their respective sections
    plt.plot([min(high_years), max(high_years)], [high_mean, high_mean], color='black', linestyle='--', linewidth=2)
    plt.plot([min(avg_years), max(avg_years)], [avg_mean, avg_mean], color='black', linestyle='--', linewidth=2)
    plt.plot([min(low_years), max(low_years)], [low_mean, low_mean], color='black', linestyle='--', linewidth=2)

    # Draw thin vertical lines between sections
    x_sep_1 = (max(low_years) + min(avg_years)) / 2
    x_sep_2 = (max(avg_years) + min(high_years)) / 2
    plt.axvline(x=x_sep_1, color='black', linestyle='-', linewidth=1)
    plt.axvline(x=x_sep_2, color='black', linestyle='-', linewidth=1)
    
    carrier_name = carrier_full_name(carrier)
    scenario_name = scenario_name_mapping(scenario)
    plt.title(f'Annual Revenue of {carrier_name} (RC {rc}) per MW Installed Capacity for {scenario_name}')
    plt.xlabel('Weather Year')
    plt.ylabel('Annual Revenue [t €] per MW installed capacity')
    plt.ylim(0.9*np.min(revenues), 1.15*np.max(revenues))  # Set y-axis limit slightly above max revenue for better visualization
    ymin, ymax = plt.ylim()

    def fmt_thousands_dot(x):
        return f"{x/1e3:,.1f}"#.replace(",", ".")

    def fmt_million_4digits(x):
        return f"{x / 1e6:,.4g}"

    # Section labels with average annual revenues
    plt.text(np.mean(low_years)-0.5, ymax * 0.99, 
             rf"Low Gas Price Years" + "\n" + rf"$\overline{{R_t}}={fmt_thousands_dot(low_mean)}\,\mathrm{{t\,€/MW}}$", 
             ha='center', va='top', fontsize=11, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=0.5))
    plt.text(np.mean(avg_years), ymax * 0.99, 
             rf"Average Gas Price Years" + "\n" + rf"$\overline{{R_t}}={fmt_thousands_dot(avg_mean)}\,\mathrm{{t\,€/MW}}$", 
             ha='center', va='top', fontsize=11, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=0.5))
    plt.text(np.mean(high_years)+0.5, ymax * 0.99, 
             rf"High Gas Price Years" + "\n" + rf"$\overline{{R_t}}={fmt_thousands_dot(high_mean)}\,\mathrm{{t\,€/MW}}$", 
             ha='center', va='top', fontsize=11, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=0.5))

    # Thousands separators for y-axis tick labels
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt_thousands_dot(x)))
    plt.grid(True)
    if save_to_file:
        results_directory = f'../results/{run_name}'
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/annual_revenue_per_MW_{carrier}_{rc}_{scenario}.png'
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
    data_range = float(np.max(flat_data) - np.min(flat_data))
    n_bins = max(20, int(140 * data_range))
    plt.hist(flat_data, bins=n_bins, color=color_theme(carrier), edgecolor='black', density=True, alpha=0.6, label='Histogram ($\sigma$={:.4f})'.format(np.std(flat_data)))
    
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    
    fit_stats = {}  # dict of {name: ks_stat} for best-fit comparison
    
    if scenario == 'mb' or scenario == 'pc':
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
        plt.plot(x, p_t, 'm:', linewidth=2, label=f"Student's t (KS = {ks_stat_t:.4f},\n df = {df_t:.1f}, scale = {scale_t:.4f})")
        fit_stats["Student's t"] = ks_stat_t

        # # --- Johnson SU Distribution (CfD scenarios) ---
        # a_jsu, b_jsu, loc_jsu, scale_jsu = johnsonsu.fit(flat_data)
        # p_jsu = johnsonsu.pdf(x, a_jsu, b_jsu, loc_jsu, scale_jsu)
        # ks_stat_jsu, p_val_jsu = kstest(flat_data, 'johnsonsu', args=(a_jsu, b_jsu, loc_jsu, scale_jsu))
        # plt.plot(x, p_jsu, 'c-', linewidth=2, label=f'Johnson SU (KS = {ks_stat_jsu:.4f})')
        # fit_stats['Johnson SU'] = ks_stat_jsu

    scenario_name = scenario_name_mapping(scenario)
    plt.title(f'Lifetime Normalized Revenue Distribution for {gen_name} over {t} years for {scenario_name}')
    plt.xlabel('Lifetime Normalized Revenue')
    plt.ylabel('Frequency')
    plt.xlim(0.7, 1.4)
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
    if scenario == 'mb' or scenario == 'pc':
        print(f"Log-Normal:  KS Stat = {ks_stat_lognorm:.4f}, p-value = {p_val_lognorm:.4e}")
    else:
        print(f"Student's t: KS Stat = {ks_stat_t:.4f}, p-value = {p_val_t:.4e} (df = {df_t:.1f})")
        # print(f"Johnson SU:  KS Stat = {ks_stat_jsu:.4f}, p-value = {p_val_jsu:.4e}")
    
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
def plot_lifetime_risk_per_rc(run_name, carrier, T=0, N=10000, save_to_file=True, scenarios=['mb', 'pc', 'pb', 'pi', 'cb']):
    n = get_network(run_name)
    if T == 0:
        if carrier == 'solar' or carrier == 'solar-hsat':
            T = 40
        elif carrier == 'offwind' or carrier == 'offwind-ac' or carrier == 'offwind-dc' or carrier == 'onwind':
            T = 30
        elif carrier == 'offwind-float':
            T = 20
        else:
            T = 25

    n_rc_actual = n.generators.index[n.generators.carrier == carrier].size
    n_rc = n_rc_actual if n_rc_actual > 0 else 1
    rc_values = list(range(n_rc))

    scenario_colors = {
        'mb': cm_5,
        'pc': cm_6,
        'pb': cm_1,
        'pi': cm_2,
        'cb': cm_3,
    }
    scenario_labels = {
        'mb': 'Direct Marketing',
        'pc': 'DM with Price Cap',
        'pb': 'Production-based CfD',
        'pi': 'Production-independent CfD',
        'cb': 'Capacity-based CfD',
    }

    plt.figure(figsize=(10, 6))
    all_valid_risks = []

    for scenario in scenarios:
        risks = []
        for rc in rc_values:
            rev_mcs = lifetime_revenues_mcs_per_MW(run_name, n, carrier, rc, T=T, N=N, scenario=scenario)
            if rev_mcs is None:
                risks.append(np.nan)
                continue
            mean_rev = np.mean(rev_mcs)
            risk = np.std(rev_mcs) / mean_rev if mean_rev != 0 else np.nan
            # if scenario == 'pc':
            #     erp = effective_risk_premium_due_to_price_cap(run_name, carrier, n, rc, T=T)
            #     risk += erp
            risk_pct = risk * 100 if np.isfinite(risk) else np.nan
            risks.append(risk_pct)
            if np.isfinite(risk_pct):
                all_valid_risks.append(risk_pct)

        color = scenario_colors.get(scenario, color_theme(carrier))
        label = scenario_labels.get(scenario, scenario)
        plt.plot(rc_values, risks, 'o-', color=color, markeredgecolor='black', linewidth=1.8, label=label)

    carrier_name = carrier_full_name(carrier)
    plt.title(rf'Lifetime Revenue Risk $\frac{{\sigma}}{{\mu}}$ of {carrier_name} per Region')
    plt.xlabel('Region')
    plt.xticks(rc_values)
    plt.ylabel('Lifetime Revenue Risk [%]')
    if all_valid_risks:
        plt.ylim(0.0, 1.2 * np.max(all_valid_risks))
    plt.grid(True)
    plt.legend()

    if save_to_file:
        results_directory = f'../results/{run_name}'
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/lifetime_risk_per_resource_class_{carrier}_scenarios.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


# Plot effective WACC for all resource classes of a carrier
def plot_effective_wacc_per_rc(run_name, carrier, T=0, N=10000, save_to_file=True, scenarios=['mb', 'pc', 'pb', 'pi', 'cb']):
    n = get_network(run_name)
    if T == 0:
        if carrier == 'solar' or carrier == 'solar-hsat':
            T = 40
        elif carrier == 'offwind' or carrier == 'offwind-ac' or carrier == 'offwind-dc' or carrier == 'onwind':
            T = 30
        elif carrier == 'offwind-float':
            T = 20
        else:
            T = 25

    n_rc_actual = n.generators.index[n.generators.carrier == carrier].size
    n_rc = n_rc_actual if n_rc_actual > 0 else 1
    rc_values = list(range(n_rc))

    scenario_colors = {
        'mb': cm_5,
        'pc': cm_6,
        'pb': cm_1,
        'pi': cm_2,
        'cb': cm_3,
    }
    scenario_labels = {
        'mb': 'Direct Marketing',
        'pc': 'DM with Political Risk',
        'pb': 'Production-based CfD',
        'pi': 'Production-independent CfD',
        'cb': 'Capacity-based CfD',
    }

    plt.figure(figsize=(10, 6))
    all_valid_waccs = []

    for scenario in scenarios:
        waccs = []
        for rc in rc_values:
            gen_label = f"DE0 0 {rc} {carrier}"
            if gen_label in n.generators.index and n.generators.p_nom_opt[gen_label] > 0:
                wacc = wacc_updated(run_name, n, carrier, rc, T=T, N=N, scenario=scenario)
                wacc_pct = wacc * 100 if np.isfinite(wacc) else np.nan
            else:
                wacc_pct = np.nan
            waccs.append(wacc_pct)
            if np.isfinite(wacc_pct):
                all_valid_waccs.append(wacc_pct)

        color = scenario_colors.get(scenario, color_theme(carrier))
        label = scenario_labels.get(scenario, scenario)
        plt.plot(rc_values, waccs, 'o-', color=color, markeredgecolor='black', linewidth=1.8, label=label)

    carrier_name = carrier_full_name(carrier)
    plt.title(rf'Effective WACC of {carrier_name} per Region')
    plt.xlabel('Region')
    plt.xticks(rc_values)
    plt.ylabel('Effective WACC [%]')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
    # if all_valid_waccs:
    #     plt.ylim(0.9 * np.min(all_valid_waccs), 1.2 * np.max(all_valid_waccs))
    plt.grid(True)
    plt.legend()

    if save_to_file:
        results_directory = f'../results/{run_name}'
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/effective_wacc_per_resource_class_{carrier}_scenarios.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


# Violin plot for annual revenues for different CfD instruments
def plot_revenue_violin(run_name, carrier, rc, T=0, N=10000, save_to_file=True, scenario='mb', bus="DE0 0"):
    n = get_network(run_name)
    if T == 0:
        if carrier == 'solar' or carrier == 'solar-hsat':
            T = 40
        elif carrier == 'offwind' or carrier == 'offwind-ac' or carrier == 'offwind-dc' or carrier == 'onwind':
            T = 30
        elif carrier == 'offwind-float':
            T = 20
        else:
            T = 25

    lifetime_revenues = lifetime_revenues_mcs_per_MW(run_name, n, carrier, rc, T=T, N=N, scenario=scenario)
    
    carrier_name = carrier_full_name(carrier)
    scenario_name = scenario_name_mapping(scenario)

    df = pd.DataFrame({'run_name': [scenario_name] * N, 'revenue': lifetime_revenues.flatten()})
    plt.figure(figsize=(10, 6))
    
    # Try using color_theme (as used in other plotting functions) with a fallback
    try:
        color = color_theme(carrier)
    except NameError:
        color = 'blue'

    sns.violinplot(x='run_name', y='revenue', data=df, color=color)
    plt.title(f'Distribution of Lifetime Revenues per MW of {carrier_name} for {scenario_name}', fontsize=12)
    plt.xlabel(None)
    plt.ylabel('Average Annual Revenue [€] per MW installed capacity', fontsize=12)
    plt.grid(True)
    # plt.ylim(0, None)  # Start y-axis at 0
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if save_to_file:
        results_directory = f'../results/{run_name}'
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        # The filename should probably be more specific to the carrier and rc
        filename = f'{output_dir}/revenue_violin_plot_{carrier}_rc{rc}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()



# ___ Statistics on public CfD payments ___

# Precompute strike prices for all (carrier, rc) combinations to speed up state-payment statistics calculations
def _precompute_state_cfd_strike_prices(costs, n, scenario, bus="DE0 0"):
    """Compute strike prices once per (carrier, rc) for state-payment statistics."""
    if scenario not in ['pb', 'pi', 'cb']:
        return {}

    strike_prices = {}
    vre_carriers = ['offwind', 'offwind-ac', 'offwind-dc', 'offwind-float', 'onwind', 'solar', 'solar-hsat', 'solar rooftop']

    for carrier in vre_carriers:
        for rc in range(n.generators.index[n.generators.carrier == carrier].size):
            if scenario == 'pb':
                strike_prices[(carrier, rc)] = strike_price_PB(costs, carrier, n, rc, bus=bus)
            elif scenario == 'pi':
                strike_prices[(carrier, rc)] = strike_price_PI(costs, carrier, n, rc, bus=bus)
            elif scenario == 'cb':
                strike_prices[(carrier, rc)] = strike_price_CB(costs, carrier, n, rc, bus=bus)

    return strike_prices


# State CfD payments in a given scenario and year (positive for state -> operator)
def annual_state_cfd_payments(run_name, n, year, scenario='mb', cap=130, bus="DE0 0", costs=None, strike_prices=None):
    if scenario == 'mb':
        return 0  # No CfD payments in market-based scenario
    else:
        if costs is None:
            costs = get_costs(run_name)
        vre_carriers = ['offwind-ac', 'offwind-dc', 'offwind-float', 'onwind', 'solar', 'solar-hsat']
        rev_mb = 0
        rev_sc = 0
        for carrier in vre_carriers:
            for rc in range(n.generators.index[n.generators.carrier == carrier].size):
                rev_mb_stream = annual_revenue(costs, carrier, year, n, rc, bus=bus)
                rev_mb += rev_mb_stream
                if scenario == 'pc':
                    rev_sc += annual_revenue_with_price_cap(costs, carrier, year, n, rc, cap=cap, bus=bus)
                elif scenario == 'pb':
                    sp = strike_prices[(carrier, rc)] if strike_prices is not None else strike_price_PB(costs, carrier, n, rc, bus=bus)
                    if not np.isfinite(sp):
                        rev_sc += rev_mb_stream
                    else:
                        rev_sc += annual_revenue_PB(costs, carrier, year, n, rc, strike_price=sp, bus=bus)
                elif scenario == 'pi':
                    sp = strike_prices[(carrier, rc)] if strike_prices is not None else strike_price_PI(costs, carrier, n, rc, bus=bus)
                    if not np.isfinite(sp):
                        rev_sc += rev_mb_stream
                    else:
                        rev_sc += annual_revenue_PI(costs, carrier, year, n, rc, strike_price=sp, bus=bus)
                elif scenario == 'cb':
                    sp = strike_prices[(carrier, rc)] if strike_prices is not None else strike_price_CB(costs, carrier, n, rc, bus=bus)
                    if not np.isfinite(sp):
                        rev_sc += rev_mb_stream
                    else:
                        rev_sc += annual_revenue_CB(costs, carrier, year, n, rc, strike_price=sp, bus=bus)
        payments = rev_sc - rev_mb
        return payments


# Calculate annual state CfD payments for each year for a given scenario and store in a DataFrame
def calculate_annual_state_cfd_payments(run_name, n, scenario='mb', cap=130, bus="DE0 0"):
    years = n.generators_t.p.index.year.unique()
    costs = get_costs(run_name)
    strike_prices = _precompute_state_cfd_strike_prices(costs, n, scenario, bus=bus)
    payments_data = []
    for year in years:
        payments = annual_state_cfd_payments(
            run_name,
            n,
            year,
            scenario=scenario,
            cap=cap,
            bus=bus,
            costs=costs,
            strike_prices=strike_prices,
        )
        payments_data.append({'year': year, 'payments': payments})
    df_payments = pd.DataFrame(payments_data)
    file_path = f'../results/{run_name}/state_cfd_payments/state_cfd_payments_{scenario}.csv'
    os_dir = os.path.dirname(file_path)
    if not os.path.exists(os_dir):
        os.makedirs(os_dir, exist_ok=True)
    df_payments.to_csv(file_path, index=False)
    return df_payments


# Average state CfD payments in a given scenario across all years
def average_state_cfd_payments(run_name, n, scenario='mb', cap=130, bus="DE0 0"):
    # Try loading from CSV first
    file_path = f'../results/{run_name}/state_cfd_payments/state_cfd_payments_{scenario}.csv'
    if os.path.exists(file_path):
        df_payments = pd.read_csv(file_path)
        if 'payments' in df_payments.columns:
            return df_payments['payments'].mean()
        else:
            years = n.generators_t.p.index.year.unique()
            costs = get_costs(run_name)
            strike_prices = _precompute_state_cfd_strike_prices(costs, n, scenario, bus=bus)
            total_payments = sum(
                annual_state_cfd_payments(
                    run_name,
                    n,
                    year,
                    scenario=scenario,
                    cap=cap,
                    bus=bus,
                    costs=costs,
                    strike_prices=strike_prices,
                )
                for year in years
            )
            return total_payments / len(years) if years.size > 0 else 0


# Average absolute state CfD payments in a given scenario across all years
def average_abs_state_cfd_payments(run_name, n, scenario='mb', cap=130, bus="DE0 0"):
    # Try loading from CSV first
    file_path = f'../results/{run_name}/state_cfd_payments/state_cfd_payments_{scenario}.csv'
    if os.path.exists(file_path):
        df_payments = pd.read_csv(file_path)
        if 'payments' in df_payments.columns:
            return df_payments['payments'].abs().mean()
        else:
            years = n.generators_t.p.index.year.unique()
            costs = get_costs(run_name)
            strike_prices = _precompute_state_cfd_strike_prices(costs, n, scenario, bus=bus)
            total_abs_payments = sum(
                abs(annual_state_cfd_payments(
                    run_name,
                    n,
                    year,
                    scenario=scenario,
                    cap=cap,
                    bus=bus,
                    costs=costs,
                    strike_prices=strike_prices,
                ))
                for year in years
            )
            return total_abs_payments / len(years) if years.size > 0 else 0


# Standard deviation of state CfD payments in a given scenario across all years
def std_dev_state_cfd_payments(run_name, n, scenario='mb', cap=130, bus="DE0 0"):
    # Try loading from CSV first
    file_path = f'../results/{run_name}/state_cfd_payments/state_cfd_payments_{scenario}.csv'
    if os.path.exists(file_path):
        df_payments = pd.read_csv(file_path)
        if 'payments' in df_payments.columns:
            return df_payments['payments'].std()
        else:
            years = n.generators_t.p.index.year.unique()
            costs = get_costs(run_name)
            strike_prices = _precompute_state_cfd_strike_prices(costs, n, scenario, bus=bus)
            payments = [
                annual_state_cfd_payments(
                    run_name,
                    n,
                    year,
                    scenario=scenario,
                    cap=cap,
                    bus=bus,
                    costs=costs,
                    strike_prices=strike_prices,
                )
                for year in years
            ]
            return np.std(payments)


# Normalized standard deviation of state CfD payments (risk) in a given scenario across all years
def risk_state_cfd_payments(run_name, n, scenario='mb', cap=130, bus="DE0 0"):
    # Try loading from CSV first
    file_path = f'../results/{run_name}/state_cfd_payments/state_cfd_payments_{scenario}.csv'
    if os.path.exists(file_path):
        df_payments = pd.read_csv(file_path)
        if 'payments' in df_payments.columns:
            sc = system_costs(n, bus=bus)
            std_dev_payment = df_payments['payments'].std()
            return std_dev_payment * len(df_payments) / sc if sc != 0 else np.nan
    else:
        years = n.generators_t.p.index.year.unique()
        costs = get_costs(run_name)
        strike_prices = _precompute_state_cfd_strike_prices(costs, n, scenario, bus=bus)
        payments = [
            annual_state_cfd_payments(
                run_name,
                n,
                year,
                scenario=scenario,
                cap=cap,
                bus=bus,
                costs=costs,
                strike_prices=strike_prices,
            )
            for year in years
        ]
        sc = system_costs(n, bus=bus)
        std_dev_payment = np.std(payments)
        return std_dev_payment / sc * len(years) if sc != 0 else np.nan


# Aggregated market-based revenue risk across all VRE carriers
def aggregated_revenue_std_dev(run_name, n, scenario='mb', bus="DE0 0"):
    costs = get_costs(run_name)
    vre_carriers = ['offwind-ac', 'offwind-dc', 'offwind-float', 'onwind', 'solar', 'solar-hsat', 'solar rooftop']
    std_devs = 0.0
    years = n.generators_t.p.index.year.unique()
    if scenario != 'mb':
        strike_price = _precompute_state_cfd_strike_prices(costs, n, scenario, bus=bus)
    for carrier in vre_carriers:
        for rc in range(n.generators.index[n.generators.carrier == carrier].size):
            revs = []
            sp = strike_price.get((carrier, rc), np.nan) if scenario != 'mb' else np.nan
            for year in years:
                if scenario == 'mb':
                    rev = annual_revenue(costs, carrier, year, n, rc, bus=bus)
                elif scenario == 'pb':
                    rev = annual_revenue_PB(costs, carrier, year, n, rc, strike_price=sp, bus=bus) if np.isfinite(sp) else annual_revenue(costs, carrier, year, n, rc, bus=bus)
                elif scenario == 'pi':
                    rev = annual_revenue_PI(costs, carrier, year, n, rc, strike_price=sp, bus=bus) if np.isfinite(sp) else annual_revenue(costs, carrier, year, n, rc, bus=bus)
                elif scenario == 'cb':
                    rev = annual_revenue_CB(costs, carrier, year, n, rc, strike_price=sp, bus=bus) if np.isfinite(sp) else annual_revenue(costs, carrier, year, n, rc, bus=bus)
                else:
                    rev = np.nan
                if np.isfinite(rev):
                    revs.append(rev)

            if len(revs) > 0:
                std_devs += np.std(revs)
    return std_devs 


# Absolute portfolio effect for each CfD type across all years
def portfolio_effect_abs(run_name, n, scenario='mb', bus="DE0 0"):
    std_state = std_dev_state_cfd_payments(run_name, n, scenario, bus=bus)
    std_agg = aggregated_revenue_std_dev(run_name, n, scenario='mb', bus=bus)
    return std_agg - std_state


# Relative portfolio effect for each CfD type across all years
def portfolio_effect_pct(run_name, n, scenario='mb', bus="DE0 0"):
    std_state = std_dev_state_cfd_payments(run_name, n, scenario, bus=bus)
    std_agg = aggregated_revenue_std_dev(run_name, n, scenario='mb', bus=bus)
    return (std_agg - std_state) / std_agg * 100 if std_agg != 0 else np.nan


# Electricity price time series under a levying system for a given CfD type
def price_time_series_with_levy(run_name, n, scenario='mb', cap=130, bus="DE0 0"):
    years = np.array(sorted(n.generators_t.p.index.year.unique()))
    # Try loading from CSV first
    file_path = f'../results/{run_name}/state_cfd_payments/state_cfd_payments_{scenario}.csv'
    if os.path.exists(file_path):
        df_payments = pd.read_csv(file_path)
        if {'year', 'payments'}.issubset(df_payments.columns):
            payments_by_year = df_payments.set_index('year')['payments']
        elif 'payments' in df_payments.columns and len(df_payments) == len(years):
            payments_by_year = pd.Series(df_payments['payments'].to_numpy(), index=years)
        else:
            costs = get_costs(run_name)
            strike_prices = _precompute_state_cfd_strike_prices(costs, n, scenario, bus=bus)
            payments = [
                annual_state_cfd_payments(
                    run_name,
                    n,
                    year,
                    scenario=scenario,
                    cap=cap,
                    bus=bus,
                    costs=costs,
                    strike_prices=strike_prices,
                )
                for year in years
            ]
            payments_by_year = pd.Series(payments, index=years)
    else:
        costs = get_costs(run_name)
        strike_prices = _precompute_state_cfd_strike_prices(costs, n, scenario, bus=bus)
        payments = [
            annual_state_cfd_payments(
                run_name,
                n,
                year,
                scenario=scenario,
                cap=cap,
                bus=bus,
                costs=costs,
                strike_prices=strike_prices,
            )
            for year in years
        ]
        payments_by_year = pd.Series(payments, index=years)

    prices = get_hourly_marginal_price(n, bus=bus)
    hg = n.generators_t.p.sum(axis=1)
    # vre_carriers = ['offwind-ac', 'offwind-dc', 'offwind-float', 'onwind', 'solar', 'solar-hsat']
    # hg_components = []
    # for carrier in vre_carriers:
    #     n_rc = n.generators.index[n.generators.carrier == carrier].size
    #     for rc in range(n_rc):
    #         hg_components.append(get_hourly_generation(carrier, n, rc))

    # if hg_components:
    #     hg = pd.concat(hg_components, axis=1).sum(axis=1)
    # else:
    #     hg = pd.Series(0.0, index=prices.index)

    avg_annual_prices = []
    for year in years:
        mask = hg.index.year == year
        total_gen = hg[mask].sum()
        if total_gen > 0:
            payment_year = payments_by_year.loc[year] if year in payments_by_year.index else 0.0
            avg_price = ((prices[mask] * hg[mask]).sum() + payment_year) / total_gen
            avg_annual_prices.append(avg_price)
        else:
            avg_annual_prices.append(np.nan)

    return pd.Series(avg_annual_prices, index=years, name='price_with_levy')


# ___ Plots of public CfD payments statistics ___

# Plot annual state CfD payments across all years
def plot_annual_state_CfD_payments(run_name, scenario='mb', cap=130, bus="DE0 0", save_to_file=True):
    n = get_network(run_name)
    years = n.generators_t.p.index.year.unique()
    years = np.array(sorted(years))

    file_path = f'../results/{run_name}/state_cfd_payments/state_cfd_payments_{scenario}.csv'
    if os.path.exists(file_path):
        df_payments = pd.read_csv(file_path)
        if {'year', 'payments'}.issubset(df_payments.columns):
            df_payments = df_payments.sort_values('year')
            years = df_payments['year'].to_numpy()
            payments = df_payments['payments'].to_numpy()
        elif 'payments' in df_payments.columns and len(df_payments) == len(years):
            payments = df_payments['payments'].to_numpy()
        else:
            costs = get_costs(run_name)
            strike_prices = _precompute_state_cfd_strike_prices(costs, n, scenario, bus=bus)
            payments = [
                annual_state_cfd_payments(
                    run_name,
                    n,
                    year,
                    scenario=scenario,
                    cap=cap,
                    bus=bus,
                    costs=costs,
                    strike_prices=strike_prices,
                )
                for year in years
            ]
    else:
        costs = get_costs(run_name)
        strike_prices = _precompute_state_cfd_strike_prices(costs, n, scenario, bus=bus)
        payments = [
            annual_state_cfd_payments(
                run_name,
                n,
                year,
                scenario=scenario,
                cap=cap,
                bus=bus,
                costs=costs,
                strike_prices=strike_prices,
            )
            for year in years
        ]

    # Divide into quintiles
    n_years = len(years)
    q_size = n_years // 5

    low_idx = list(range(0, q_size))
    low_payments = [payments[i] for i in low_idx]
    low_years = [years[i] for i in low_idx]
    low_mean = np.mean(low_payments)

    avg_idx = list(range(q_size, 4 * q_size))
    avg_payments = [payments[i] for i in avg_idx]
    avg_years = [years[i] for i in avg_idx]
    avg_mean = np.mean(avg_payments)

    high_idx = list(range(4 * q_size, n_years))
    high_payments = [payments[i] for i in high_idx]
    high_years = [years[i] for i in high_idx]
    high_mean = np.mean(high_payments)

    plt.figure(figsize=(10, 6))
    color = cm_4
    plt.plot(years, payments, 'o', color=color)

    # Draw mean lines only in their respective sections
    plt.plot([min(high_years), max(high_years)], [high_mean, high_mean], color='black', linestyle='--', linewidth=2)
    plt.plot([min(avg_years), max(avg_years)], [avg_mean, avg_mean], color='black', linestyle='--', linewidth=2)
    plt.plot([min(low_years), max(low_years)], [low_mean, low_mean], color='black', linestyle='--', linewidth=2)

    # Draw thin vertical lines between sections
    x_sep_1 = (max(low_years) + min(avg_years)) / 2
    x_sep_2 = (max(avg_years) + min(high_years)) / 2
    plt.axvline(x=x_sep_1, color='black', linestyle='-', linewidth=1)
    plt.axvline(x=x_sep_2, color='black', linestyle='-', linewidth=1)

    def fmt_billion_4digits(x, n=2):
        return f"{x/1e9:,.{n}f}"#.replace(",", ".")

    scenario_name = scenario_name_mapping(scenario)
    plt.title(f'Annual State CfD Payments for {scenario_name}')
    plt.xlabel('Weather Year')
    plt.ylabel('Annual State CfD Payments [bn €]')
    y_min_data = np.min(payments)
    y_max_data = np.max(payments)
    y_range = y_max_data - y_min_data
    if y_range == 0:
        y_pad = max(1.0, 0.1 * abs(y_max_data))
    else:
        y_pad = 0.1 * y_range
    plt.ylim(y_min_data - y_pad, 1.2 * (y_max_data + y_pad))
    ymin, ymax = plt.ylim()

    # Section labels with average annual payments in million EUR (4 significant digits)
    plt.text(np.mean(low_years) - 0.5, ymax * 0.97,
             rf"Low Gas Price Years" + "\n" + rf"$\overline{{P_t}}={fmt_billion_4digits(low_mean)}\,\mathrm{{bn\,€}}$",
             ha='center', va='top', fontsize=11, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=0.5))
    plt.text(np.mean(avg_years), ymax * 0.97,
             rf"Average Gas Price Years" + "\n" + rf"$\overline{{P_t}}={fmt_billion_4digits(avg_mean)}\,\mathrm{{bn\,€}}$",
             ha='center', va='top', fontsize=11, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=0.5))
    plt.text(np.mean(high_years) + 0.5, ymax * 0.97,
             rf"High Gas Price Years" + "\n" + rf"$\overline{{P_t}}={fmt_billion_4digits(high_mean)}\,\mathrm{{bn\,€}}$",
             ha='center', va='top', fontsize=11, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=0.5))

    total_avg_abs_payment = np.mean(np.abs(payments))
    plt.gcf().subplots_adjust(bottom=0.16)
    plt.gca().text(
        0.5,
        0.03,
        rf"Total Average Absolute Payment: $\overline{{|P_t|}}={fmt_billion_4digits(total_avg_abs_payment)}\,\mathrm{{bn\,€}}$",
        transform=plt.gca().transAxes,
        ha='center',
        va='bottom',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=0.6),
    )

    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt_billion_4digits(x, n=0)))
    plt.grid(True)
    if save_to_file:
        results_directory = f'../results/{run_name}'
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/annual_state_CfD_payments_{scenario}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


# Scatterplot where for each year, aggregate wind and solar CfD payments are plotted against each other
def plot_wind_vs_solar_cfd_payments(run_name, scenario='mb', bus="DE0 0", save_to_file=True):
    n = get_network(run_name)
    years = n.generators_t.p.index.year.unique()
    years = np.array(sorted(years))
    costs = get_costs(run_name)
    strike_prices = _precompute_state_cfd_strike_prices(costs, n, scenario, bus=bus)

    wind_payments = []
    solar_payments = []

    for year in years:
        wind_pay = 0
        solar_pay = 0
        
        for carrier in ['onwind'    , 'offwind-ac', 'offwind-dc', 'offwind-float']:
            for rc in range(n.generators.index[n.generators.carrier == carrier].size):
                rev_mb = annual_revenue(costs, carrier, year, n, rc, bus=bus)
                rev_sc = 0
                if scenario == 'pb':
                    sp = strike_prices[(carrier, rc)]
                    rev_sc = annual_revenue_PB(costs, carrier, year, n, rc, strike_price=strike_prices[(carrier, rc)], bus=bus) if np.isfinite(sp) else rev_mb
                elif scenario == 'pi':
                    sp = strike_prices[(carrier, rc)]
                    rev_sc = annual_revenue_PI(costs, carrier, year, n, rc, strike_price=strike_prices[(carrier, rc)], bus=bus) if np.isfinite(sp) else rev_mb
                elif scenario == 'cb':
                    sp = strike_prices[(carrier, rc)]
                    rev_sc = annual_revenue_CB(costs, carrier, year, n, rc, strike_price=strike_prices[(carrier, rc)], bus=bus) if np.isfinite(sp) else rev_mb
                wind_pay += rev_sc - rev_mb
        
        for carrier in ['solar', 'solar-hsat']:
            for rc in range(n.generators.index[n.generators.carrier == carrier].size):
                rev_mb = annual_revenue(costs, carrier, year, n, rc, bus=bus)
                rev_sc = 0
                if scenario == 'pb':
                    sp = strike_prices[(carrier, rc)]
                    rev_sc = annual_revenue_PB(costs, carrier, year, n, rc, strike_price=strike_prices[(carrier, rc)], bus=bus) if np.isfinite(sp) else rev_mb
                elif scenario == 'pi':
                    sp = strike_prices[(carrier, rc)]
                    rev_sc = annual_revenue_PI(costs, carrier, year, n, rc, strike_price=strike_prices[(carrier, rc)], bus=bus) if np.isfinite(sp) else rev_mb
                elif scenario == 'cb':
                    sp = strike_prices[(carrier, rc)]
                    rev_sc = annual_revenue_CB(costs, carrier, year, n, rc, strike_price=strike_prices[(carrier, rc)], bus=bus) if np.isfinite(sp) else rev_mb
                solar_pay += rev_sc - rev_mb
        
        wind_payments.append(wind_pay)
        solar_payments.append(solar_pay)

    if len(wind_payments) > 1:
        cov_wind_solar = np.cov(wind_payments, solar_payments, ddof=1)[0, 1]
    else:
        cov_wind_solar = np.nan

    cov_norm = cov_wind_solar / (np.std(wind_payments, ddof=1) * np.std(solar_payments, ddof=1)) if np.std(wind_payments, ddof=1) > 0 and np.std(solar_payments, ddof=1) > 0 else np.nan

    plt.figure(figsize=(10, 6))
    plt.scatter(
        wind_payments,
        solar_payments,
        color=cm_4,
        s=100,
        edgecolor='black',
        label=f'Annual pairs (Cov = {cov_norm:.2f})'
    )
    
    # Fit a linear regression line
    z = np.polyfit(wind_payments, solar_payments, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(wind_payments), max(wind_payments), 100)
    plt.plot(x_line, p(x_line), 'k--', linewidth=2, label=f'Linear fit: y={z[0]:.2f}x+{z[1]:.2e}')
    
    # Add thin lines at x=0 and y=0 to highlight quadrants
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.xlabel('Annual Wind CfD Payments [billion €]')
    plt.ylabel('Annual Solar CfD Payments [billion €]')
    scenario_name = scenario_name_mapping(scenario)
    plt.title(f'Wind vs. Solar State CfD Payments for {scenario_name}')
    plt.grid(True)
    plt.legend(loc='upper left')

    def fmt_thousands_dot(x):
        return f"{x/1e9:,.2f}"#.replace(",", ".")

    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt_thousands_dot(x)))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt_thousands_dot(x)))
    if save_to_file:
        results_directory = f'../results/{run_name}'
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/wind_vs_solar_cfd_payments_{scenario}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


# ___ CO2 Emissions Calculation ___

# Function to calculate annual CO2 emissions
def annual_CO2_emissions(costs, n, year):
    total_emissions = 0
    fossil_carriers = n.carriers.loc[n.carriers['co2_emissions'] > 0.0].index.tolist()
    for carrier in fossil_carriers:
        if carrier == 'OCGT' or carrier == 'CCGT':
            carkey = 'gas'
        else:
            carkey = carrier
        co2_intensity = costs.at[carkey, 'CO2 intensity']
        efficiency = costs.at[carrier, 'efficiency']
        if efficiency != efficiency or pd.isna(efficiency):  # Check for NaN
            efficiency = 1.0
        hourly_generation = get_hourly_generation(carrier, n)
        annual_generation = hourly_generation[hourly_generation.index.year == year]
        emissions = (annual_generation.values.flatten()).sum() * co2_intensity / efficiency / 1e6  # Convert to MtCO2
        total_emissions += emissions
        # print(f"Emissions from {carrier} added {emissions:.2f} MtCO2.")
    return total_emissions # Unit is MtCO2


# Function to calculate total CO2 emissions
def total_CO2_emissions(costs, n):
    total_emissions = 0
    for year in n.generators_t.p.index.year.unique():
        total_emissions += annual_CO2_emissions(costs, n, year)
    return total_emissions




# ____ Plots to compare statistics across scenarios ___
# System costs, average electricity prices, CO2 emissions, installed capacities, avg. state CfD payments and their std. dev.

# Compare average effective WACC (over all resource classes) for all carriers across scenarios
def plot_compare_average_effective_wacc_all_carriers(run_name='1_Iteration_One_v2', T=0, N=10000, save_to_file=True, sensitivity=False, carriers=['onwind', 'solar', 'offwind-dc'], scenarios=['mb', 'pc', 'pb', 'pi', 'cb'], aggregate=True):
    n = get_network(run_name)
    scenario_labels = {
        'mb': 'Direct Marketing',
        'pc': 'DM with Political Risk',
        'pb': 'Production-based CfD',
        'pi': 'Production-independent CfD',
        'cb': 'Capacity-based CfD',
    }
    scenario_colors = {
        'mb': cm_5,
        'pc': cm_6,
        'pb': cm_1,
        'pi': cm_2,
        'cb': cm_3,
    }
    carrier_groups = []
    seen_group_keys = set()
    for carrier in carriers:
        if aggregate and carrier in ['solar', 'solar-hsat']:
            group_key = 'solar_agg'
            group_label = 'Solar'
            group_members = ['solar', 'solar-hsat']
        elif aggregate and carrier in ['offwind', 'offwind-ac', 'offwind-dc', 'offwind-float']:
            group_key = 'offshore_wind_agg'
            group_label = 'Offshore Wind'
            group_members = ['offwind-ac', 'offwind-dc', 'offwind-float']
        else:
            group_key = carrier
            group_label = carrier_full_name(carrier)
            group_members = [carrier]

        if group_key not in seen_group_keys:
            carrier_groups.append((group_key, group_label, group_members))
            seen_group_keys.add(group_key)

    plt.figure(figsize=(12, 8))
    all_valid_waccs = []

    x = np.arange(len(carrier_groups))
    bar_width = 0.8 / max(1, len(scenarios))

    for i, scenario in enumerate(scenarios):
        avg_waccs = []
        for _, _, group_members in carrier_groups:
            weighted_wacc_sum = 0.0
            total_capacity = 0.0

            for carrier in group_members:
                n_rc_actual = n.generators.index[n.generators.carrier == carrier].size
                n_rc = n_rc_actual if n_rc_actual > 0 else 1

                for rc in range(n_rc):
                    gen_label = f"DE0 0 {rc} {carrier}"
                    if gen_label in n.generators.index and n.generators.p_nom_opt[gen_label] > 0:
                        wacc = wacc_updated(run_name, n, carrier, rc, T=T, N=N, scenario=scenario)
                        if np.isfinite(wacc):
                            cap = n.generators.p_nom_opt[gen_label]
                            weighted_wacc_sum += (wacc * 100) * cap
                            total_capacity += cap

            avg_wacc = weighted_wacc_sum / total_capacity if total_capacity > 0 else np.nan
            avg_waccs.append(avg_wacc)
            if np.isfinite(avg_wacc):
                all_valid_waccs.append(avg_wacc)

        offset = (i - (len(scenarios) - 1) / 2) * bar_width
        color = scenario_colors.get(scenario, 'gray')
        bars = plt.bar(
            x + offset,
            avg_waccs,
            width=bar_width,
            color=color,
            edgecolor='black',
            linewidth=1.2,
            label=scenario_labels.get(scenario, scenario),
        )
        for bar, value in zip(bars, avg_waccs):
            if np.isfinite(value):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f'{value:.2f}%',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                )

    plt.title('Average Effective WACC Across Carriers per Scenario')
    # plt.xlabel('Carrier')
    plt.xticks(x, [group_label for _, group_label, _ in carrier_groups])
    plt.ylabel('Average Effective WACC [%]')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
    if all_valid_waccs:
        plt.ylim(0.8 * np.min(all_valid_waccs), 1.3 * np.max(all_valid_waccs))
    plt.grid(True)
    plt.legend()#loc='upper right')

    if save_to_file:
        if sensitivity != False:
            output_dir = f'../results/my_plots/{sensitivity}'
        else:
            output_dir = '../results/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/average_effective_wacc_all_carriers_scenarios.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


# Compare all systemic indicators across scenarios in a single grouped bar chart
def plot_compare_systemic_indicators(run_names, scenarios=['mb', 'pc', 'pb', 'pi', 'cb'], scenario_labels=['Direct Marketing', 'DM with Political Risk', 'Production-based CfD', 'Production-independent CfD', 'Capacity-based CfD'], cap=130, bus="DE0 0", save_to_file=True, sensitivity=False):
    if len(run_names) != len(scenarios):
        raise ValueError(f"Length mismatch: got {len(run_names)} run_names but {len(scenarios)} scenarios.")
    if len(scenarios) != len(scenario_labels):
        raise ValueError(f"Length mismatch: got {len(scenarios)} scenarios but {len(scenario_labels)} scenario_labels.")

    indicator_labels = [
        'Avg Annual System Costs',
        'Avg Electricity Price',
        'Avg Annual CO2 Emissions',
        'Avg VRE Generation Share',
        # 'Average Absolute State CfD Payments',
    ]

    indicator_values = {label: [] for label in indicator_labels}

    for run_name, scenario in zip(run_names, scenarios):
        n = get_network(run_name)
        costs = get_costs(run_name)

        avg_system_costs = system_costs(n, bus=bus) / 35
        avg_price = average_price_per_MWh(n, bus=bus)
        avg_co2 = total_CO2_emissions(costs, n) / 35
        total_generation = n.generators_t.p.sum().sum()
        vre_mask = n.generators.carrier.isin(['offwind-ac', 'offwind-dc', 'offwind-float', 'onwind', 'solar', 'solar-hsat'])
        vre_generation = n.generators_t.p.loc[:, vre_mask].sum().sum()
        vre_gen_share = (vre_generation / total_generation) * 100 if total_generation > 0 else np.nan
        # avg_abs_state_payments = average_abs_state_cfd_payments(run_name, n, scenario=scenario, cap=cap, bus=bus)

        indicator_values['Avg Annual System Costs'].append(avg_system_costs)
        indicator_values['Avg Electricity Price'].append(avg_price)
        indicator_values['Avg Annual CO2 Emissions'].append(avg_co2)
        indicator_values['Avg VRE Generation Share'].append(vre_gen_share)
        # indicator_values['Average Absolute State CfD Payments'].append(avg_abs_state_payments)

    # Convert heterogeneous units to a common index for one-axis comparison.
    # Baseline is Direct Marketing ('mb') if present, otherwise the first scenario.
    baseline_idx = scenarios.index('mb') if 'mb' in scenarios else 0
    indexed_values = {}
    for label in indicator_labels:
        vals = np.array(indicator_values[label], dtype=float)
        baseline = vals[baseline_idx]
        if np.isfinite(baseline) and baseline != 0:
            indexed = vals / baseline * 100
        else:
            max_abs = np.nanmax(np.abs(vals))
            indexed = vals / max_abs * 100 if np.isfinite(max_abs) and max_abs > 0 else np.full_like(vals, np.nan)
        indexed_values[label] = indexed

    scenario_colors = {
        'mb': cm_5,
        'pc': cm_6,
        'pb': cm_1,
        'pi': cm_2,
        'cb': cm_3,
    }

    x = np.arange(len(indicator_labels))
    bar_width = 0.8 / max(1, len(scenarios))
    plt.figure(figsize=(14, 8))

    for i, scenario in enumerate(scenarios):
        scenario_series = [indexed_values[label][i] for label in indicator_labels]
        offset = (i - (len(scenarios) - 1) / 2) * bar_width
        plt.bar(
            x + offset,
            scenario_series,
            width=bar_width,
            color=scenario_colors.get(scenario, 'gray'),
            edgecolor='black',
            linewidth=1.0,
            label=scenario_labels[i],
        )

    plt.title('Comparison of Systemic Indicators Across Scenarios')
    plt.xticks(x, indicator_labels)#, rotation=20, ha='right')
    plt.ylabel('Indicator Value [Index, 100 = Direct Marketing]')
    plt.ylim(95, 105)
    plt.yticks(np.arange(96, 105, 2))
    # ax = plt.gca()
    # ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}\%'))
    plt.axhline(100, color='black', linestyle='--', linewidth=1.0)
    plt.grid(True, which='both', axis='y', alpha=0.75)
    plt.legend()

    if save_to_file:
        if sensitivity != False:
            output_dir = f'../results/my_plots/{sensitivity}'
        else:
            output_dir = '../results/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/compare_systemic_indicators.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


# Compare total system costs across scenarios in a bar chart
def plot_compare_annualized_system_costs(run_names, scenario_labels =['Direct Marketing', 'DM with Political Risk', 'Production-based CfD', 'Production-independent CfD', 'Capacity-based CfD'], bus="DE0 0", save_to_file=True, sensitivity=False):
    costs_list = []
    for run_name in run_names:
        n = get_network(run_name)
        sc = system_costs(n, bus=bus)/35
        costs_list.append(sc)

    width = 0.35

    plt.figure(figsize=(10, 6))
    bars = plt.bar(scenario_labels, costs_list, width, color=cm_1, edgecolor='black')
    plt.xticks(fontsize=8)
    
    def fmt_billions_dot(x, n=0):
        return f"{x/1e9:,.{n}f}".replace(",", ".")

    plt.ylabel('Average Annual System Costs [Billion €]')
    plt.title('Comparison of Average Annual System Costs Across Scenarios')
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt_billions_dot(x, 0)))
    ax.yaxis.offsetText.set_visible(False)
    plt.grid(axis='y', alpha=0.75)
    plt.ylim(0.8 * min(costs_list), 1.1 * max(costs_list))

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height * 1.01,
             fmt_billions_dot(height, 2) + " B€", ha='center', va='bottom', fontsize=11, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=0.5))

    if save_to_file:
        if sensitivity != False:
            output_dir = f'../results/my_plots/{sensitivity}'
        else:
            output_dir = '../results/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/compare_system_costs.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


# Compare average electricity prices across scenarios in a bar chart
def plot_compare_avg_prices(run_names, scenario_labels=['Direct Marketing', 'DM with Political Risk', 'Production-based CfD', 'Production-independent CfD', 'Capacity-based CfD'], bus="DE0 0", save_to_file=True, sensitivity=False):
    avg_prices = []
    for run_name in run_names:
        n = get_network(run_name)
        avg_price = average_price_per_MWh(n, bus=bus)
        avg_prices.append(avg_price)

    width = 0.35

    plt.figure(figsize=(10, 6))
    bars = plt.bar(scenario_labels, avg_prices, width, color=cm_2, edgecolor='black')
    plt.xticks(fontsize=8)
    
    def fmt_euros_dot(x, n):
        return f"{x:,.{n}f}".replace(",", ".")

    plt.ylabel('Average Electricity Price [€/MWh]')
    plt.title('Comparison of Average Electricity Prices Across Scenarios')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt_euros_dot(x, 0)))
    plt.grid(axis='y', alpha=0.75)
    plt.ylim(0.8 * min(avg_prices), 1.1 * max(avg_prices))

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height * 1.01,
                 fmt_euros_dot(height, 2) + " €/MWh", ha='center', va='bottom', fontsize=11, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=0.5))

    if save_to_file:
        if sensitivity != False:
            output_dir = f'../results/my_plots/{sensitivity}'
        else:
            output_dir = '../results/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/compare_avg_prices.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


# Compare total CO2 emissions across scenarios in a bar chart
def plot_compare_CO2_emissions(run_names, scenario_labels=['Direct Marketing', 'DM with Political Risk', 'Production-based CfD', 'Production-independent CfD', 'Capacity-based CfD'], bus="DE0 0", save_to_file=True, sensitivity=False):
    emissions_list = []
    for run_name in run_names:
        n = get_network(run_name)
        costs = get_costs(run_name)
        total_emissions = total_CO2_emissions(costs, n)/35
        emissions_list.append(total_emissions)

    width = 0.35

    plt.figure(figsize=(10, 6))
    bars = plt.bar(scenario_labels, emissions_list, width, color=cm_5, edgecolor='black')
    plt.xticks(fontsize=8)
    
    def fmt_millions_dot(x, n):
        return f"{x:,.{n}f}".replace(",", ".")

    plt.ylabel('Average Annual CO₂ Emissions [MtCO₂]')
    plt.title('Comparison of Average Annual CO₂ Emissions Across Scenarios')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt_millions_dot(x, 0)))
    plt.grid(axis='y', alpha=0.75)
    plt.ylim(0.8 * min(emissions_list), 1.1 * max(emissions_list))

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height * 1.01,
                 fmt_millions_dot(height, 2) + " Mt CO₂", ha='center', va='bottom', fontsize=11, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=0.5))

    if save_to_file:
        if sensitivity != False:
            output_dir = f'../results/my_plots/{sensitivity}'
        else:
            output_dir = '../results/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/compare_CO2_emissions.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


# Compare aggregated installed VRE, gas, as well as battery capacities across scenarios in a bar chart
def plot_compare_capacities(run_names, scenario_labels=['Direct Marketing', 'DM with Political Risk', 'Production-based CfD', 'Production-independent CfD', 'Capacity-based CfD'], bus="DE0 0", save_to_file=True, sensitivity=False):
    vre_capacities = []
    # gas_capacities = []
    # bat_capacities = []
    for run_name in run_names:
        n = get_network(run_name)
        vre_capacity = n.generators.p_nom_opt[n.generators.carrier.isin(['offwind-ac', 'offwind-dc', 'offwind-float', 'onwind', 'solar', 'solar-hsat'])].sum()
        # gas_capacity = n.generators.p_nom_opt[n.generators.carrier.isin(['OCGT', 'CCGT'])].sum()
        # bat_capacity = n.stores.e_nom_opt[n.stores.carrier == 'battery'].sum()
        vre_capacities.append(vre_capacity)
        # gas_capacities.append(gas_capacity)
        # bat_capacities.append(bat_capacity)


    x = np.arange(len(scenario_labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    bars_vre = plt.bar(x, vre_capacities, width, label='VRE Capacity', color=cm_3, edgecolor='black')
    # bars_vre = plt.bar(x - width/2, vre_capacities, width, label='VRE Capacity', color=cm_3, edgecolor='black')
    # bars_gas = plt.bar(x + width/2, gas_capacities, width, label='Gas Capacity', color=cm_2, edgecolor='black')
    # bars_bat = plt.bar(x, bat_capacities, width, label='Battery Capacity', color=cm_4, edgecolor='black')
    plt.xticks(fontsize=8)

    def fmt_gigawatts_dot(x, n):
        return f"{x/1e3:,.{n}f}".replace(",", ".")

    plt.ylabel('Installed Capacity [GW]')
    plt.title('Comparison of Installed VRE Capacities Across Scenarios')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt_gigawatts_dot(x, 0)))
    plt.xticks(x, scenario_labels)
    plt.grid(axis='y', alpha=0.75)
    # plt.legend()
    plt.ylim(0.8 * min(vre_capacities), 1.1 * max(vre_capacities))

    # # Add value labels on top of bars
    for bar in bars_vre:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height * 1.01,
                 fmt_gigawatts_dot(height, 2) + " GW", ha='center', va='bottom', fontsize=11, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=0.5))
    # for bar in bars_gas:
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width() / 2.0, height * 1.01,
    #              fmt_gigawatts_dot(height, 2) + " GW", ha='center', va='bottom', fontsize=11, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=0.5))
    # for bar in bars_bat:
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width() / 2.0, height * 1.01,
    #              fmt_gigawatts_dot(height, 2) + " GW", ha='center', va='bottom', fontsize=11, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=0.5))
    if save_to_file:
        if sensitivity != False:
            output_dir = f'../results/my_plots/{sensitivity}'
        else:
            output_dir = '../results/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/compare_capacities.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')


# Compare VRE generation share across scenarios in a bar chart -> stack individual VRE carrier shares 
def plot_compare_vre_generation_share(run_names, scenario_labels=['Direct Marketing', 'DM with Political Risk', 'Production-based CfD', 'Production-independent CfD', 'Capacity-based CfD'], bus="DE0 0", save_to_file=True, sensitivity=False):
    vre_shares = []
    for run_name in run_names:
        n = get_network(run_name)
        total_generation = n.generators_t.p.sum().sum()
        vre_mask = n.generators.carrier.isin(['offwind-ac', 'offwind-dc', 'offwind-float', 'onwind', 'solar', 'solar-hsat'])
        vre_generation = n.generators_t.p.loc[:, vre_mask].sum().sum()
        share = (vre_generation / total_generation) * 100 if total_generation > 0 else np.nan
        vre_shares.append(share)

    width = 0.35

    plt.figure(figsize=(10, 6))
    bars = plt.bar(scenario_labels, vre_shares, width, color=cm_3, edgecolor='black')
    plt.xticks(fontsize=8)

    def fmt_percent_dot(x, n):
        return f"{x:.{n}f}%".replace(",", ".")

    plt.ylabel('VRE Generation Share [%]')
    plt.title('Comparison of VRE Generation Share Across Scenarios')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt_percent_dot(x, 0)))
    plt.grid(axis='y', alpha=0.75)
    plt.ylim(0.8 * min(vre_shares), 1.1 * max(vre_shares))

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height * 1.01,
                 fmt_percent_dot(height, 2), ha='center', va='bottom', fontsize=11, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=0.5))

    if save_to_file:
        if sensitivity != False:
            output_dir = f'../results/my_plots/{sensitivity}'
        else:
            output_dir = '../results/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/compare_vre_generation_share.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


# Compare std dev of state CfD payments across scenarios in a bar chart
def plot_compare_std_dev_state_cfd_payments(run_names, scenario_labels=['Production-based CfD', 'Production-independent CfD', 'Capacity-based CfD'], scenarios=['pb', 'pi', 'cb'], bus="DE0 0", save_to_file=True, sensitivity=False):
    std_devs = []
    for i in range(len(run_names)):
        n = get_network(run_names[i])
        std_dev = std_dev_state_cfd_payments(run_names[i], n, scenario=scenarios[i], bus=bus)
        std_devs.append(std_dev)

    width = 0.35

    plt.figure(figsize=(10, 6))
    bars = plt.bar(scenario_labels, std_devs, width, color=cm_2, edgecolor='black')
    plt.xticks(fontsize=8)
    
    def fmt_billions_dot(x, n=0):
        return f"{x/1e9:,.{n}f}".replace(",", ".")

    plt.ylabel('Standard Deviation of Annual State CfD Payments [Billion €]')
    plt.title('Comparison of Standard Deviation of Annual State CfD Payments Across Scenarios')
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt_billions_dot(x, 2)))
    ax.yaxis.offsetText.set_visible(False)
    plt.grid(axis='y', alpha=0.75)
    plt.ylim(0.8 * min(std_devs), 1.1 * max(std_devs))

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height * 1.01,
             fmt_billions_dot(height, 2) + " B€", ha='center', va='bottom', fontsize=11, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=0.5))

    if save_to_file:
        if sensitivity != False:
            output_dir = f'../results/my_plots/{sensitivity}'
        else:
            output_dir = '../results/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/compare_std_dev_state_cfd_payments.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


# Plot average annual electricity price for all CfD types as well as no CfD across all years
# This has to be adapted to be able to run sensitivity runs
def plot_avg_price_with_levy(run_names, scenarios=['mb', 'pc', 'pb', 'pi', 'cb'], cap=130, bus="DE0 0", save_to_file=True, sensitivity=False):
    if isinstance(run_names, str):
        run_names = [run_names] * len(scenarios)
    elif len(run_names) != len(scenarios):
        raise ValueError("run_names must be a string or a list with the same length as scenarios")

    n = get_network(run_names[0])
    plt.figure(figsize=(10, 6))
    scenario_colors = {
        'mb': cm_5,
        'pc': cm_6,
        'pb': cm_1,
        'pi': cm_2,
        'cb': cm_3,
    }
    scenario_names = {
        'mb': 'Direct Marketing',
        'pc': 'DM with Political Risk',
        'pb': 'Production-Based CfD',
        'pi': 'Production-Independent CfD',
        'cb': 'Capacity-Based CfD'
    }

    years = np.array(sorted(n.generators_t.p.index.year.unique()))
    n_years = len(years)
    q_size = n_years // 5

    low_idx = list(range(0, q_size))
    avg_idx = list(range(q_size, 4 * q_size))
    high_idx = list(range(4 * q_size, n_years))
    mb_section_means = None
    total_avg_price = None

    low_means = {}
    avg_means = {}
    high_means = {}
    for run, scenario, scenario_label in zip(run_names, scenarios, scenario_names.values()):
        n_run = get_network(run)
        if scenario == 'pc':
            price_series = price_time_series_with_levy(run, n_run, scenario='mb', cap=cap, bus=bus)
            # Necessary to use 'mb' price series from 'pc' scenario because it runs with higher WACC assumptions factoring in political risk
        else:
            price_series = price_time_series_with_levy(run, n_run, scenario=scenario, cap=cap, bus=bus)
        scenario_name = scenario_names.get(scenario, scenario_name_mapping(scenario))
        price_mean = np.nanmean(price_series.values)
        price_std = np.nanstd(price_series.values)
        label_text = scenario_label if scenario_label else scenario_name
        legend_label = f"{label_text} ($\\overline{{p_t}}={price_mean:.1f}\\,\\mathrm{{€/MWh}}$, $\\sigma={price_std:.1f}\\,\\mathrm{{€/MWh}}$)"
        color = scenario_colors.get(scenario, 'blue')
        plt.plot(price_series.index, price_series.values, marker='o', label=legend_label, color=color, markeredgecolor='black')

        total_avg_price = price_mean
        # Mark average prices in low/average/high gas-price year sectors.
        if q_size > 0 and len(high_idx) > 0:
            low_mean = np.nanmean(price_series.values[low_idx])
            low_means[scenario] = low_mean
            avg_mean = np.nanmean(price_series.values[avg_idx])
            avg_means[scenario] = avg_mean
            high_mean = np.nanmean(price_series.values[high_idx])
            high_means[scenario] = high_mean

        if scenario == 'mb':
            plt.ylim(0.8 * min(price_series.values), 1.1 * max(price_series.values))
            plt.plot([years[low_idx[0]], years[low_idx[-1]]], [low_mean, low_mean], color='black', linestyle='--', linewidth=1.8)
            plt.plot([years[avg_idx[0]], years[avg_idx[-1]]], [avg_mean, avg_mean], color='black', linestyle='--', linewidth=1.8)
            plt.plot([years[high_idx[0]], years[high_idx[-1]]], [high_mean, high_mean], color='black', linestyle='--', linewidth=1.8)
        elif total_avg_price is None:
            total_avg_price = np.nanmean(price_series.values)
    mb_section_means = (low_means, avg_means, high_means)        
    
    if q_size > 0 and len(high_idx) > 0:
        x_sep_1 = (years[low_idx[-1]] + years[avg_idx[0]]) / 2
        x_sep_2 = (years[avg_idx[-1]] + years[high_idx[0]]) / 2
        plt.axvline(x=x_sep_1, color='black', linestyle='-', linewidth=1)
        plt.axvline(x=x_sep_2, color='black', linestyle='-', linewidth=1)

        ymin, ymax = plt.ylim()
        if mb_section_means is not None:
            low_mean, avg_mean, high_mean = mb_section_means
            low_label = f"Low Gas Price Years\nDM: $\\overline{{p_t}}={low_means['mb']:.1f}\\,\\mathrm{{€/MWh}}$\nPB: $\\overline{{p_t}}={low_means['pb']:.1f}\\,\\mathrm{{€/MWh}}$"
            avg_label = f"Average Gas Price Years\nDM: $\\overline{{p_t}}={avg_means['mb']:.1f}\\,\\mathrm{{€/MWh}}$\nPB: $\\overline{{p_t}}={avg_means['pb']:.1f}\\,\\mathrm{{€/MWh}}$"
            high_label = f"High Gas Price Years\nDM: $\\overline{{p_t}}={high_means['mb']:.1f}\\,\\mathrm{{€/MWh}}$\nPB: $\\overline{{p_t}}={high_means['pb']:.1f}\\,\\mathrm{{€/MWh}}$"
        else:
            low_label = 'Low Gas Price Years'
            avg_label = 'Average Gas Price Years'
            high_label = 'High Gas Price Years'

        plt.text(np.mean(years[low_idx]), ymin*1.15, low_label, ha='center', va='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=0.5))
        plt.text(np.mean(years[avg_idx]), ymin*1.15, avg_label, ha='center', va='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=0.5))
        plt.text(np.mean(years[high_idx]), ymin*1.15, high_label, ha='center', va='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=0.5))

    # if total_avg_price is not None:
    #     _, ymax = plt.ylim()
    #     total_label = f"Total Average Price: $\\overline{{p_t}}={total_avg_price:.1f}\\,\\mathrm{{€/MWh}}$"
    #     plt.text(np.max(years), ymax * 1.0, total_label, ha='right', va='top', fontsize=11,
    #              bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=0.7))

    plt.xlabel('Weather Year')
    plt.ylabel('Average Annual Electricity Price with Levy [€/MWh]')
    plt.title('Average Annual Electricity Price with Levy Across Scenarios')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.ylim(0.25 * total_avg_price, 2.25 * total_avg_price)
    if save_to_file:
        if sensitivity != False:
            output_dir = f'../results/my_plots/{sensitivity}'
        else:
            output_dir = '../results/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/avg_price_with_levy.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


# Compare average state CfD payments across scenarios in a bar chart
# This plot is somewhat unnecessary as those are approximately zero for given strike price assumptions
def plot_compare_avg_state_cfd_payments(run_names, scenario_labels=['Production-based CfD', 'Production-independent CfD', 'Capacity-based CfD'], scenarios=['pb', 'pi', 'cb'], bus="DE0 0", save_to_file=True, sensitivity=False):
    avg_payments = []
    for i in range(len(run_names)):
        n = get_network(run_names[i])
        avg_payment = average_state_cfd_payments(run_names[i], n, scenario=scenarios[i], bus=bus)
        avg_payments.append(avg_payment)

    width = 0.35
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(scenario_labels, avg_payments, width, color=cm_1, edgecolor='black')
    plt.xticks(fontsize=8)
    
    def fmt_billions_dot(x, n=0):
        return f"{x/1e9:,.{n}f}".replace(",", ".")

    plt.ylabel('Average Annual State CfD Payments [Billion €]')
    plt.title('Comparison of Average Annual State CfD Payments Across Scenarios')
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt_billions_dot(x, 0)))
    ax.yaxis.offsetText.set_visible(False)
    plt.grid(axis='y', alpha=0.75)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height * 1.01,
             fmt_billions_dot(height, 2) + " B€", ha='center', va='bottom', fontsize=11, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=0.5))

    if save_to_file:
        if sensitivity != False:
            output_dir = f'../results/my_plots/{sensitivity}'
        else:
            output_dir = '../results/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/compare_avg_state_cfd_payments.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()