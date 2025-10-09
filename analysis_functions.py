#%%

import pypsa
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import lognorm
import seaborn as sns
import pandas as pd

run_name = "2030_without_CfD_v5" # Replace with correct run name
results_directory = "results/{}".format(run_name)
n = pypsa.Network(f"{results_directory}/networks/base_s_1_EP__2030.nc")  # Replace with correct network name
scenario = 'cap' #'no_cfd'  # Replace with correct scenario name if needed


# Color theme
mcgreen = (0.8, 0.898, 0.8) 
mcorange = (1.0, 0.9294, 0.8)
mcblue = (0.8, 0.8, 1.0)
mcred = (1.0, 0.8, 0.8)


#%% Function definitions

# Function to get hourly generation of any carrier
def get_hourly_generation(carrier):
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
def get_hourly_generation_per_MW(carrier, n=n):
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


# Function to get CO2 intensity of all carriers
def CO2_intensity_carriers():
    n_temp = pypsa.Network("resources/2030_without_CfD_v5/networks/base_s_1_elec_EP.nc")
    co2_intensity = n_temp.carriers.co2_emissions
    return co2_intensity


# Function to calculate annual CO2 emissions
def annual_CO2_emissions(year):
    co2_intensity = CO2_intensity_carriers()
    total_emissions = 0
    for carrier in co2_intensity.index:
        if carrier in n.generators.carrier.unique():
            hourly_generation = get_hourly_generation(carrier)
            annual_generation = hourly_generation[hourly_generation.index.year == year]
            emissions = (annual_generation.values.flatten() * co2_intensity[carrier]).sum() / 1e6  # Convert to MtCO2
            total_emissions += emissions
        else:
            print(f"Carrier {carrier} not found in the network generators.")
    return total_emissions

# co2_intensity = CO2_intensity_carriers()
# print(co2_intensity.index)
# print(n.generators.carrier)

# for carrier in co2_intensity.index:
#         if carrier in n.generators.carrier.unique():
#             print(f"Carrier {carrier} found in the network generators.")

# Function to calculate total CO2 emissions
def total_CO2_emissions():
    co2_intensity = CO2_intensity_carriers()
    total_emissions = 0
    for carrier in co2_intensity.index:
        if carrier in n.generators.carrier.unique():
            hourly_generation = get_hourly_generation(carrier)
            emissions = (hourly_generation.values.flatten() * co2_intensity[carrier]).sum() / 1e6  # Convert to MtCO2
            total_emissions += emissions
    return total_emissions


# Get hourly marginal price at each node for any carrier
def get_hourly_marginal_price(bus="DE0 0", n=n):
    price_t = n.buses_t.marginal_price[bus]
    return price_t


# Histogram of marginal prices at node 'DE0 0'
def plot_price_histogram(bus="DE0 0", bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, np.inf]):
    price_t = get_hourly_marginal_price(bus)
    plt.figure(figsize=(10, 6))
    plt.hist(price_t, bins=bins, color=mcblue, alpha=0.7)
    plt.title(f'Histogram of Marginal Prices at {bus}')
    plt.xlabel('Marginal Price [€/MWh]')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


# Statistical quantities of marginal prices at node 'DE0 0'
def price_statistics(bus="DE0 0"):
    price_t = get_hourly_marginal_price(bus)
    stats = {
        "mean": price_t.mean(),
        "median": price_t.median(),
        "std_dev": price_t.std(),
        "min": price_t.min(),
        "max": price_t.max()
    }
    return stats


# Number of hours with prices above 5000 €/MWh at node 'DE0 0'
def hours_above_threshold(bus="DE0 0", threshold=5000):
    price_t = get_hourly_marginal_price(bus)
    hours = (price_t > threshold).sum()
    return hours


# Values in those hours
def prices_above_threshold(bus="DE0 0", threshold=5000):
    price_t = get_hourly_marginal_price(bus)
    prices = price_t[price_t > threshold]
    return prices


# Set values in those hours to threshold value
def cap_prices_above_threshold(bus="DE0 0", threshold=5000):
    price_t = get_hourly_marginal_price(bus)
    price_t[price_t > threshold] = threshold
    return price_t


# Price duration curve at node 'DE0 0'
def plot_price_duration_curve(bus="DE0 0", save_to_file=True):
    price_t = get_hourly_marginal_price(bus)
    sorted_prices = np.sort(price_t)[::-1]
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_prices, color=mcblue, linewidth=2) 
    # plt.title(f'Price Duration Curve at {bus}')
    plt.xlabel('SortedHours')
    plt.ylabel('Marginal Price [€/MWh]')
    plt.ylim(0, 100)  # Cap y-axis at 100 €/MWh
    plt.grid(True)
    if save_to_file:
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/price_duration_curve.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


# General statistics
def statistics(bus="DE0 0", n=n):
    # Define boolean masks for carriers
    n=n
    vre_i = n.generators.carrier.isin(['onwind', 'offwind-ac', 'offwind-dc', 'offwind-float', 'solar', 'solar rooftop'])
    onwind_i = n.generators.carrier == 'onwind'
    offwind_i = n.generators.carrier.isin(['offwind-ac', 'offwind-dc', 'offwind-float'])
    solar_i = n.generators.carrier.isin(['solar', 'solar rooftop'])

    stats = {
        "Average Price": price_statistics(bus)["mean"],
        "Price Std Dev": price_statistics(bus)["std_dev"],
        "Generation Total": n.generators_t.p.sum().sum(),
        "Generation VRE": n.generators_t.p.loc[:, vre_i].sum().sum(),
        "Generation Onshore Wind": n.generators_t.p.loc[:, onwind_i].sum().sum(),
        "Generation Offshore Wind": n.generators_t.p.loc[:, offwind_i].sum().sum(),
        "Generation Solar": n.generators_t.p.loc[:, solar_i].sum().sum(),
        "Generation Gas": n.generators_t.p.loc[:, n.generators.carrier == 'gas'].sum().sum(),
        "Generation Coal": n.generators_t.p.loc[:, n.generators.carrier == 'coal'].sum().sum(),
        "Generation Oil": n.generators_t.p.loc[:, n.generators.carrier == 'oil primary'].sum().sum(),
        "VRE Generation Share": n.generators_t.p.loc[:, vre_i].sum().sum() / n.generators_t.p.sum().sum(),
        "Capacity Total": n.generators.p_nom_opt.sum(),
        "Capacity Onshore Wind": n.generators.p_nom_opt[onwind_i].sum(),
        "Capacity Offshore Wind": n.generators.p_nom_opt[offwind_i].sum(),
        "Capacity Solar": n.generators.p_nom_opt[solar_i].sum(),
        "Capacity Hydro": n.generators.p_nom_opt[n.generators.carrier == 'ror'].sum(),
        "Capacity Bio": n.generators.p_nom_opt[n.generators.carrier.isin(['biogas', 'solid biomass', 'unsustainable biogas', 'unsustainable bioliquids', 'unsustainable solid biomass'])].sum(),
        "Capacity Gas": n.generators.p_nom_opt[n.generators.carrier == 'gas'].sum(),
        "Capacity Coal": n.generators.p_nom_opt[n.generators.carrier == 'coal'].sum(),
        "Capacity Oil": n.generators.p_nom_opt[n.generators.carrier == 'oil primary'].sum(),
        "VRE Capacity Share": n.generators.p_nom_opt[vre_i].sum() / n.generators.p_nom_opt.sum(),
        "CO2 Emissions": total_CO2_emissions()
    }
    return stats


# Plot comparison of key statistics for different scenarios
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
    bars = plt.bar(['No CfD', 'With CfD', 'With CM'], [stat_nocfd, stat_cfd, stat_cap], color=[mcred, mcgreen, mcorange])
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


# Annual revenue per MW of a carrier for a given year
def annual_revenue_per_MW(carrier, year, bus="DE0 0", n=n):
    hourly_generation_per_MW = get_hourly_generation_per_MW(carrier, n=n)
    hourly_generation_year_per_MW = hourly_generation_per_MW[hourly_generation_per_MW.index.year == year]
    price = get_hourly_marginal_price(bus, n=n)
    price_year = price[price.index.year == year]
    rev_per_MW_year = (hourly_generation_year_per_MW.values.flatten() * price_year.values.flatten()).sum()
    return rev_per_MW_year


# Annual Revenue Statistics for a given carrier across all years
def annual_revenue_statistics(carrier, bus="DE0 0"):
    years = n.generators_t.p.index.year.unique()
    revenues = [annual_revenue_per_MW(carrier, year, bus) for year in years]
    stats = {
        "mean": np.mean(revenues),
        "median": np.median(revenues),
        "std_dev": np.std(revenues),
        "risk": np.std(revenues) / np.mean(revenues) if np.mean(revenues) != 0 else np.nan,
        "min": np.min(revenues),
        "max": np.max(revenues)
    }
    return stats 


# Plot annual revenue of a carrier for all years
def plot_annual_revenue_per_MW(carrier, bus="DE0 0", save_to_file=True):
    years = n.generators_t.p.index.year.unique()
    revenues = [annual_revenue_per_MW(carrier, year, bus) for year in years]
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
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/annual_revenue_per_MW_{carrier}_{scenario}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


# Violin plot for annual revenues of different carriers
def plot_revenue_violin(carrier, bus="DE0 0", save_to_file=True):
    years = n.generators_t.p.index.year.unique()
    all_revenues = []
    carrier_names_map = {
        'offwind': "Offshore Wind",
        'onwind': "Onshore Wind",
        'solar': "Ground-Mounted Solar PV",
        'solar rooftop': "Rooftop Solar PV"
    }
    carrier_name = carrier_names_map.get(carrier, carrier)

    rev_no_cfd = [annual_revenue_per_MW(carrier, year, bus) for year in years]
    rev_cfd = [annual_revenue_per_MW_with_CfD(carrier, year, bus) for year in years]
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
        "Rooftop Solar PV": mcred
    }

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
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        # The filename should probably be more specific to the carrier
        filename = f'{output_dir}/revenue_violin_plot_{carrier}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()



# Annual revenue per MW for a given carrier and year under a standard financial CfD 
# (using the average market profile as the reference profile - equivalent to a conventional CfD in this case)
def annual_revenue_per_MW_with_CfD(carrier, year, use_avg_rev_per_MWh=True, strike_price=0, bus="DE0 0"):
    if carrier == 'offwind':
        # For 'offwind-ac'
        hg_ac_per_mw = get_hourly_generation_per_MW('offwind-ac')
        hg_ac_year_per_mw = hg_ac_per_mw[hg_ac_per_mw.index.year == year]
        if use_avg_rev_per_MWh:
            stats_ac = annual_revenue_statistics('offwind-ac', bus)
            hg_per_mw_ac = get_hourly_generation_per_MW('offwind-ac')
            avg_annual_gen_per_MW_ac = hg_per_mw_ac.sum().values[0] / 9  # Total generation per MW divided by number of years
            strike_price_ac = stats_ac["mean"] / avg_annual_gen_per_MW_ac  # Average revenue per MWh
            # print(f"Using average revenue per MWh as AC strike price: {strike_price_ac:.2f} €/MWh")
        rev_year_per_MW_with_CfD_ac = (strike_price_ac * hg_ac_year_per_mw.values.flatten()).sum()
        # For 'offwind-dc'
        hg_dc_per_mw = get_hourly_generation_per_MW('offwind-dc')
        hg_dc_year_per_mw = hg_dc_per_mw[hg_dc_per_mw.index.year == year]
        if use_avg_rev_per_MWh:
            stats_dc = annual_revenue_statistics('offwind-dc', bus)
            hg_per_mw_dc = get_hourly_generation_per_MW('offwind-dc')
            avg_annual_gen_per_MW_dc = hg_per_mw_dc.sum().values[0] / 9  # Total generation per MW divided by number of years
            strike_price_dc = stats_dc["mean"] / avg_annual_gen_per_MW_dc  # Average revenue per MWh
            # print(f"Using average revenue per MWh as DC strike price: {strike_price_dc:.2f} €/MWh")
        rev_year_per_MW_with_CfD_dc = (strike_price_dc * hg_dc_year_per_mw.values.flatten()).sum()
        # For 'offwind-float'
        hg_float_per_mw = get_hourly_generation_per_MW('offwind-float')
        hg_float_year_per_mw = hg_float_per_mw[hg_float_per_mw.index.year == year]
        if use_avg_rev_per_MWh:
            stats_float = annual_revenue_statistics('offwind-float', bus)
            hg_per_mw_float = get_hourly_generation_per_MW('offwind-float')
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
        hourly_generation_per_MW = get_hourly_generation_per_MW(carrier)
        hourly_generation_year_per_MW = hourly_generation_per_MW[hourly_generation_per_MW.index.year == year]
        if use_avg_rev_per_MWh:
            stats = annual_revenue_statistics(carrier, bus)
            hg_per_mw = get_hourly_generation_per_MW(carrier)
            avg_annual_gen_per_MW = hg_per_mw.sum().values[0] / 9  # Total generation per MW divided by number of years
            strike_price = stats["mean"] / avg_annual_gen_per_MW  # Average revenue per MWh
            # print(f"Using average revenue per MWh as strike price: {strike_price:.2f} €/MWh")
        # Revenue calculation with CfD
        rev_year_per_MW_with_CfD = (strike_price * hourly_generation_year_per_MW.values.flatten()).sum()
    return rev_year_per_MW_with_CfD


# Annual Revenue Statistics with CfD for a given carrier across all years
def annual_revenue_statistics_with_CfD(carrier, use_avg_rev_per_MWh=True, strike_price=0, bus="DE0 0"):
    years = n.generators_t.p.index.year.unique()
    revenues = [annual_revenue_per_MW_with_CfD(carrier, year, use_avg_rev_per_MWh, strike_price, bus) for year in years]
    stats = {
        "mean": np.mean(revenues),
        "median": np.median(revenues),
        "std_dev": np.std(revenues),
        "risk": np.std(revenues) / np.mean(revenues) if np.mean(revenues) != 0 else np.nan,
        "min": np.min(revenues),
        "max": np.max(revenues)
    }
    return stats


# Plot annual revenue of a carrier for all years under a standard financial CfD
def plot_annual_revenue_per_MW_with_CfD(carrier, use_avg_rev_per_MWh=True, strike_price=0, bus="DE0 0", save_to_file=True): 
    years = n.generators_t.p.index.year.unique()
    revenues = [annual_revenue_per_MW_with_CfD(carrier, year, use_avg_rev_per_MWh, strike_price, bus) for year in years]
    rev_0 = [annual_revenue_per_MW(carrier, year, bus) for year in years]
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
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/annual_revenue_per_MW_with_CfD_{carrier}_{scenario}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()



# Plot histogram of normalized distribution of annual revenues of all carriers around their average
def plot_normalized_annual_revenue_distribution(carriers=['offwind', 'onwind', 'solar', 'solar rooftop'], bus="DE0 0", save_to_file=True):
    years = n.generators_t.p.index.year.unique()
    plt.figure(figsize=(10, 6))
    normalized_revenues = []
    for carrier in carriers:
        revenues = [annual_revenue_per_MW(carrier, year, bus) for year in years]
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
    plt.title('Normalized Distribution of Annual Revenues per MW installed capacity')
    plt.xlabel('Normalized Annual Revenue (Revenue / Average Revenue)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    if save_to_file:    
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/normalized_annual_revenue_distribution.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


# Table of revenue statistics with and without CfD for a given carrier
def revenue_statistics_comparison(carrier, use_avg_rev_per_MWh=True, strike_price=0, bus="DE0 0"):
    stats_without_CfD = annual_revenue_statistics(carrier, bus)
    stats_with_CfD = annual_revenue_statistics_with_CfD(carrier, use_avg_rev_per_MWh, strike_price, bus)
    print(f"Revenue Statistics for {carrier}:")
    print(f"{'Statistic':<10} {'Without CfD (€)':<20} {'With CfD (€)':<20}")
    for key in stats_without_CfD.keys():
        print(f"{key:<10} {stats_without_CfD[key]:<20.5f} {stats_with_CfD[key]:<20.5f}")
    return stats_without_CfD, stats_with_CfD


# Updated discount rate (i.e. WACC) assuming (simplified!) linear relation between WACC and revenue variability
def updated_discount_rate(carrier, bus="DE0 0"):
    base_wacc = 0.07  # Base WACC
    base_wacc_solar_rooftop = 0.04  # Base WACC for solar (avg. of grounded (0.07) and rooftop (0.04))
    risk_free_rate = 0.02  # Risk-free rate
    stats_without_CfD = annual_revenue_statistics(carrier, bus)
    revenue_variability_without_CfD = stats_without_CfD["std_dev"]
    stats_with_CfD = annual_revenue_statistics_with_CfD(carrier, True, 0, bus)
    revenue_variability_with_CfD = stats_with_CfD["std_dev"]
    if carrier == 'solar rooftop':
        updated_wacc = risk_free_rate + (base_wacc_solar_rooftop - risk_free_rate) * (revenue_variability_with_CfD / revenue_variability_without_CfD)
    else:
        updated_wacc = risk_free_rate + (base_wacc - risk_free_rate) * (revenue_variability_with_CfD / revenue_variability_without_CfD)
    return updated_wacc


# Updated annuity factor based on updated discount rate
def updated_capital_cost(carrier, bus="DE0 0"):
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
    wacc = updated_discount_rate(carrier, bus)
    annuity_factor_new = (1 - (1 + wacc) ** -lifetime) / wacc
    capital_cost = 0.0
    if carrier == 'offwind':
        capital_cost = 1682122.6 * annuity_factor_old / annuity_factor_new # Taken from the costs_2030.csv file for offshore wind
    else:
        capital_cost = n.generators.capital_cost[n.generators.carrier == carrier] * annuity_factor_old / annuity_factor_new
    return capital_cost


# Annual state CfD payments for a given carrier and year
def annual_CfD_payments_per_MW(carrier, year, use_avg_rev_per_MWh=True, strike_price=0, bus="DE0 0"):
    rev_without_CfD = annual_revenue_per_MW(carrier, year, bus)
    rev_with_CfD = annual_revenue_per_MW_with_CfD(carrier, year, use_avg_rev_per_MWh, strike_price, bus)
    payments = rev_with_CfD - rev_without_CfD
    return payments


# Plot annual state CfD payments for a given carrier across all years
def plot_annual_CfD_payments(carrier, use_avg_rev_per_MWh=True, strike_price=0, bus="DE0 0", save_to_file=True):
    years = n.generators_t.p.index.year.unique()
    payments = [annual_CfD_payments_per_MW(carrier, year, use_avg_rev_per_MWh, strike_price, bus) for year in years]
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
        output_dir = f'{results_directory}/my_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/annual_CfD_payments_{carrier}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f'Plot saved to {filename}')
    else:
        plt.show()


# Plot installed capacities of a carrier across different model iterations
def plot_installed_capacities(carrier, save_to_file=True): 
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


#%% Execute these things

carriers = ['offwind', 'onwind', 'solar', 'solar rooftop']
for carrier in carriers:
    plot_annual_revenue_per_MW(carrier)
    plot_annual_revenue_per_MW_with_CfD(carrier)

for carrier in carriers:
    revenue_statistics_comparison(carrier)

for carrier in carriers:
    print(updated_discount_rate(carrier))
    print(updated_capital_cost(carrier))

for carrier in carriers: #['offwind-ac', 'offwind-dc', 'offwind-float', 'onwind', 'solar', 'solar rooftop']:
    print(n.generators.capital_cost[n.generators.carrier == carrier])
    print(updated_capital_cost(carrier))

for carrier in carriers:
    plot_revenue_violin(carrier)

# Only execute once all model iterations have been run succesfully
for carrier in carriers:
    plot_installed_capacities(carrier)


n.generators.capital_cost[n.generators.carrier == 'offwind']


stats = statistics()
for tag in ["generation_total", "VRE_generation_share", "Generation Onshore Wind", "Generation Offshore Wind", "Generation Solar", "Generation Gas", "Generation Coal", "Generation Oil", "Average Price", "CO2_emissions"]:
    compare_statistics_plot(tag)





#%% Test calls and examples

# Network nodes
nodes = n.buses.index.tolist()
print(nodes)

# List of generators at node 'DE0 0'
generators_de0_0 = n.generators[n.generators.bus == 'DE0 0']
print(generators_de0_0)

# List of onshore wind generators at each node
wind_onshore = n.generators[n.generators.carrier == 'onwind']
print(wind_onshore)

# List of carriers
carriers = n.generators.carrier.unique()
print(carriers)


hgy_onwind = get_hourly_generation('onwind')
hgy_onwind_2004 = hgy_onwind[hgy_onwind.index.year == 2004]
print(hgy_onwind_2004)

p_nom_opt_onwind = n.generators.p_nom_opt[n.generators.carrier == 'onwind'].sum()
print(p_nom_opt_onwind)

hgy_onwind_per_MW = hgy_onwind_2004 / p_nom_opt_onwind
print(hgy_onwind_per_MW)

hg_per_mw = get_hourly_generation_per_MW('offwind-ac')
print(hg_per_mw.sum().values[0])
np.max(hg_per_mw.values.flatten())
np.min(hg_per_mw.values.flatten())
np.average(hg_per_mw.values.flatten())



price = get_hourly_marginal_price()
price_2004 = price[price.index.year == 2004]
print(price_2004)
type(price_2004)

print(price_2004.values)

rev_2004 = hgy_onwind_2004.values.flatten() * price_2004.values.flatten()
print(rev_2004)


# Installed generation capacities per generator type at nodes ['DE0 0', ..., 'DE0 12']
p_set_values = [n.generators['DE0 {}'.format(i)]['p_set'] for i in range(0, 12)]


# Optimized installed capacities
installed_capacities = n.generators.p_nom_opt
print(installed_capacities)

installed_capacities_onwind = installed_capacities[installed_capacities.index.str.contains('onwind')]
print(installed_capacities_onwind)
installed_capacities_offwind_ac = installed_capacities[installed_capacities.index.str.contains('offwind-ac')]
print(installed_capacities_offwind_ac)
installed_capacities_offwind_dc = installed_capacities[installed_capacities.index.str.contains('offwind-dc')]
print(installed_capacities_offwind_dc)
installed_capacities_offwind_float = installed_capacities[installed_capacities.index.str.contains('offwind-float')]
print(installed_capacities_offwind_float)
installed_capacities_solar = installed_capacities[installed_capacities.index.str.contains('solar')]
print(installed_capacities_solar)


# Initial installed capacities
initial_installed_capacities = n.generators.p_nom
print(initial_installed_capacities)

# Show all non-zero entries in initial_installed_capacities
print(initial_installed_capacities[initial_installed_capacities > 0])
print(initial_installed_capacities[initial_installed_capacities.index.str.contains('onwind')])
print(installed_capacities[installed_capacities.index.str.contains('onwind')])


# Capital costs (in €/MW)
capital_costs = n.generators.capital_cost
print(capital_costs)
capital_costs_onwind = capital_costs[capital_costs.index.str.contains('onwind')].sum()
print(capital_costs_onwind)

# Lifetime
lifetime = n.generators.lifetime
print(lifetime)
lifetime_onwind = lifetime[lifetime.index.str.contains('onwind')].sum()
print(lifetime_onwind)

# Annualized full costs (in €/MW)
annualized_full_costs_onwind = capital_costs_onwind / lifetime_onwind
print(annualized_full_costs_onwind)

total_revenues_per_MW_onwind = sum(annual_revenue_per_MW('onwind', year) for year in range(2004, 2012))
print(total_revenues_per_MW_onwind)


n.generators_t.p.sum().sum()

# Get CO2 emissions




avg_annual_rev = [86324.46, 232970.37, 43409.03]
risk_without_CfD = [6591.28, 22642.75, 2463.83]
risk_without_CfD_norm = [risk_without_CfD[i] / avg_annual_rev[i] for i in range(len(avg_annual_rev))]
print([round(x, 5) for x in risk_without_CfD_norm])
risk_fin_CfD = [5990.00, 7114.20, 1172.30]
risk_fin_CfD_norm = [risk_fin_CfD[i] / avg_annual_rev[i] for i in range(len(avg_annual_rev))]
print([round(x, 5) for x in risk_fin_CfD_norm])
risk_state_fin_CfD = [a - b for a, b in zip(risk_without_CfD, risk_fin_CfD)]
print([round(x, 5) for x in risk_state_fin_CfD])



#%%

n_test = pypsa.Network("resources/2030_without_CfD_v5/networks/base_s_1_elec_EP.nc")

print(n_test.generators.p_nom_min)
print(n_test.generators.p_nom_max)
print(n_test.generators.p_nom_opt)

n_test2 = pypsa.Network("results/2030_without_CfD_v5/networks/base_s_1_EP__2030.nc")

print(n_test2.generators.p_nom_opt)
print(n_test2.generators.p_nom_min)
print(n_test2.generators.p_nom_max)


n2 = pypsa.Network("results/2030_without_CfD_v5/networks/base_s_1_EP__2030.nc")

print(n2.generators.p_nom_opt)
print(n.generators.p_nom_opt)

print(n.generators_t.p.sum()) # Results with CM
print(n2.generators_t.p.sum()) # Results without CfD

print(n.generators_t.p.sum().sum())
print(n2.generators_t.p.sum().sum())
