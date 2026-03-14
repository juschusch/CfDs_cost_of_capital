<!--
SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
SPDX-License-Identifier: CC-BY-4.0
-->

# CfDs_Cost_of_Capital

This project was used for the paper "Taming the Electricity Market RollerCOSTer – A Systemic Analysis of Contracts for Difference as a Risk-Hedging Instrument" (add link) based on the v2025.07.0 version of PyPSA-EUR (https://github.com/PyPSA/pypsa-eur/releases/tag/v2025.07.0). 


## General idea

### Overview

This repository is used to model the potential impact of Contracts for Difference (CfDs) on the energy system—here on the example of the German electricity price zone. The general workflow uses three iterations:

1. **Calibration scenario** — calculated using mostly default parameters to calibrate initial cost of capital assumptions for all generators
2. **Reference scenario** — uses the resulting price and dispatch time series from calibration under direct marketing assumptions for all generators
3. **CfD scenarios** — uses updated cost of capital values based on revenues under different CfD designs as inputs for the final modeling

### System Configuration

The general configuration is an overnight (or "greenfield") scenario for Germany with the following specifications:

#### Spatial and Temporal Setup
- **Region:** Germany (single node, effectively ignoring transmission constraints)
- **Time period:** Weather years 1982 to 2016 (35 years of historical weather data)
- **Snapshots:** Hourly resolution

#### Technology Portfolio

**Extendable technologies:**
- Solar, solar-hsat (horizontal single-axis tracking)
- Onwind, offwind-ac, offwind-dc, offwind-float
- OCGT, CCGT, coal, lignite
- Batteries, hydrogen storage

**Non-extendable technologies (existing capacity constraints):**
- Hydro, geothermal, biomass — capped to current expansion levels

#### Key Assumptions

- **Offshore capacity:** ~80 GW long-term potential (legal expansion target for 2045 in Germany)
  - Achieved through increased `capacity_per_sqkm` parameter (6.5 instead of default 2)
- **CO2 price:** 100 €/t CO2eq (with sensitivity analysis)

### Monte Carlo Simulation and Revenue Analysis

A Monte Carlo Simulation is performed across all weather years to generate a distribution of possible lifetime generator revenues under different CfD designs:

| Scenario | Design | Description |
|----------|--------|-------------|
| **mb** | Direct marketing | Market-based revenues without CfD support |
| **pb** | Production-based CfD | Conventional CfD with production-based payment |
| **pi** | Production-independent CfD | Financial CfD with production-independent payment |
| **cb** | Capacity-based CfD | Capacity payment independent of actual generation |

#### Cost of Capital Calculation

The resulting lifetime revenue risk (std dev / mean) is used to calculate the corresponding cost of capital (WACC / discount rate) value for each technology and resource class:

$$\text{WACC} = f_{irr} - \frac{\ln(1 - \text{erf}(\sigma \cdot 2^{-3/2}))}{T}$$

where:
- $f_{irr}$ = base risk-free rate
- $\sigma$ = normalized revenue volatility (risk)
- $T$ = project lifetime (years)

#### Analysis Functions

All functions for the Monte Carlo Simulation, revenue calculations, and general evaluation of results are defined in [analysis/analysis_functions.py](analysis/analysis_functions.py).



## Modifications to the original code

This section documents the key modifications made to the base PyPSA-EUR framework to support contract-for-difference analysis and resource class-specific cost of capital assumptions.

### 1. Value of Lost Load (VoLL) Generator

**Purpose:** Cap electricity prices at a realistic technical maximum while allowing the model to maintain feasibility.

**Implementation:**
- File: [scripts/add_electricity.py](scripts/add_electricity.py)
- A synthetic "load" generator with:
  - **Marginal cost:** 4,000 €/MWh (technical maximum at European Energy Exchange)
  - **Investment & O&M costs:** 0 €/MW(h)
  - Effect: Caps resulting price time series at 4,000 €/MWh

**Configuration:**
```yaml
electricity:
  everywhere_powerplants: [load]
costs:
  overwrites:
    marginal_cost:
      load: 4000.0
    capital_cost:
      load: 0.0
```

---

### 2. Synthetic Demand Data (ERAA 2023)

**Purpose:** Use realistic European demand projections instead of default assumptions.

**Data source:** ERAA 2023 demand projections for Germany at 2030 demand level, mapped to historical weather years 1982–2016.

**Implementation:**
- Files: 
  - [analysis/Convert_ERAA2023_file_format.py](analysis/Convert_ERAA2023_file_format.py) — conversion script
  - [scripts/build_electricity_demand.py](scripts/build_electricity_demand.py) — integration
  - Data: [data/electricity_demand_ERAA2023.csv](data/electricity_demand_ERAA2023.csv)

**Configuration:**
```yaml
load:
  custom_demand_file: "data/electricity_demand_ERAA2023.csv"
```

---

### 3. Equal-Size Resource Classes

**Purpose:** Distribute renewable generators into resource classes of equal size rather than by percentile, improving spatial resolution.

**Original PyPSA-EUR approach:** Percentile-based clustering yields regions with vastly different sizes.

**Implementation:** Modified resource class feature divides each node into multiple regions of equal size, ranked by average capacity factor.

**File:** [scripts/build_renewable_profiles.py](scripts/build_renewable_profiles.py)

**Configuration:**
```yaml
renewable:
  onwind:
    resource_classes: 10
  solar:
    resource_classes: 10
  solar-hsat:
    resource_classes: 10
  offwind-ac:
    resource_classes: 4
  offwind--dc:
    resource_classes: 4
  offwind-float:
    resource_classes: 4
```

---

### 4. Custom Area Constraints

**Purpose:** Enforce realistic legal limits on renewable capacity expansion based on German policy targets.

**Constraints:**
- **Onshore wind:** Maximum 160 GW total
- **Solar:** Maximum 400 GW total (aggregated across `solar` and `solar-hsat`)
- **Minimum capacity:** 0.1 MW per resource class (ensures revenue data generation for all resource classes)

**Distribution:** Upper limits are distributed across resource classes proportional to their average capacity factor.

**Implementation:**
- File: [data/resource_class_capacity_constraint.py](data/resource_class_capacity_constraint.py)
- Integrated via the `custom_extra_functionality` feature

**Configuration:**
```yaml
solving:
  options:
    custom_extra_functionality: "../data/resource_class_capacity_constraint.py"
```

---

### 5. Resource Class-Specific Cost of Capital

**Purpose:** Allow region- and technology-specific discount rate (WACC) assumptions to reflect different financing costs across resource classes.

**Implementation:** Extended cost overwrite system to support per-resource-class parameters.

**File:** [scripts/add_electricity.py](scripts/add_electricity.py)

**Configuration example:**
```yaml
costs:
  overwrites:
    discount rate:
      onwind:
        DE0 0 0: 0.03
        DE0 0 1: 0.03
        DE0 0 2: 0.03
        # ... continues for all resource classes
      offwind-ac:
        DE0 0 0: 0.04
        # ... etc
```

**Format:** `<technology>:<region> <cluster> <resource_class>: <discount_rate>`

---

### 6. Dynamic Fuel Price Modeling

**Purpose:** Support scenario-based or stochastic fuel price assumptions beyond historical values.

**Options:**

#### 6a. Scenario-based pricing (actually used in the paper)
Divides the time series into discrete price regimes with specified probabilities and relative price levels.

**Configuration:**
```yaml
conventional:
  dynamic_fuel_price: scenarios
  dfp_fuel_types: [gas]
  scn_distribution: [[0.2, 0.5], [0.6, 1.0], [0.2, 2.0]]
  # [share_of_snapshots, relative_price_multiplier]
```

Example interpretation: 20% of snapshots use 50% of default gas price, 60% use 100%, and 20% use 200%.

#### 6b. Random walk pricing
Generates a stochastic price path based on historical volatility.

**Configuration:**
```yaml
conventional:
  dynamic_fuel_price: random_walk
  dfp_fuel_types: [gas]
  rw_random_seed: 140523498
  rw_random_walk_stddev: 0.05  # 5% standard deviation
```

**File:** [scripts/add_electricity.py](scripts/add_electricity.py)




## Workflow

### 1. Installation of PyPSA-EUR v2025.07.0

This project is based on the PyPSA-EUR version v2025.07.0. Follow these steps to set up the environment:

#### Prerequisites
- Anaconda/Miniconda installed
- Git installed
- Gurobi solver (optional, but recommended for optimization)

#### Step 1: Clone the PyPSA-EUR repository
```bash
git clone https://github.com/PyPSA/pypsa-eur.git
cd pypsa-eur
```

#### Step 2: Checkout the specific version
```bash
git checkout v2025.07.0
```

#### Step 3: Create and activate the conda environment
```bash
conda env create -f environment.yaml
conda activate pypsa-eur
```

#### Step 4: Install additional dependencies (if needed)
```bash
pip install -e .
```

#### Step 5: Verify installation
```bash
python -c "import pypsa; print(pypsa.__version__)"
```

#### Step 6: Configure Gurobi (optional)
If you have a Gurobi license, activate it:
```bash
grbgetkey <your-license-key>
```

#### Step 7: Clone this modified CfD project
Clone this repository into your workspace and configure it to use the PyPSA-EUR installation from Step 1-3:

```bash
git clone <repository-url>
cd CfDs_cost_of_capital
```

The project uses Snakemake to manage the workflow. All rules reference the PyPSA-EUR installation, so ensure the conda environment is activated before running Snakemake.

---

### 2. Running the analysis

The analysis workflow consists of three main iterations, each building on the results of the previous one. Follow these steps in order:

---

#### Step 1: Create the Weather Data Cutout

**Purpose:** Generate the ERA5 weather data cutout required for all subsequent runs.

**Data:** The cutout contains hourly weather data for Germany (2016 × 35 years) from the ERA5 reanalysis database.

**Download time:** ~2–48 hours (depending on CDS server availability)  
**File size:** ~11.6 GB

**Command:**
```bash
snakemake "cutouts/de-1982-2016-era5.nc" --configfile config/create_cutout.yaml --cores <N>
```

Replace `<N>` with the number of CPU cores to use. The cutout only needs to be created once; subsequent runs will reuse it.

---

#### Step 2: Run Iteration Zero (Calibration)

**Purpose:** Establish baseline system and calibrate cost of capital assumptions using default WACC of 7.00%.

**Configuration:** [config/iteration_zero_v3.yaml](config/iteration_zero_v3.yaml)

**Output:** Network solution stored in `results/0_Iteration_Zero_v3/networks/base_s_1_elec_EP.nc`

**Command:**
```bash
snakemake solve_elec_networks --configfile config/iteration_zero_v3.yaml --cores <N>
```

---

#### Step 3: Monte Carlo Simulation and WACC Calculation (Iteration Zero)

**Purpose:** Generate lifetime revenue distributions and calculate risk-adjusted WACC values for the market-based ('mb') scenario.

**Key calculation:** Revenue risk is converted to WACC using the Modigliani-Miller theorem with adjustments for volatility.

**Python code** (run in [analysis/](analysis/) directory or Jupyter notebook):

```python
import importlib
import analysis_functions as af

importlib.reload(af)

# Load network and cost data
run_name = "0_Iteration_Zero_v3"
n = af.get_network(run_name)
costs = af.get_costs(run_name)

# Generate MCS for project lifetimes: 20, 30, 40 years
for t in [20, 30, 40]:
    af.years_mcs(run_name, n, T=t, N=10000)
    af.revenue_mcs(costs, run_name, n, T=t, N=10000)

# Calculate updated WACC for market-based scenario
# Repeat for each carrier and resource class
updated_wacc_mb = af.wacc_updated(run_name, n, carrier='onwind', rc=0, scenario='mb')
print(f"Updated WACC (mb): {updated_wacc_mb:.4f}")
```

**Output:** MCS data saved to `results/0_Iteration_Zero_v3/MCS/` directory  
**Next step:** Fill calculated WACC values into `config/iteration_one_v2.yaml`

---

#### Step 4: Run Iteration One (Market-Based Reference)

**Purpose:** Re-run the system with updated WACC values from Iteration Zero to establish a stable market-based baseline.

**Configuration:** [config/iteration_one_v2.yaml](config/iteration_one_v2.yaml)

**Setup:** Copy the WACC values calculated in Step 3 into the `costs.overwrites.discount rate` section, organized by technology and resource class:

```yaml
costs:
  overwrites:
    discount rate:
      onwind:
        DE0 0 0: 0.0347  # Updated value from Step 3
        DE0 0 1: 0.0354
        # ... continue for all resource classes
      offwind-ac:
        DE0 0 0: 0.0412
        # ... etc
```

**Command:**
```bash
snakemake solve_elec_networks --configfile config/iteration_one_v2.yaml --cores <N>
```

**Output:** Network solution stored in `results/1_Iteration_One_v2/networks/base_s_1_elec_EP.nc`

---

#### Step 5: Analyze Iteration One Results (All CfD Scenarios)

**Purpose:** Calculate WACC values for the three CfD scenarios ('pb', 'pi', 'cb') using Iteration One network and prices.

**Python code:**

```python
# Repeat Step 3 analysis for CfD scenarios
run_name = "1_Iteration_One_v2"
n = af.get_network(run_name)
costs = af.get_costs(run_name)

for t in [20, 30, 40]:
    af.years_mcs(run_name, n, T=t, N=10000)
    af.revenue_mcs(costs, run_name, n, T=t, N=10000)

# Calculate WACC for each scenario
for scenario in ['pb', 'pi', 'cb']:
    wacc = af.wacc_updated(run_name, n, carrier='onwind', rc=0, scenario=scenario)
    print(f"Updated WACC ({scenario}): {wacc:.4f}")
```

**Output:** MCS data for three additional scenarios  
**Next step:** Use these WACC values for Iteration Two runs

---

#### Step 6: Run Iteration Two (CfD Scenarios)

**Purpose:** Run final scenarios with WACC values optimized for each CfD mechanism.

**Three parallel runs required:**

1. **Production-based (pb)** conventional CfD
   ```bash
   snakemake solve_elec_networks --configfile config/iteration_two_pb.yaml --cores <N>
   ```

2. **Production-independent (pi)** financial CfD
   ```bash
   snakemake solve_elec_networks --configfile config/iteration_two_pi.yaml --cores <N>
   ```

3. **Capacity-based (cb)** capacity-based CfD
   ```bash
   snakemake solve_elec_networks --configfile config/iteration_two_cb.yaml --cores <N>
   ```

**Setup:** For each scenario, fill the corresponding WACC values from Step 5 into the respective config file's `costs.overwrites.discount rate` section.

---

#### Step 7: Final Analysis and Comparison

**Purpose:** Analyze and compare results across all four CfD scenarios.

**Key analyses:**

1. **System-wide metrics:** Total investment, generation by technology, system costs
2. **Revenue analysis:** Lifetime revenues per MW by carrier and resource class
3. **Risk metrics:** Revenue volatility (std dev / mean) under each CfD design
4. **Cost of capital:** Compare WACC reduction across scenarios
5. **Distributional analysis:** Fit probability distributions to revenues

**Python code:**

```python
import importlib
import analysis_functions as af

importlib.reload(af)

# Compare results across scenarios
scenarios = ['mb', 'pb', 'pi', 'cb']
carriers = ['onwind', 'offwind-ac', 'offwind-dc', 'solar', 'solar-hsat']

for scenario in scenarios[1:]:  # Skip mb (already analyzed)
    run_name = f"2_Iteration_Two_{scenario.upper()}"
    n = af.get_network(run_name)
    costs = af.get_costs(run_name)
    
    # Revenue statistics and distributions
    for carrier in carriers:
        for rc in range(n.generators.index[n.generators.carrier == carrier].size):
            af.revenue_statistics_comparison(run_name, carrier, n, rc)
            af.lifetime_revenue_mcs_dist_plot(run_name, scenario=scenario, carrier=carrier, N=10000)
```

**Output locations:**
- Plots: `results/<run_name>/my_plots/`
- MCS data: `results/<run_name>/MCS/`
- Network: `results/<run_name>/networks/`

---

### Notes

- **Computational time:** Each Snakemake run typically requires 1-6 hours depending on hardware and solver performance
- **Storage:** Expect ~10–20 GB total for all four Iteration Two runs
- **Solver**: Gurobi (commercial license required; academic licenses available)