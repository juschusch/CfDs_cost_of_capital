# SPDX-FileCopyrightText: : 2023- The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

import logging
import pandas as pd

logger = logging.getLogger(__name__)

def resource_class_share_constraint(n, snapshots, snakemake):
    """
    Add custom extra functionality constraints for resource class distribution.
    Constrains the *optimized* capacities (p_nom_opt) to follow a specific distribution.
    """
    logger.info("Adding custom constraints for resource class distribution...")
    
    # Define the technologies to apply this to
    target_techs = ["onwind", "solar", "solar-hsat"] 
    # Define distribution factor
    # Highest bin gets a times the share of the lowest bin
    a = 2.0
    # Define relaxation factor (e.g., 0.2 for +/- 20% flexibility)
    # relaxation = 0.2
    
    for tech in target_techs:
        # Find all generators for this technology
        # We filter for generators that are extendable, as fixed ones are parameters
        gens = n.generators[(n.generators.carrier == tech) & n.generators.p_nom_extendable]
        
        # Filter out generators with negligible potential (e.g. < 1 MW)
        # This prevents bins with 0 potential from forcing the total capacity to 0
        if "p_nom_max" in n.generators.columns:
            gens = gens[gens.p_nom_max > 1.0]
        
        if gens.empty:
            continue
            
        # Group by bus to handle each location separately
        gens_by_bus = gens.groupby("bus")
        
        for bus, group in gens_by_bus:
            nbins = len(group)
            if nbins <= 1:
                continue
                
            # Distribute shares such that highest bin gets a times the share of the lowest bin
            # Linear distribution from x to a*x.
            # Sum of shares = 1.
            # base_share = 2 / (nbins * (1 + a) - a + 1)
            # ith share (0-indexed): s_i = base_share * (1 + i * (a-1) / nbins))
            
            shares = [2 / nbins * (nbins + (a - 1) * i) / ((a + 1) * nbins - a + 1) for i in range(nbins)]
            
            # Create a Series for shares aligned with the group
            shares_series = pd.Series(shares, index=group.index)
            
            # Get the p_nom variables for these generators
            # n.model["Generator-p_nom"] is the optimization variable for capacity (p_nom_opt)
            p_nom_opt_vars = n.model["Generator-p_nom"].loc[group.index]
            
            # Calculate the sum of optimized capacities for this bus and tech
            sum_p_nom_opt = p_nom_opt_vars.sum()

            # Vectorized constraint
            lhs = p_nom_opt_vars - sum_p_nom_opt * shares_series 

            n.model.add_constraints(lhs == 0, name=f"ResourceClassShare_{bus}_{tech}")

            # Ensure that the model does not return the trivial solution of all zeros.
            n.model.add_constraints(sum_p_nom_opt >= 1, name=f"ResourceClassShare_nontrivial_{bus}_{tech}")
            
            # Vectorized constraint with relaxation window
            # p_nom_opt_i >= sum_p_nom_opt * share_i * (1 - relaxation)
            # p_nom_opt_i <= sum_p_nom_opt * share_i * (1 + relaxation)
            
            # lhs_lower = p_nom_opt_vars - sum_p_nom_opt * shares_series * (1 - relaxation)
            # lhs_upper = p_nom_opt_vars - sum_p_nom_opt * shares_series * (1 + relaxation)
            
            # n.model.add_constraints(lhs_lower >= 0, name=f"ResourceClassShare_lower_{bus}_{tech}")
            # n.model.add_constraints(lhs_upper <= 0, name=f"ResourceClassShare_upper_{bus}_{tech}")



            
                
    logger.info("Finished adding resource class constraints.")
