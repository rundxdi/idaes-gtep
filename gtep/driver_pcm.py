#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#################################################################################

import os
import csv
import logging
import pyomo.environ as pyo
from pyomo.contrib.appsi.solvers.highs import Highs
from pyomo.contrib.appsi.solvers.gurobi import Gurobi

from gtep.gtep_model import ExpansionPlanningModel
from gtep.gtep_data import ExpansionPlanningData
from gtep.gtep_solution import ExpansionPlanningSolution
from gtep.gtep_data_processing import DataProcessing

# Optional if using xpress
import xpress

import math

def sanitize_loaded_md(md, representative_data=None):
    """
    Clean NaN / None values in loaded Egret model data and representative data.

    - fixes branch ratings/reactance/resistance
    - fixes renewable p_max timeseries values
    - fixes generator p_min/p_max/ramp fields
    - fixes storage rates/capacity if present
    """
    elems = md.data["elements"]

    # ------------------------------------------------------------------
    # generators in base md
    # ------------------------------------------------------------------
    for gen, g in elems.get("generator", {}).items():
        for key, fallback in [
            ("p_min", 0.0),
            ("p_max", 0.0),
            ("startup_capacity", 0.0),
            ("shutdown_capacity", 0.0),
            ("ramp_up_60min", 0.0),
            ("ramp_down_60min", 0.0),
            ("ramp_q", 0.0),
            ("ramp_agc", 0.0),
            ("fuel_cost", 0.0),
            ("non_fuel_startup_cost", 0.0),
            ("min_up_time", 1.0),
            ("min_down_time", 1.0),
            ("spinning_reserve_frac", 0.0),
            ("quickstart_reserve_frac", 0.0),
            ("max_spinning_reserve", 0.0),
            ("max_quickstart_reserve", 0.0),
            ("heat_rate", 0.0),
        ]:
            if key in g:
                try:
                    if g[key] is None or (isinstance(g[key], float) and math.isnan(g[key])):
                        g[key] = fallback
                except Exception:
                    g[key] = fallback

    # ------------------------------------------------------------------
    # branches in base md
    # ------------------------------------------------------------------
    for br, b in elems.get("branch", {}).items():
        for key, fallback in [
            ("resistance", 0.0),
            ("reactance", 1e-6),  # avoid divide-by-zero / NaN
            ("charging_susceptance", 0.0),
            ("rating_long_term", 0.0),
            ("rating_short_term", 0.0),
            ("rating_emergency", 0.0),
        ]:
            if key in b:
                try:
                    if b[key] is None or (isinstance(b[key], float) and math.isnan(b[key])):
                        b[key] = fallback
                except Exception:
                    b[key] = fallback

    # ------------------------------------------------------------------
    # dc branches in base md
    # ------------------------------------------------------------------
    for br, b in elems.get("dc_branch", {}).items():
        for key, fallback in [
            ("rating_long_term", 0.0),
            ("rating_short_term", 0.0),
            ("rating_emergency", 0.0),
        ]:
            if key in b:
                try:
                    if b[key] is None or (isinstance(b[key], float) and math.isnan(b[key])):
                        b[key] = fallback
                except Exception:
                    b[key] = fallback

    # ------------------------------------------------------------------
    # storage in base md
    # ------------------------------------------------------------------
    for s, st in elems.get("storage", {}).items():
        for key, fallback in [
            ("max_discharge_rate", 0.0),
            ("min_discharge_rate", 0.0),
            ("max_charge_rate", 0.0),
            ("min_charge_rage", 0.0),
            ("energy_capacity", 0.0),
            ("initial_state_of_charge", 0.5),
            ("minimum_state_of_charge", 0.0),
            ("charge_efficiency", 1.0),
            ("discharge_efficiency", 1.0),
            ("ramp_up_input_60min", 0.0),
            ("ramp_up_output_60min", 0.0),
            ("ramp_down_input_60min", 0.0),
            ("ramp_down_output_60min", 0.0),
        ]:
            if key in st:
                try:
                    if st[key] is None or (isinstance(st[key], float) and math.isnan(st[key])):
                        st[key] = fallback
                except Exception:
                    st[key] = fallback

    # ------------------------------------------------------------------
    # representative data renewable p_max cleanup
    # ------------------------------------------------------------------
    if representative_data is not None:
        for rep in representative_data:
            for gen, g in rep.data["elements"].get("generator", {}).items():
                if g.get("generator_type") == "renewable":
                    pmax = g.get("p_max")

                    # dict time series case
                    if isinstance(pmax, dict) and "values" in pmax:
                        vals = pmax.get("values", [])
                        fallback = 0.0

                        # derive fallback from static/base md
                        base_g = elems["generator"].get(gen, {})
                        base_pmax = base_g.get("p_max", 0.0)

                        if isinstance(base_pmax, (int, float)):
                            fallback = float(base_pmax)
                        elif isinstance(base_pmax, dict):
                            if "reference_value" in base_pmax and base_pmax["reference_value"] is not None:
                                fallback = float(base_pmax["reference_value"])
                            elif "values" in base_pmax and base_pmax["values"]:
                                try:
                                    fallback = max(
                                        float(v) for v in base_pmax["values"]
                                        if v is not None and not (isinstance(v, float) and math.isnan(v))
                                    )
                                except Exception:
                                    fallback = 0.0

                        cleaned = []
                        for v in vals:
                            if v is None or (isinstance(v, float) and math.isnan(v)):
                                cleaned.append(fallback)
                            else:
                                cleaned.append(float(v))
                        g["p_max"]["values"] = cleaned

                    # scalar pmax case
                    elif pmax is None or (isinstance(pmax, float) and math.isnan(pmax)):
                        g["p_max"] = 0.0

    return md

logger = logging.getLogger("gtep.driver_naerm")
logger.setLevel(logging.INFO)

###############################################################################
# USER INPUTS
###############################################################################

# Converted Prescient/GMLC-compatible case directory
from pathlib import Path

# data_path = "~/2030_pcm_case/base_case_pcm_2030"
data_path = "/ascldap/users/jkskolf/2030_pcm_case/base_case_pcm_2030"
print(data_path)
# gen_csv_path = data_path / "gen.csv"
# storage_csv_path = data_path / "storage.csv"
# branch_csv_path = data_path / "branch.csv"


# Representative periods for 2030
rep_days = [
    "2030-01-28 00:00",
    "2030-04-23 00:00",
    "2030-07-05 00:00",
    "2030-10-14 00:00",
]
rep_weights = [1, 1, 1, 1]

# Output folder
data_date = "07-13-2026"
dir_name = f"GTEP_2030_converted_case_{data_date}"
os.makedirs(dir_name, exist_ok=True)
print(f"\nCreated output directory: {dir_name}")

# Toggle candidate cost processing
include_candidate_cost_data = False

# Candidate generation cost files (only used if include_candidate_cost_data = True)
bus_data_path = "./data/costs/Bus_data_gen_weights_mappings.csv"
cost_data_path = "./data/costs/2022_v3_Annual_Technology_Baseline_Workbook_Mid-year_update_2-15-2023_Clean.xlsx"
ng_cost_path = "./data/costs/Total_Energy_Supply_Disposition_and_Price_Summary.csv"
candidate_gens = [
    "Natural Gas_CT",
    "Natural Gas_FE",
    "Solar - Utility PV",
    "Land-Based Wind",
]

###############################################################################
# LOAD DATA
###############################################################################

print("Creating ExpansionPlanningData object...")

# Keep same style as your driver_naerm.py
# NOTE: if your installed gtep_data.py does not support
# duration_representative_period / save_period_structure_file /
# period_structure_json_file, comment those out and use the simpler API.
data_object = ExpansionPlanningData(
    stages=1,
    num_reps=4,
    num_commit=24,
    num_dispatch=1,
    duration_representative_period=24,
    save_period_structure_file=False,
    period_structure_json_file=None,
)

print(data_path)
print(f"Loading converted case from: {data_path}")


print("data_object.num_reps =", getattr(data_object, "num_reps", None))
print("len(rep_days) =", len(rep_days))
print("len(rep_weights) =", len(rep_weights))
# print("len(representative_data) =", len(data_object.representative_data))

data_object.load_prescient(
    data_path,
    representative_dates=rep_days,
    representative_weights=rep_weights,
)

print("data_object.num_reps =", getattr(data_object, "num_reps", None))
print("len(rep_days) =", len(rep_days))
print("len(rep_weights) =", len(rep_weights))
print("len(representative_data) =", len(data_object.representative_data))


from pprint import pprint

print("\n=== Renewable generator p_max inspection ===")

renewables = []
for gen in data_object.md.data["elements"]["generator"]:
    g = data_object.md.data["elements"]["generator"][gen]
    if g.get("generator_type") == "renewable":
        renewables.append(gen)

print(f"Total renewable generators: {len(renewables)}")

for gen in renewables[:20]:
    g = data_object.md.data["elements"]["generator"][gen]
    print(f"\nGenerator: {gen}")
    print("unit_type:", g.get("unit_type"))
    print("p_max type:", type(g.get("p_max")))
    print("p_max value:")
    pprint(g.get("p_max"))

print("\n=== Representative period renewable p_max inspection ===")
for gen in renewables[:10]:
    print(f"\nGenerator: {gen}")
    for i, rep in enumerate(data_object.representative_data[:2]):
        try:
            pmax = rep.data["elements"]["generator"][gen]["p_max"]
            print(f"  rep {i} type={type(pmax)} value={pmax}")
        except Exception as e:
            print(f"  rep {i} ERROR: {e}")

# raise SystemExit

print("Prescient/GMLC case loaded successfully.")
print(f"Representative periods loaded: {len(data_object.representative_data)}")
print(f"Top-level keys: {list(data_object.md.data.keys())}")
print(f"Element groups: {list(data_object.md.data['elements'].keys())}")

for k, v in data_object.md.data["elements"].items():
    try:
        print(f"  {k}: {len(v)}")
    except Exception:
        print(f"  {k}: <non-sized>")

print("\n=== Checking renewable p_max values in representative data ===")

bad = []

for gen in data_object.md.data["elements"]["generator"]:
    g = data_object.md.data["elements"]["generator"][gen]
    if g.get("generator_type") != "renewable":
        continue

    for i, rep in enumerate(data_object.representative_data):
        try:
            pmax = rep.data["elements"]["generator"][gen]["p_max"]
            vals = pmax["values"] if isinstance(pmax, dict) and "values" in pmax else None

            if vals is None:
                bad.append((gen, i, "missing values field"))
                continue

            if len(vals) == 0:
                bad.append((gen, i, "empty values"))
                continue

            if all(v is None for v in vals):
                bad.append((gen, i, "all None"))
                continue

            if any(v is None for v in vals):
                bad.append((gen, i, "partial None"))
                continue

        except Exception as e:
            bad.append((gen, i, f"error: {e}"))

print(f"Found {len(bad)} bad renewable representative-series cases")
for row in bad[:100]:
    print(row)
        

###############################################################################
# OPTIONAL COST DATA FOR CANDIDATE GENERATORS
###############################################################################

data_processing_object = None

if include_candidate_cost_data:
    print("Loading candidate generator cost data...")

    data_processing_object = DataProcessing()

    # Keep this close to your original usage, but depending on your local
    # DataProcessing implementation, you may need to use the simpler call:
    #
    # data_processing_object.load_gen_data(
    #     bus_data_path,
    #     cost_data_path,
    #     ng_cost_path,
    #     candidate_gens,
    # )
    #
    # The version below assumes your local DataProcessing supports the
    # additional keyword arguments used in the attached driver_naerm.py.
    data_processing_object.load_gen_data(
        bus_data_path=bus_data_path,
        cost_data_path=cost_data_path,
        # ng_cost_path=ng_cost_path,
        candidate_gens=candidate_gens,
        save_csv=False,
        candidate_gen_csv_path=gen_csv_path,
        candidate_storage_csv_path=storage_csv_path,
        candidate_branch_csv_path=branch_csv_path,
    )

    print("Candidate generation cost data loaded.")

###############################################################################
# BUILD GTEP MODEL
###############################################################################

# Check specific bad generator
gname = "MilnerButteL_9aab36f6"
for i, rep in enumerate(data_object.representative_data):
    if gname in rep.data["elements"]["generator"]:
        print("rep", i, gname, rep.data["elements"]["generator"][gname].get("p_max"))

# Check specific bad branch
bname = "10289_10558_1"
if bname in data_object.md.data["elements"]["branch"]:
    print("branch", bname, data_object.md.data["elements"]["branch"][bname])

sanitize_loaded_md(data_object.md, data_object.representative_data)

# Check specific bad generator
gname = "MilnerButteL_9aab36f6"
for i, rep in enumerate(data_object.representative_data):
    if gname in rep.data["elements"]["generator"]:
        print("rep", i, gname, rep.data["elements"]["generator"][gname].get("p_max"))

# Check specific bad branch
bname = "10289_10558_1"
if bname in data_object.md.data["elements"]["branch"]:
    print("branch", bname, data_object.md.data["elements"]["branch"][bname])

print("Creating ExpansionPlanningModel...")

mod_object = ExpansionPlanningModel(
    data=data_object,
    cost_data=data_processing_object,
)

mod_object.config["include_investment"] = False
mod_object.config["include_commitment"] = False
mod_object.config["include_redispatch"] = True
mod_object.config["scale_loads"] = False
mod_object.config["transmission"] = True
mod_object.config["storage"] = False
mod_object.config["flow_model"] = "transport"
mod_object.config["advanced_hydro"] = False

# Save config
config_csv_path = f"{dir_name}/model_config.csv"
with open(config_csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["config_key", "config_value", "value_type"])
    for key, value in sorted(mod_object.config.items()):
        writer.writerow([key, repr(value), type(value).__name__])

print(f"Saved model configuration to: {config_csv_path}")

print("Building model...")
mod_object.create_model()
print("Model created.")

###############################################################################
# GDP TRANSFORMATION
###############################################################################

print("Applying GDP transformation...")
pyo.TransformationFactory("gdp.bigm").apply_to(mod_object.model)
print("Model transformed.")

###############################################################################
# SOLVER
###############################################################################

solver = "gurobi"
# solver = "highs"
solver = "xpress"

if solver == "gurobi":
    opt = pyo.SolverFactory("gurobi")
elif solver == "highs":
    opt = pyo.SolverFactory("highs")
elif solver == "xpress":
    opt = pyo.SolverFactory("xpress")
    # If using xpress, uncomment:
    xpress.init("naerm_xpauth.xpr")
else:
    raise ValueError(f"Unsupported solver: {solver}")

print(f"Solving with {solver}...")

mod_object.results = opt.solve(
    mod_object.model,
    tee=True,
)

print(mod_object.results)

###############################################################################
# SAVE RESULTS
###############################################################################

# Only execute if solve succeeds far enough for solution extraction
try:
    sol_object = ExpansionPlanningSolution(data_path)
    sol_object.save_results_in_json_files(mod_object, dir_name)

    # Plot generation mix if desired
    plot_type = "all"

    case_json = "dispatchables"
    sol_object.create_plots(case_json, dir_name, data_path, plot_type)

    case_json = "renewables"
    sol_object.create_plots(case_json, dir_name, data_path, plot_type)

    case_json = "combined"
    sol_object.create_plots(case_json, dir_name, data_path, plot_type)

    # Representative day stackgraph examples for 2030
    day_hour_list = [
        ("2030-07-05 00:00", 18),
        ("2030-10-14 00:00", 4),
    ]
    sol_object.create_stackgraph_and_metrics(dir_name, rep_days, day_hour_list)

except Exception as e:
    print("Post-processing / solution export failed.")
    print(f"Error: {e}")

pass