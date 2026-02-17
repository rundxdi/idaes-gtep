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

from gtep.gtep_model import ExpansionPlanningModel
from gtep.gtep_data import ExpansionPlanningData
from gtep.gtep_solution import ExpansionPlanningSolution
from pyomo.core import TransformationFactory
from pyomo.core.base.expression import ScalarExpression, IndexedExpression
from pyomo.environ import SolverFactory
import gc
import sys
import os
import json
import psutil
from gtep.gtep_data_processing import DataProcessing

gc.disable()

if len(sys.argv) > 1:
    num_investment_periods = int(sys.argv[1])
    num_representative_periods = int(sys.argv[2])
    length_representative_periods = int(sys.argv[3])
    num_commitment_periods = int(sys.argv[4])
    num_dispatch_periods = int(sys.argv[5])
    thermal_investment = bool(sys.argv[6])
    renewable_investment = bool(sys.argv[7])
    storage_investment = bool(sys.argv[8])
    flow_model = sys.argv[9]
    unit_commitment = bool(sys.argv[10])
    dispatch = bool(sys.argv[11])
    log_folder = sys.argv[12]
else:
    pass

if not os.path.exists(log_folder):
    os.makedirs(log_folder)

with open(log_folder + "/input.log", "w") as fil:
    input_str = "".join([str(i) for i in sys.argv])
    fil.write(input_str)

data_path = "./gtep/data/Texas_2000"
# data_path = "./gtep/data/5bus"
data_object = ExpansionPlanningData()
data_object.load_prescient(data_path, length_representative_periods)

# load_scaling_path = data_path + "/ERCOT-Adjusted-Forecast.xlsb"
# data_object.import_load_scaling(load_scaling_path)
# outage_path = data_path + "/may_20.csv"
# data_object.import_outage_data(outage_path)

bus_data_path = "./gtep/data/costs/Bus_data_gen_weights_mappings.csv"
cost_data_path = "./gtep/data/costs/2022_v3_Annual_Technology_Baseline_Workbook_Mid-year_update_2-15-2023_Clean.xlsx"
candidate_gens = [
    "Natural Gas_CT",
    "Natural Gas_FE",
    "Solar - Utility PV",
    "Land-Based Wind",
]

data_processing_object = DataProcessing()
data_processing_object.load_gen_data(
    bus_data_path=bus_data_path,
    cost_data_path=cost_data_path,
    candidate_gens=candidate_gens,
    save_csv=False,
)


# data_object.texas_case_study_updates(data_path)

## Change num_reps from 4 to 5 to include extreme days

mod_object = ExpansionPlanningModel(
    stages=num_investment_periods,
    data=data_object,
    cost_data=data_processing_object,
    num_reps=num_representative_periods,
    len_reps=length_representative_periods,
    num_commit=num_commitment_periods,
    num_dispatch=num_dispatch_periods,
    duration_dispatch=60,
)
# print(mod_object.data.data["elements"]["generator"]["1"])
# import sys
# sys.exit()
mod_object.config["include_investment"] = True
mod_object.config["scale_loads"] = False
mod_object.config["scale_texas_loads"] = False
mod_object.config["transmission"] = True
mod_object.config["storage"] = storage_investment
mod_object.config["flow_model"] = flow_model

# mod_object.config["thermal_investment"] = thermal_investment
# mod_object.config["renewable_investment"] = renewable_investment
mod_object.create_model()

with open(log_folder + "/timer.log", "a") as fil:
    mod_object.timer.toc("Model Created", ostream=fil)

with open(log_folder + "/memory.log", "a") as fil:
    pid = os.getpid()
    process = psutil.Process(pid)
    mem_info = process.memory_info()
    used_bytes = mem_info.rss
    used_gb = used_bytes / (1024**3)
    fil.write(f"Model created \n Total used memory: {used_gb:.2f} GiB\n")


# TransformationFactory("gdp.bound_pretransformation").apply_to(mod_object.model)
# mod_object.timer.toc("double horrible")

TransformationFactory("gdp.bigm").apply_to(mod_object.model)
with open(log_folder + "/timer.log", "a") as fil:
    mod_object.timer.toc("Model Transformed", ostream=fil)

with open(log_folder + "/memory.log", "a") as fil:
    pid = os.getpid()
    process = psutil.Process(pid)
    mem_info = process.memory_info()
    used_bytes = mem_info.rss
    used_gb = used_bytes / (1024**3)
    fil.write(f"Model transformed \n Total used memory: {used_gb:.2f} GiB\n")

# import sys
# sys.exit()

opt = SolverFactory("gurobi_direct_v2")
mod_object.timer.toc(
    "let's start to solve -- this is really the start of the handoff to gurobi"
)

from pyomo.contrib.iis import iis

iis.write_iis(mod_object.model, log_folder + "/infeasible_model.ilp")


mod_object.results = opt.solve(
    mod_object.model,
    tee=True,
    solver_options={
        "LogFile": log_folder + "/gurobi.log",
        "MIPGap": 0.01,
        "Threads": 2,
    },
)

# mod_object.model.write('bad_sol.sol')
# mod_object.results = opt.solve(mod_object.model)

# import sys
# sys.exit()
with open(log_folder + "/timer.log", "a") as fil:
    mod_object.timer.toc("Model Solved", ostream=fil)
with open(log_folder + "/memory.log", "a") as fil:
    pid = os.getpid()
    process = psutil.Process(pid)
    mem_info = process.memory_info()
    used_bytes = mem_info.rss
    used_gb = used_bytes / (1024**3)
    fil.write(f"Model solved \n Total used memory: {used_gb:.2f} GiB\n")

import pyomo.environ as pyo
import pyomo.gdp as gdp

valid_names = ["Inst", "Oper", "Disa", "Ext", "Ret"]
# thermal_names = ["genInst", "genOper", "genDisa", "genExt", "genRet"]
renewable_investments = {}
dispatchable_investments = {}
load_shed = {}
for var in mod_object.model.component_objects(pyo.Var, descend_into=True):
    for index in var:
        if "Shed" in var.name:
            if pyo.value(var[index]) >= 0.001:
                load_shed[var.name + "." + str(index)] = pyo.value(var[index])
        for name in valid_names:
            if name in var.name:
                if pyo.value(var[index]) >= 0.001:
                    renewable_investments[var.name + "." + str(index)] = pyo.value(
                        var[index]
                    )
for var in mod_object.model.component_objects(gdp.Disjunct, descend_into=True):
    for index in var:
        for name in valid_names:
            if name in var.name:
                if pyo.value(var[index].indicator_var) == True:
                    dispatchable_investments[var.name + "." + str(index)] = pyo.value(
                        var[index].indicator_var
                    )

costs = {}
for exp in mod_object.model.component_objects(pyo.Expression, descend_into=True):
    if "Cost" in exp.name or "cost" in exp.name:
        if type(exp) is ScalarExpression:
            costs[exp.name] = pyo.value(exp)
        if type(exp) is IndexedExpression:
            for e in exp:
                costs[exp[e].name] = pyo.value(exp[e])

folder_name = log_folder
renewable_investment_name = folder_name + "/renewable_investments.json"
dispatchable_investment_name = folder_name + "/dispatchable_investments.json"
load_shed_name = folder_name + "/load_shed.json"
costs_name = folder_name + "/costs.json"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

with open(renewable_investment_name, "w") as fil:
    json.dump(renewable_investments, fil)
with open(dispatchable_investment_name, "w") as fil:
    json.dump(dispatchable_investments, fil)
with open(load_shed_name, "w") as fil:
    json.dump(load_shed, fil)
with open(costs_name, "w") as fil:
    json.dump(costs, fil)

with open(log_folder + "timer.log", "a") as fil:
    mod_object.timer.toc(
        "we've dumped; get everybody and the stuff together", ostream=fil
    )

with open(log_folder + "/memory.log", "a") as fil:
    pid = os.getpid()
    process = psutil.Process(pid)
    mem_info = process.memory_info()
    used_bytes = mem_info.rss
    used_gb = used_bytes / (1024**3)
    fil.write(f"Model dumped \n Total used memory: {used_gb:.2f} GiB\n")
