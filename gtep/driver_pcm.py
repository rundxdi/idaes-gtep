#!/usr/bin/env python3

from __future__ import annotations

import csv
import logging
from logging.handlers import RotatingFileHandler
import math
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo

from gtep.gtep_model import ExpansionPlanningModel
from gtep.gtep_data import ExpansionPlanningData


# ---------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------

LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "driver_pcm.log"

logger = logging.getLogger("gtep.driver_pcm")
logger.setLevel(logging.INFO)
logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)

sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(sh)


# ---------------------------------------------------------------------
# config
# ---------------------------------------------------------------------

DATA_PATH = Path("~/2030_pcm_case/base_case_pcm_2030").expanduser()
REP_DAYS = [
    "2030-01-28 00:00:00",
    "2030-04-23 00:00:00",
    "2030-07-05 00:00:00",
    "2030-10-14 00:00:00",
]
REP_WEIGHTS = [1, 1, 1, 1]

OUTPUT_DIR = Path("./GTEP_2030_run")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOLVER_NAME = "gurobi"
# SOLVER_NAME = "highs"


# ---------------------------------------------------------------------
# diagnostics / sanitation helpers
# ---------------------------------------------------------------------

def log_case_path_info(data_path: Path):
    logger.info("Using data path: %s", data_path)
    logger.info("data path exists: %s", data_path.exists())
    logger.info("gen.csv exists: %s", (data_path / "gen.csv").exists())
    logger.info("branch.csv exists: %s", (data_path / "branch.csv").exists())
    logger.info("storage.csv exists: %s", (data_path / "storage.csv").exists())
    logger.info("simulation_objects.csv exists: %s", (data_path / "simulation_objects.csv").exists())
    logger.info("timeseries_pointers.csv exists: %s", (data_path / "timeseries_pointers.csv").exists())


def sanitize_loaded_md(md, representative_data=None):
    elems = md.data["elements"]

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

    for br, b in elems.get("branch", {}).items():
        for key, fallback in [
            ("resistance", 0.0),
            ("reactance", 1e-6),
            ("charging_susceptance", 0.0),
            ("rating_long_term", 100.0),
            ("rating_short_term", 100.0),
            ("rating_emergency", 100.0),
        ]:
            if key in b:
                try:
                    if b[key] is None or (isinstance(b[key], float) and math.isnan(b[key])):
                        b[key] = fallback
                except Exception:
                    b[key] = fallback

    for br, b in elems.get("dc_branch", {}).items():
        for key, fallback in [
            ("rating_long_term", 100.0),
            ("rating_short_term", 100.0),
            ("rating_emergency", 100.0),
        ]:
            if key in b:
                try:
                    if b[key] is None or (isinstance(b[key], float) and math.isnan(b[key])):
                        b[key] = fallback
                except Exception:
                    b[key] = fallback

    for s, st in elems.get("storage", {}).items():
        for key, fallback in [
            ("max_discharge_rate", 0.0),
            ("min_discharge_rate", 0.0),
            ("max_charge_rate", 0.0),
            ("min_charge_rate", 0.0),
            ("energy_capacity", 0.0),
            ("initial_state_of_charge", 0.5),
            ("minimum_state_of_charge", 0.0),
            ("charge_efficiency", 1.0),
            ("discharge_efficiency", 1.0),
            ("retention_rate_60min", 1.0),
            ("capital_multiplier", 1.0),
            ("extension_multiplier", 0.0),
            ("ramp_up_input_60min", 0.0),
            ("ramp_up_output_60min", 0.0),
            ("ramp_down_input_60min", 0.0),
            ("ramp_down_output_60min", 0.0),
            ("charge_cost", 0.0),
            ("discharge_cost", 0.0),
            ("investment_cost", 0.0),
        ]:
            if key in st:
                try:
                    if st[key] is None or (isinstance(st[key], float) and math.isnan(st[key])):
                        st[key] = fallback
                except Exception:
                    st[key] = fallback
            else:
                st[key] = fallback

        # backward compatibility if converter/storage CSV ever used typo
        if "min_charge_rage" in st and "min_charge_rate" not in st:
            try:
                st["min_charge_rate"] = safe_float(st["min_charge_rage"], 0.0)
            except Exception:
                st["min_charge_rate"] = 0.0

    if representative_data is not None:
        for rep in representative_data:
            for gen, g in rep.data["elements"].get("generator", {}).items():
                if g.get("generator_type") == "renewable":
                    pmax = g.get("p_max")

                    if isinstance(pmax, dict) and "values" in pmax:
                        vals = pmax.get("values", [])
                        fallback = 0.0

                        base_g = elems["generator"].get(gen, {})
                        base_pmax = base_g.get("p_max", 0.0)

                        if isinstance(base_pmax, (int, float)):
                            try:
                                fallback = max(float(base_pmax), 0.0)
                            except Exception:
                                fallback = 0.0
                        elif isinstance(base_pmax, dict):
                            if "reference_value" in base_pmax and base_pmax["reference_value"] is not None:
                                fallback = max(float(base_pmax["reference_value"]), 0.0)
                            elif "values" in base_pmax and base_pmax["values"]:
                                try:
                                    good = [
                                        float(v) for v in base_pmax["values"]
                                        if v is not None and not (isinstance(v, float) and math.isnan(v))
                                    ]
                                    fallback = max(good) if good else 0.0
                                except Exception:
                                    fallback = 0.0

                        cleaned = []
                        for v in vals:
                            if v is None or (isinstance(v, float) and math.isnan(v)):
                                cleaned.append(fallback)
                            else:
                                cleaned.append(max(float(v), 0.0))
                        g["p_max"]["values"] = cleaned

                    elif pmax is None or (isinstance(pmax, float) and math.isnan(pmax)):
                        g["p_max"] = 0.0

    return md


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def export_bad_inputs(data_object, outdir: Path):
    bad_renew_rows = []
    for gen in data_object.md.data["elements"]["generator"]:
        g = data_object.md.data["elements"]["generator"][gen]
        if g.get("generator_type") != "renewable":
            continue

        for i, rep in enumerate(data_object.representative_data):
            try:
                pmax = rep.data["elements"]["generator"][gen].get("p_max")
                if not isinstance(pmax, dict):
                    bad_renew_rows.append({
                        "GEN UID": gen,
                        "rep_period": i,
                        "issue": f"p_max not dict: {type(pmax)}",
                    })
                    continue

                vals = pmax.get("values")
                if vals is None:
                    bad_renew_rows.append({
                        "GEN UID": gen,
                        "rep_period": i,
                        "issue": "missing values field",
                    })
                    continue

                if len(vals) == 0:
                    bad_renew_rows.append({
                        "GEN UID": gen,
                        "rep_period": i,
                        "issue": "empty values",
                    })
                    continue

                if all(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals):
                    bad_renew_rows.append({
                        "GEN UID": gen,
                        "rep_period": i,
                        "issue": "all NaN/None",
                    })
                    continue

                if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals):
                    bad_renew_rows.append({
                        "GEN UID": gen,
                        "rep_period": i,
                        "issue": "partial NaN/None",
                    })
            except Exception as e:
                bad_renew_rows.append({
                    "GEN UID": gen,
                    "rep_period": i,
                    "issue": f"error: {e}",
                })

    bad_renew_df = pd.DataFrame(bad_renew_rows)
    bad_renew_path = outdir / "bad_renewable_pmax.csv"
    bad_renew_df.to_csv(bad_renew_path, index=False)
    logger.info("Wrote renewable pmax diagnostics to %s", bad_renew_path)

    bad_branch_rows = []
    for br, b in data_object.md.data["elements"].get("branch", {}).items():
        for key in ["rating_long_term", "rating_short_term", "rating_emergency"]:
            v = b.get(key)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                bad_branch_rows.append({
                    "branch": br,
                    "field": key,
                    "from_bus": b.get("from_bus"),
                    "to_bus": b.get("to_bus"),
                    "value": v,
                })

    bad_branch_df = pd.DataFrame(bad_branch_rows)
    bad_branch_path = outdir / "bad_branch_ratings.csv"
    bad_branch_df.to_csv(bad_branch_path, index=False)
    logger.info("Wrote branch rating diagnostics to %s", bad_branch_path)


def log_converter_fill_reports(data_path: Path):
    md = data_path / "metadata"

    for fname in [
        "branch_rating_fills.csv",
        "dc_branch_rating_fills.csv",
        "renewable_timeseries_fallbacks.csv",
    ]:
        fpath = md / fname
        if fpath.exists():
            try:
                df = pd.read_csv(fpath)
                logger.info("Converter fill report %s rows=%s", fname, len(df))
                if len(df) > 0:
                    logger.info("First few rows of %s:\n%s", fname, df.head(10).to_string(index=False))
            except Exception as e:
                logger.warning("Could not read %s: %s", fpath, e)


def main():
    log_case_path_info(DATA_PATH)

    logger.info("Creating ExpansionPlanningData object...")
    data_object = ExpansionPlanningData(
        stages=1,
        num_reps=1,
        len_reps=1,
        num_commit=1,
        num_dispatch=1,
    )

    logger.info("Loading Prescient case...")
    data_object.load_prescient(
        str(DATA_PATH),
        representative_dates=REP_DAYS,
        representative_weights=REP_WEIGHTS,
        options_dict={
            "data_path": str(DATA_PATH),
            "num_days": 365,
            "ruc_horizon": 36,
            "start_date": "2030-01-01",
            "sced_frequency_minutes": 60,
        },
    )

    logger.info("Prescient case loaded successfully.")
    logger.info("data_object.num_reps = %s", getattr(data_object, "num_reps", None))
    logger.info("len(REP_DAYS) = %s", len(REP_DAYS))
    logger.info("len(REP_WEIGHTS) = %s", len(REP_WEIGHTS))
    logger.info("len(representative_data) = %s", len(data_object.representative_data))

    if len(data_object.representative_data) != len(REP_DAYS):
        logger.warning(
            "Representative period count mismatch: built %s, expected %s. Trimming.",
            len(data_object.representative_data),
            len(REP_DAYS),
        )
        data_object.representative_data = data_object.representative_data[: len(REP_DAYS)]

    logger.info("Final representative_data length = %s", len(data_object.representative_data))

    for i, rep in enumerate(data_object.representative_data):
        try:
            keys = rep.data["system"]["time_keys"]
            logger.info("rep %s start=%s end=%s", i, keys[0], keys[-1])
        except Exception:
            logger.info("rep %s unable to report time_keys", i)

    log_converter_fill_reports(DATA_PATH)

    export_bad_inputs(data_object, OUTPUT_DIR)

    sanitize_loaded_md(data_object.md, data_object.representative_data)
    logger.info("Applied sanitize_loaded_md()")

    export_bad_inputs(data_object, OUTPUT_DIR)

    bad_gen = "MilnerButteL_9aab36f6"
    if bad_gen in data_object.md.data["elements"]["generator"]:
        for i, rep in enumerate(data_object.representative_data):
            if bad_gen in rep.data["elements"]["generator"]:
                logger.info(
                    "rep %s %s p_max=%s",
                    i,
                    bad_gen,
                    rep.data["elements"]["generator"][bad_gen].get("p_max"),
                )

    bad_branch = "10289_10558_1"
    if bad_branch in data_object.md.data["elements"]["branch"]:
        logger.info(
            "branch %s data=%s",
            bad_branch,
            data_object.md.data["elements"]["branch"][bad_branch],
        )

    logger.info("Creating ExpansionPlanningModel...")
    mod_object = ExpansionPlanningModel(
        data=data_object,
        cost_data=None,
    )

    mod_object.config["include_investment"] = False
    mod_object.config["include_commitment"] = False
    mod_object.config["include_redispatch"] = True
    mod_object.config["scale_loads"] = False
    mod_object.config["transmission"] = True
    mod_object.config["storage"] = True
    mod_object.config["flow_model"] = "transport"
    mod_object.config["advanced_hydro"] = False

    config_csv_path = OUTPUT_DIR / "model_config.csv"
    with open(config_csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config_key", "config_value", "value_type"])
        for key, value in sorted(mod_object.config.items()):
            writer.writerow([key, repr(value), type(value).__name__])
    logger.info("Saved model configuration to %s", config_csv_path)

    logger.info("Building model...")
    mod_object.create_model()
    logger.info("Model created.")

    logger.info("Applying GDP big-M transformation...")
    pyo.TransformationFactory("gdp.bigm").apply_to(mod_object.model)
    logger.info("GDP transformation complete.")

    # from pyomo.contrib.iis import write_iis

    # # 'instance' is your built/instantiated Pyomo model
    # # 'solver' can be "gurobi", "cplex", or "xpress"
    # write_iis(mod_object.model, iis_file_name="model_iis.ilp", solver="gurobi")

    logger.info("Selecting solver: %s", SOLVER_NAME)
    if SOLVER_NAME == "gurobi":
        opt = pyo.SolverFactory("gurobi")
    elif SOLVER_NAME == "highs":
        opt = pyo.SolverFactory("highs")
    else:
        raise ValueError(f"Unsupported solver: {SOLVER_NAME}")

    logger.info("Solving model...")
    mod_object.results = opt.solve(mod_object.model, tee=True)
    logger.info("Solve complete.")
    logger.info("%s", mod_object.results)


if __name__ == "__main__":
    main()