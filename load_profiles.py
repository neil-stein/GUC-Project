import os, io, asyncio, aioboto3, polars as pl
from botocore.config import Config
from botocore import UNSIGNED
from datetime import datetime
from tqdm.auto import tqdm

BUCKET = "oedi-data-lake"
BASE = "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024/resstock_tmy3_release_2/timeseries_individual_buildings/by_state"
REGION = "us-west-2"  # OEDI region

COLUMNS = [
    "timestamp",
    "out.electricity.net.energy_consumption",   # adjusted to use net electricity, previously was using total
    "out.electricity.heating.energy_consumption",
    "out.electricity.heating_hp_bkup.energy_consumption",
    "out.electricity.cooling.energy_consumption",
    "out.natural_gas.total.energy_consumption",
    "out.natural_gas.heating.energy_consumption",
]
RENAME = {
    "electricity.total": "out.electricity.net.energy_consumption",    # remapped to net electricity, previously was using total
    "electricity.heating": "out.electricity.heating.energy_consumption",
    "electricity.secondary_heating": "out.electricity.heating_hp_bkup.energy_consumption",
    "electricity.cooling": "out.electricity.cooling.energy_consumption",
    "natural_gas.total": "out.natural_gas.total.energy_consumption",
    "natural_gas.heating": "out.natural_gas.heating.energy_consumption",
}

select_by_id = pl.selectors.matches(r"^\-?[0-9]+$|^.*_whole$")
select_quants = pl.selectors.matches(r"total_load|total_cost")
select_labels = pl.selectors.temporal()|pl.selectors.matches("id")|pl.selectors.string()
select_costs = pl.selectors.ends_with("_cost").exclude("total_cost")

# logical metric keys from RENAME, e.g. "electricity.total"
LOGICAL_KEYS = list(RENAME.keys())
# map "electricity.total" -> ("electricity", "total")
FUEL_ENDUSE = {k: tuple(k.split(".", 1)) for k in LOGICAL_KEYS}

# for mapping to metadata weight column names
FUEL_WEIGHT_PREFIX = {
    "electricity": "elec",
    "natural_gas": "gas",
}

cfg = Config(
    region_name=REGION,
    signature_version=UNSIGNED,
    retries={"max_attempts": 5, "mode": "adaptive"},
    max_pool_connections=256,          # allow many parallel sockets
    read_timeout=120, connect_timeout=10,
)

def key_for(state, bid, upgrade):
    return f"{BASE}/upgrade={upgrade}/state={state}/{bid}-{upgrade}.parquet"

async def fetch_one(s3, key, logicals):
    # stream object into memory
    obj = await s3.get_object(Bucket=BUCKET, Key=key)
    body = await obj["Body"].read()
    buf = io.BytesIO(body)
    # parse only needed cols for all logicals; skip ts work unless first
    cols_to_read = ['timestamp'] + [RENAME[logical] for logical in logicals]
    df = pl.read_parquet(buf, columns=cols_to_read)
    ts = (
        df.select("timestamp")
          .with_columns(pl.col("timestamp").dt.offset_by("-15m"))
          .with_columns(pl.col("timestamp").dt.replace(year=2025))
          .with_columns(
              pl.col("timestamp")
                .dt.replace_time_zone("UTC")
                .dt.strftime("%Y-%m-%dT%H:%M:%S%z")
          )
          .sort("timestamp")
    )
    vals = df.drop("timestamp", strict=False).rename({RENAME[logical]:logical for logical in logicals})
    return ts, vals

CHUNK = 1000  # save every 1000 ids


async def fetch_building_metric(s3, state, bid, upgrade, logicals, sem):
    """
    Fetch one building's parquet from S3, return (timestamp_df, building_value_df)
    where building_value_df has columns for each logical metric.
    """
    async with sem:
        key = key_for(state, bid, upgrade)
        ts, vals = await fetch_one(s3, key, logicals)
    # Rename vals columns to include bid, e.g. "electricity.total" -> "electricity.total|123"
    renamed_vals = vals.rename({logical: f"{logical}|{bid}" for logical in logicals})
    return ts, renamed_vals


async def get_load_profiles(
    metadata: pl.DataFrame,
    fuel: list[str] | str = ["electricity"],
    end_use: str = "total",
    concurrency: int = 64,
    outdir: str = "outputs/load_profiles",
) -> dict[str, pl.DataFrame]:
    """
    Final version:
    - Fetch ALL fuels together in one S3 call per building
    - Split and save to separate canonical files per fuel
    - Per-combo output written under utility folders with fuel subfolders
    - Return value is: { f"{fuel}_{end_use}": raw_df_for_that_fuel }
    """

    # Normalize fuel to list if needed
    if isinstance(fuel, str):
        fuel = [fuel]
    
    # Validate fuels
    for f in fuel:
        logical = f"{f}.{end_use}"
        if logical not in RENAME.keys():
            raise ValueError(f"Invalid metric {logical}")

    sem = asyncio.Semaphore(concurrency)

    # Build logicals list for this run
    logicals = [f"{f}.{end_use}" for f in fuel]

    group_cols = ["state", "elec_utility", "gas_utility", "upgrade"]

    # Output container: per fuel, per upgrade
    output_by_fuel: dict[str, dict[str, pl.DataFrame]] = {
        f"{f}_{end_use}": {} for f in fuel
    }

    # Iterate over combos
    combo_iter = metadata.group_by(group_cols, maintain_order=True)
    for (state, eu, gu, upgrade), meta_g in tqdm(
        combo_iter,
        desc=f"Combos ({','.join(fuel)}.{end_use})",
        total=combo_iter.agg().height,
        leave=True,
    ):
        ids = meta_g["bldg_id"].unique().sort().to_list()

        # ---------- check which building IDs are missing across all fuels ----------
        # Load canonical files for each fuel to find which IDs are missing
        canonical_dfs_by_fuel = {}
        missing_ids_by_fuel = {}
        
        for f in fuel:
            canonical_dir = os.path.join(outdir, state, "_canonical", str(upgrade), f)
            os.makedirs(canonical_dir, exist_ok=True)
            canonical_path = os.path.join(canonical_dir, f"{end_use}.parquet")
            
            if os.path.exists(canonical_path):
                canonical_dfs_by_fuel[f] = pl.read_parquet(canonical_path)
                # Extract building IDs: columns are like "123" after splitting from "electricity.total|123"
                existing_ids = {c for c in canonical_dfs_by_fuel[f].columns if c.isdigit()}
            else:
                canonical_dfs_by_fuel[f] = None
                existing_ids = set()
            
            missing_ids_by_fuel[f] = [bid for bid in ids if str(bid) not in existing_ids]

        # Find union of all missing IDs (we'll fetch once for all fuels)
        all_missing = set()
        for missing_list in missing_ids_by_fuel.values():
            all_missing.update(missing_list)
        all_missing = sorted(list(all_missing))

        # ---------- fetch missing IDs (async + chunked) - ONCE for all fuels ----------
        if all_missing:
            print(f"{len(all_missing)} missing IDs for combo, fetching...")
            async with aioboto3.Session().client("s3", config=cfg) as s3:
                # Get timestamp reference from any existing canonical
                ts_ref = None
                for f in fuel:
                    if canonical_dfs_by_fuel[f] is not None:
                        ts_ref = canonical_dfs_by_fuel[f].select("timestamp")
                        break

                fetch_pbar = tqdm(
                    total=len(all_missing),
                    desc=f"{state}-{eu}_{gu}-{upgrade} downloading",
                    leave=False,
                )

                for i in range(0, len(all_missing), CHUNK):
                    batch = all_missing[i : i + CHUNK]
                    tasks = [
                        asyncio.create_task(
                            fetch_building_metric(
                                s3,
                                state,
                                bid,
                                upgrade,
                                logicals,
                                sem,
                            )
                        )
                        for bid in batch
                    ]

                    cols_list = []

                    for fut in asyncio.as_completed(tasks):
                        try:
                            ts, col = await fut
                        except Exception as e:
                            print("skip:", e)
                            continue

                        # Timestamp sanity check
                        if ts_ref is None:
                            ts_ref = ts
                        else:
                            if not ts_ref["timestamp"].equals(ts["timestamp"]):
                                raise ValueError(
                                    f"Timestamp misalignment for state={state}, "
                                    f"upgrade={upgrade}, fuels={fuel}"
                                )

                        cols_list.append(col)
                        fetch_pbar.update(1)

                    if not cols_list:
                        continue

                    new_wide = pl.concat(cols_list, how="horizontal")
                    # Add timestamp back (it was stripped before concat)
                    new_wide = pl.concat([ts_ref, new_wide], how="horizontal")

                    # ---------- split fetched data by fuel (accumulate, don't write yet) ----------
                    for f in fuel:
                        # Extract columns for this fuel: "electricity.total|123", "electricity.total|456", etc
                        fuel_col_pattern = f"{f}.{end_use}|"
                        fuel_cols_in_batch = [c for c in new_wide.columns if c.startswith(fuel_col_pattern)]
                        
                        if not fuel_cols_in_batch:
                            continue
                        
                        # Extract just this fuel's data (with timestamp)
                        fuel_batch = new_wide.select(["timestamp"] + fuel_cols_in_batch)
                        # Rename from "electricity.total|123" to "123" using rsplit for safety
                        rename_map = {col: col.rsplit("|", 1)[-1] for col in fuel_cols_in_batch}
                        fuel_batch = fuel_batch.rename(rename_map)
                        
                        if canonical_dfs_by_fuel[f] is None:
                            # First batch for this fuel: just use this batch
                            canonical_dfs_by_fuel[f] = fuel_batch
                        else:
                            # Append new columns to existing canonical (join on timestamp to combine building IDs)
                            canonical_dfs_by_fuel[f] = canonical_dfs_by_fuel[f].join(
                                fuel_batch, on="timestamp", how="outer", coalesce=True
                            )

                fetch_pbar.close()

                # ---------- write canonical files once after all batches complete ----------
                for f in fuel:
                    if canonical_dfs_by_fuel[f] is None:
                        continue
                    
                    # Sort building ID columns numerically (filter to only numeric column names)
                    numeric_cols = [c for c in canonical_dfs_by_fuel[f].columns if c != "timestamp" and c.isdigit()]
                    cols_sorted = ["timestamp"] + sorted(numeric_cols, key=int)
                    canonical_dfs_by_fuel[f] = canonical_dfs_by_fuel[f].select(cols_sorted)

                    # Write canonical file (once, after all batches)
                    canonical_dir = os.path.join(outdir, state, "_canonical", str(upgrade), f)
                    canonical_path = os.path.join(canonical_dir, f"{end_use}.parquet")
                    print(f"  Writing {len(numeric_cols)} columns for {f} to {canonical_path}...")
                    canonical_dfs_by_fuel[f].write_parquet(canonical_path)
                    import gc; gc.collect()  # Force garbage collection


        # ---------- per fuel processing: extract combo data and write to utility folders ----------
        for f in fuel:
            logical_key = f"{f}_{end_use}"
            canonical_df = canonical_dfs_by_fuel[f]
            
            if canonical_df is None:
                continue

            # Extract this fuel's columns and restrict to this combo's building IDs
            combo_bids = [str(bid) for bid in ids if str(bid) in canonical_df.columns]
            
            if not combo_bids:
                continue

            combo_raw = canonical_df.select(["timestamp"] + combo_bids)
            combo_raw = combo_raw.with_columns(pl.col("timestamp").str.to_datetime(time_zone="UTC"))

            # ---------- write combo-level output under utilities with fuel subfolder ----------
            combo_dir = os.path.join(
                outdir,
                state,
                eu if eu else "Other",
                gu if gu else "Other",
                str(upgrade),
                f,
            )
            if os.path.exists(os.path.join(combo_dir, f"{end_use}.parquet")):
                pass
            else:
                os.makedirs(combo_dir, exist_ok=True)
                combo_path = os.path.join(combo_dir, f"{end_use}.parquet")
                combo_raw.write_parquet(combo_path)

            # ---------- accumulate per-upgrade raw DF ----------
            up_key = upgrade
            output_by_fuel[logical_key][up_key] = combo_raw

    # Build return dict: per fuel, merge upgrades into "all"
    result = {}
    for logical_key, upgrade_dict in output_by_fuel.items():
        if not upgrade_dict:
            raise ValueError(f"No results produced for {logical_key}")

        try:
            dfs = [df for k, df in upgrade_dict.items()]
            if dfs:
                from functools import reduce
                merged_all = reduce(
                    lambda a,b: a.join(b, on="timestamp", how="inner").drop(pl.selectors.ends_with("_right")), dfs)
                all_cols = [c for c in merged_all.columns if c != "timestamp"]
                merged_all = merged_all.select("timestamp", *sorted(all_cols, key=lambda x: int(x)))
                upgrade_dict["all"] = merged_all
        except Exception as e:
            # don't fail the whole process for this bonus aggregation; warn and continue
            print(f"Warning: could not build 'all' merged dataframe for {logical_key}:", e)

        result[logical_key] = upgrade_dict["all"] if "all" in upgrade_dict else dfs[0] if dfs else None

    return result


def adjust_load_profile(load, metadata, weight_col='elec_weight', customer_count=None, segment_monthly_consumption=None):
    """
    Adjust load profiles by applying weights and scaling to match customer count and monthly consumption targets.
    
    Args:
        load (pl.DataFrame): Wide DataFrame with 'timestamp' column and building ID columns containing load values
        metadata (pl.DataFrame): DataFrame with 'bldg_id' and f'{system}_weight' columns
        customer_count (float, optional): Target customer count to scale weights to
        segment_monthly_consumption (dict, optional): Target monthly consumption {date_string: kwh_value}
    
    Returns:
        dict: {
            'adjusted_load': Wide DataFrame with adjusted load profiles,
            'monthly_summary': DataFrame with monthly aggregated consumption,
            'customer_count_scale_factor': Float,
            'monthly_scale_factors': Dict mapping months to their scale factors
        }
    """
    import pandas as pd
    
    # Validate inputs
    if load.is_empty() or metadata.is_empty():
        raise ValueError("Load and metadata DataFrames cannot be empty")
    
    if "bldg_id" not in metadata.columns or weight_col not in metadata.columns:
        raise ValueError(f"Metadata must have 'bldg_id' and {weight_col} columns")
    
    # Aggregate weights depending on system chosen using metadata weights, then sort by bldg_id
    meta = metadata.group_by("bldg_id", maintain_order=True).agg(pl.col(weight_col).sum())
    bldg_ids = load.select(select_by_id).columns
    if len(bldg_ids) > len(meta) != 0:
        raise ValueError("Load Dataframe numeric columns must be less than the number of building IDs in metadata DataFrame")
    else:
        meta = meta.filter(pl.col("bldg_id").cast(str).is_in(bldg_ids))
        weights = meta[weight_col]
    
    # Build explicit list of building ID column expressions in metadata order
    load = load.select(select_labels, select_quants, select_by_id, select_costs)
    
    # Original customer count (sum of weights)
    original_customer_count = weights.sum()
    
    # Step 1: Apply customer count scaling if provided
    if customer_count is not None:
        customer_count_scale = customer_count / original_customer_count
        weights = weights * customer_count_scale
    else:
        customer_count_scale = 1.0

    meta = meta.with_columns(weights.alias(weight_col))
    metadata = metadata.join(meta, on="bldg_id", how="left").with_columns(
        pl.when(pl.col("bldg_id").cast(str).is_in(bldg_ids))
        .then(pl.col(weight_col)*(weights/(pl.col(weight_col).sum().over("bldg_id"))))
    ).drop(weight_col+"_right")

    # Step 2: Apply weights to load profiles
    weighted_load = load.with_columns(
        [pl.col(str(bid)) * weight for bid, weight in zip(bldg_ids, weights)]
    )
    
    if 'timestamp' in weighted_load.columns and 'rateName' not in weighted_load.columns:
        # Calculate monthly aggregation
        monthly_agg = weighted_load.group_by_dynamic("timestamp", every="1mo").agg(
            pl.sum_horizontal(select_by_id).sum().alias("kwh")
        ).sort(select_labels)
    
    # Step 3: Apply monthly consumption scaling if provided
    if segment_monthly_consumption is not None:
        # Convert target to DataFrame
        target_df = pl.DataFrame({
            "timestamp": [pd.to_datetime(k, utc=True) for k in segment_monthly_consumption.keys()],
            "target_kwh": list(segment_monthly_consumption.values())
        }).sort("timestamp")
        
        # Join with current monthly totals
        comparison = monthly_agg.join(target_df, on="timestamp", how="left")
        
        # Calculate monthly scale factors
        comparison = comparison.with_columns(
            (pl.col("target_kwh") / pl.col("kwh")).alias("monthly_scale_factor")
        )
        
        monthly_scale_factors = {
            row[0]: row[2] for row in comparison.select(["timestamp", "kwh", "monthly_scale_factor"]).iter_rows()
        }
        
        # Apply monthly scaling to load profile
        adjusted_load = weighted_load.with_columns(
            pl.col("timestamp").dt.truncate("1mo").alias("month_key")
        ).join(
            comparison.select([pl.col("timestamp").alias("month_key"), "monthly_scale_factor"]),
            on="month_key",
            how="left"
        ).with_columns(
            [pl.col(str(bid)) * pl.col("monthly_scale_factor") for bid in bldg_ids]
        ).drop(["month_key", "monthly_scale_factor"])
        
        # Recalculate monthly summary
        monthly_summary = adjusted_load.group_by_dynamic("timestamp", every="1mo").agg(
            pl.sum_horizontal(select_by_id).sum().alias("kwh")
        ).sort("timestamp")
        detailed_summary = adjusted_load.select([
            pl.col("timestamp"),
            pl.sum_horizontal(select_by_id).alias("kwh")
        ])
    else:
        adjusted_load = weighted_load
        if 'timestamp' in weighted_load.columns and 'rateName' not in weighted_load.columns:
            monthly_summary = monthly_agg
            monthly_scale_factors = {row[0]: 1.0 for row in monthly_agg.select("timestamp").iter_rows()}
        else:
            monthly_agg = monthly_summary = monthly_scale_factors = "Unavailable"
        detailed_summary = adjusted_load.select(
            select_labels,select_quants,
            pl.sum_horizontal(select_by_id).alias("kwh")
        )
    
    return {
        'metadata': metadata,
        'adjusted_load': adjusted_load,
        'original_monthly': monthly_agg,
        'monthly_summary': monthly_summary,
        'detailed_summary': detailed_summary,
        'customer_count_scale_factor': customer_count_scale,
        'monthly_scale_factors': monthly_scale_factors
    }


if __name__=="__main__":
    from segments import get_segment, join_upgrade

    state = "NJ"
    utility = "Atlantic City Electric Co"
    base = get_segment(state, utility)  # + kwargs
    upgrade_ids = get_segment(
        state,
        utility,
        heating_type=["Gas"],
        county=["Ocean"],
        sample=0.5,
    )["bldg_id"].shuffle(313).to_list()
    joined = join_upgrade(
        base,
        {
            1: upgrade_ids[: len(upgrade_ids) // 2],
            2: upgrade_ids[len(upgrade_ids) // 2 :],
        },
    )
    # default: electricity.total and natural_gas.total
    test = asyncio.run(get_load_profiles(joined, fuel=["electricity", "natural_gas"], end_use="total"))
    print(test)
    pass
