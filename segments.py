import polars as pl

NON_METADATA_COLS = [
    "elec_utility",
    "elec_weight",
    "gas_utility",
    "gas_weight",
    "elec_county_coverage",
    "gas_county_coverage",
    "SP_elec",
    "SP_gas",
    "electricity.total",
    "natural_gas.total",
    "heating_pct",
    "cooling_pct",
    "hot_water_pct",
    "cooking_pct",
    "others_pct",
    "total_kwh"
]

def get_segment(state, utility="", gas_utility="", upgrade = 0, sample=None, **kwargs):
    """
    State is required, utility and gas_utility are recommended. The rest of the arguments are optional.

    Provide each segment with a *list* of options applicable to that segment.
    
    It works by finding the all entries that contain any string provided as input for a given segment. For
    example, providing "Electric" for heating_type will include all electric resistance heating systems
    (baseboards, boilers, etc). Below is a list of all available options for each segment, but there are important
    options that can be matched that don't appear in this list, such as the county or city, as well as a few others.

    "heating_type":         "Electric | Natural Gas | Propane | Oil | Shared | None", (k)\n
    "building_type":        "SF | Small MF | Large MF | Mobile", (e) \n
    "wh_type":              "Heat Pump | Electric | Natural Gas | Propane | Oil", (k)\n
    "area":                 "0-1499 | 1500-2499 | 2500-3999 | 4000+", (e)\n
    "income":               "Not Available | Low Income (<40,000) | Moderate Income (40,000-99,999) | High Income (>100,000)", (e)\n
    "climate_zone":         "Cold | Hot-Dry | Hot-Humid | Marine | Mixed-Dry | Mixed-Humid | Very Cold", (e)\n
    "vintage":              "<1980 | 1980-2000 | 2000-2010 | >2010", (e)\n
    "insulation_level":     "Good Insulation | Average Insulation | Poor Insulation", (e)\n
    "has_solar":            "Yes | No" (e)

    (e) = exact match only, (k) = keyword match

    """
    df = pl.read_parquet("outputs/RMI_2024_7.5_blk_grp_pop_cust_cnt.parquet").filter((pl.col("state").str.contains(state, literal=True)))

    # Handle utility as string or list
    if isinstance(utility, str):
        df_util = df.filter(pl.col("elec_utility").str.contains(utility, literal=True))
    else:
        df_util = df.filter(pl.col("elec_utility").str.contains_any(utility, literal=True))

    df_util = df_util.with_columns(
        bldg_id = pl.col("bldg_id").cast(str),
        zip_code = pl.col("zip_code").cast(str).str.zfill(5)
    )

    # Electric Utility name invalid for state
    if df_util.is_empty():
        raise Exception(f"Utility name invalid. Has to be one of the following: {', '.join(df['elec_utility'].drop_nulls().unique().sort())}")
    
    # Handle gas utility as string or list
    if isinstance(gas_utility, str):
        df_util = df_util.filter(pl.col("gas_utility").str.contains(gas_utility, literal=True))
    else:
        df_util = df_util.filter(pl.col("gas_utility").str.contains_any(gas_utility, literal=True))
    
    # Gas Utility name invalid for state or no buildings with the selected electric utility are served gas by the selected gas utility
    if df_util.is_empty():
        raise Exception(f"Gas utility name invalid or no buildings with the selected electric utility are served gas by the selected gas utility. Choose one of the following: {', '.join(df['gas_utility'].unique().sort().to_list())}?")

    # Filtering
    for col_name, values in kwargs.items():
        if not values:
            continue  # Skip empty values
        if col_name not in df.columns:
            raise Exception(f"{col_name} is not a valid name. Allowed names are {', '.join(df.columns)}")

        if all(~df[col_name].str.contains_any(values) if type(values[0]) is str else ~df[col_name].is_in(values)):
            raise Exception(f"One of {values} is not valid for {col_name}. Allowed options are {df[col_name].unique().sort().to_list()}")

        df_util = df_util.filter(pl.col(col_name).str.contains_any(values)) if type(values[0]) is str else df_util.filter(pl.col(col_name).is_in(values))
    
    if df_util.is_empty():
        raise Exception("Segment is too narrow for this utility and has no samples in ResStock.\nCheck sgements_by_utility folder for avilable segments for this utility.")

    if sample is not None:
        if sample <= 1:
            sample = int(sample * df_util.height)
        df_util = df_util.sample(n=min(sample,df_util.height),seed=313)

    if upgrade!=0:
        columns = df_util.columns
        df_up = pl.read_parquet(f"outputs/RMI_2024_7_blk_grp_pop_upgrade{upgrade}.parquet").filter((pl.col("state").str.contains(state, literal=True))).with_columns(
            bldg_id = pl.col("bldg_id").cast(str),
        )
        merge = [c for c in df.columns if c not in df_up.columns or c == "bldg_id"]
        df_util = df_util.select(merge)
        df_util = df_util.join(df_up, on = "bldg_id", how="left").select(columns)
        pass

    return df_util.sort(pl.col("bldg_id").cast(int))    #adjusted with Al to to ensure bldg_id is sorted numerically not lexographically, which was causing some issues with the load profile aggregation weighting

if __name__=="__main__":
    
    state = "NJ"
    utility = "Atlantic City Electric Co"
    gas_utility = ""
    segment = get_segment(state,utility,gas_utility) # + kwargs, u know what they are

    pass