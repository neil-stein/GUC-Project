import polars as pl
from tqdm.auto import tqdm
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from utils.bill_object import Bill
from utils.load_profiles import adjust_load_profile

select_by_id = pl.selectors.matches(r"^\-?[0-9]+$|^.*_whole$")
select_quants = pl.selectors.matches(r"total_cost")
select_costs = pl.selectors.ends_with("_cost").exclude("total_cost")

def batch_agg_by_period(df, every, period=None, batch_size=500, op ="sum"):
    """Fast aggregation for wide DataFrames: groups by time and sums numeric columns in batches.
    
    Processes numeric columns in batches to avoid Polars creating thousands of separate sum operations.
    Useful for DataFrames with many building ID columns (15k+).
    
    Args:
        df: DataFrame with timestamp + numeric (building ID) columns
        every: Period string (e.g., "1mo", "1d", "1h")
        period: Optional period parameter for group_by_dynamic
        batch_size: Number of numeric columns per batch (default 500)
        op: Polars aggregation function as string (default "sum", can be "max", "mean", etc.)
    
    Returns:
        Aggregated DataFrame with same structure (timestamp + summed numeric columns)
    """
    cols = df.select(select_by_id|select_quants|select_costs).columns
    
    if not cols:
        return df
    
    results = []
    for i in range(0, len(cols), batch_size):
        batch_cols = cols[i:i+batch_size]
        batch_df = df.select(["timestamp"] + batch_cols)
        func = {
            "sum": pl.all().exclude("timestamp").sum(),
            "max": pl.all().exclude("timestamp").max(),
            "mean": pl.all().exclude("timestamp").mean(),
            "min": pl.all().exclude("timestamp").min()
        }
        if period:
            agg = batch_df.group_by_dynamic("timestamp", every=every, period=period).agg(
                func[op]
            )
        else:
            agg = batch_df.group_by_dynamic("timestamp", every=every).agg(
                func[op]
            )
        
        results.append(agg)
    
    # Join all batches back together on timestamp
    final = results[0]
    for r in results[1:]:
        final = final.join(r, on="timestamp", how="inner")
    
    return final.select(["timestamp",*cols]).sort("timestamp")

@dataclass
class charge_type():
    unit: str = ""

    def __call__(self, load):
        id_cols = load.select(select_by_id).columns

        if self.unit in ["month", "bill"]:
            load = batch_agg_by_period(load, "1mo")
            load = load.with_columns([pl.lit(1.0, dtype=pl.Float32).alias(c) for c in id_cols])
            
        elif self.unit in ["day"]:
            load = batch_agg_by_period(load, "1d")
            load = load.with_columns([pl.lit(1.0, dtype=pl.Float32).alias(c) for c in id_cols])
            
        elif self.unit in ["year"]:
            load = batch_agg_by_period(load, "1y")
            load = load.with_columns([pl.lit(1.0, dtype=pl.Float32).alias(c) for c in id_cols])

        elif self.unit in ["therm", "ccf"]:
            load = load.with_columns(
                (select_by_id*0.03412).round(9) if self.unit=="therm" else (select_by_id/30.2).round(9) if self.unit=="ccf" else select_by_id.round(9)
            )
        else:
            pass
        return load

@dataclass
class seasonal():
    season: str = ""

    def __call__(self, load):
        mo1, day1 = map(int, self.season.split("-")[0].split("/"))
        mo2, day2 = map(int, self.season.split("-")[1].split("/"))
        t1, t2 = (datetime(2025,mo1,day1,tzinfo=timezone.utc),datetime(2025,mo2,day2,tzinfo=timezone.utc))
        if mo1 <= mo2:
            t2 += timedelta(hours=23,minutes=59)
            filter = pl.col("timestamp").is_between(t1,t2)
        else:
            t1 -= timedelta(minutes=1)
            t2 += timedelta(days=1)
            filter = ~pl.col("timestamp").is_between(t2,t1)
        
        load = load.with_columns(
            pl.when(filter).then(select_by_id).otherwise(pl.lit(0.0))
        )

        return load

@dataclass
class tou():
    tou: str = ""

    def __call__(self, load):
        if eval(self.tou):
            filter = False
            tou = eval(self.tou)
            for t_d, t_h in tou:
                start_day, end_day = t_d
                start_time, end_time = t_h
                if start_time < end_time:
                    filter = filter | ((pl.col("timestamp").dt.hour() >= start_time) & (pl.col("timestamp").dt.hour() < end_time) &
                                    (pl.col("timestamp").dt.weekday()>= start_day) & (pl.col("timestamp").dt.weekday() <= end_day))
                else:
                    filter = filter | (((pl.col("timestamp").dt.hour() >= start_time) | (pl.col("timestamp").dt.hour() < end_time)) &
                                    (pl.col("timestamp").dt.weekday()>= start_day) & (pl.col("timestamp").dt.weekday() <= end_day))
            load = load.with_columns(
                pl.when(filter).then(select_by_id).otherwise(pl.lit(0.0))
            )
        return load

@dataclass
class tiered():
    start: float = 0.0
    end: float = float("inf")
    freq: str = "1mo"
    
    def __call__(self,load):
        self.start = float(self.start)
        self.end = float(self.end)
        load = batch_agg_by_period(load, self.freq)
        load = load.with_columns((select_by_id - self.start).clip(0,self.end-self.start))
        return load

@dataclass
class demand():
    unit: str = "30min"

    def __call__(self, load):
        if "60min" in self.unit:
            load = batch_agg_by_period(load, "15m", period="1h")
            load = batch_agg_by_period(load, "1mo", func="max")
        elif "30min" in self.unit:
            load = batch_agg_by_period(load, "15m", period="30m")
            load = batch_agg_by_period(load, "1mo", func="max")
            load = load.with_columns(select_by_id*2)
        else:
            load = batch_agg_by_period(load, "1mo", func="max")
            load = load.with_columns(select_by_id*4)
        return load

@dataclass
class Rate:
    id: int = 0
    name: str = ""
    rate: dict = field(default_factory=dict)
    category: str = ""
    unit: str = ""
    season: str = "01/01-12/31"
    start: float = 0.0
    end: float = float("inf")
    tou: str = "[]"
    tier_freq: str = "1mo"
    applied_to: str = ""
    metadata: pl.DataFrame = field(default_factory=pl.DataFrame)
    custom: callable = None

    load_components: tuple[callable, ...] = field(default_factory=tuple)
    cost_components: tuple[callable, ...] = field(default_factory=tuple)

    def __post_init__(self):
        self.unit = self.unit.split("per ")[-1].lower()
        self.tier_freq = "1d" if "day" in self.tier_freq else "1mo"
        self.category = "FIXED_M" if self.unit in ["bill", "month"] else "FIXED_D" if self.unit=="day" else "FIXED_Y" if self.unit=="year" else self.category

        if not self.custom:
            self.load_components = [charge_type(self.unit)]

            self.load_components += [
                seasonal(self.season),
                tou(self.tou)
            ]
            
            if self.unit in ["60min kw", "30min kw", "kw"]:
                self.load_components += [demand(self.unit)]
            
            self.load_components += [tiered(self.start,self.end,self.tier_freq)] if self.unit not in ["bill", "month", "day", "year", "percent"] else []

        else:
            self.load_components = [self.custom]

    def process_load(self, profile):

        for component in self.load_components:
            profile = component(profile)
        
        return profile
    
    def monthly_cost(self, orig_load):
        """
        Pass timestamp and load as a 2-column df
        """
        load = self.process_load(orig_load)
        load = batch_agg_by_period(load, "1mo")

        cost_profile = load.with_columns([
            (pl.col(col) * self.rate["rate"]).round(9).alias(f"{col}_cost") 
            for col in load.select(select_by_id).columns
        ])
        cost_profile = cost_profile.select([
            pl.col("timestamp").alias("timestamp"),
            pl.lit(self.id).alias("id"),
            pl.lit(self.name).alias("rateName"),
            pl.lit(self.category).alias("category"),
            pl.sum_horizontal(select_costs).round(9).alias("total_cost"),
            select_by_id.round(9),
            select_costs.round(9)
        ])

        return cost_profile

    def process_cost_riders(self, profile):

        self.category = "COST"
        profile = self.process_load(profile)        
        result = profile.with_columns(select_costs*(self.rate["rate"]/100).round(9))

        return result.select([
            pl.col("timestamp").alias("timestamp"),
            pl.lit(self.id).alias("id"),
            pl.lit(self.name).alias("rateName"),
            pl.lit(self.category).alias("category"),
            pl.sum_horizontal(select_costs).round(9).alias("total_cost"),
            select_by_id.round(9),
            select_costs.round(9)
        ])

class Tariff:
    def __init__(self, tariff, metadata = pl.DataFrame(), use_weights=False, custom_rates = {}):
        self.use_weights = use_weights          # customer counts flag
        self.metadata = metadata                # all info about buildings being processed
        self.tariff = tariff
        if "id" not in tariff.columns:
            tariff = tariff.select([pl.int_range(0,pl.len()).alias("id"),pl.all()])
        else:
            tariff = tariff.with_columns(pl.col("id"))
        rates = self.parse_tariff(tariff, custom_rates)       # parses all the rates from input tariff

        # separates load based rates from cost based rates 
        self.consumption_rates = [r for r in rates if "percent" not in r.unit]
        self.cost_rates = [r for r in rates if "percent" in r.unit]
        
        if any("percent" in r.unit for r in rates) and all(r.applied_to=="" for r in self.cost_rates):
            print(f"WARNING! Detected cost-based riders but applied_to column is empty, excluding... Populate applied_to column to include cost-based riders.")
        
        # Determine which weight column to use based on consumption rate units
        # If kWh or kW exists: use elec_weight; if therm or ccf exists: use gas_weight
        self.weight_col = None
        if not metadata.is_empty():
            units = [r.unit for r in self.consumption_rates]
            if any(u in ["kwh", "kw", "60min kw", "30min kw"] for u in units):
                self.weight_col = "elec_weight"
            elif any(u in ["therm", "ccf"] for u in units):
                self.weight_col = "gas_weight"
            
    def parse_tariff(self, tariff, custom_rates):
        if len([k for k in tariff.columns if "/" in k])==12:
            pass
        else:
            raise ValueError("Invalid tariff structure")
        rates=[]
        for rate in tariff.iter_rows(named=True):
            # Skip location based rates
            if rate.get("Location") or not rate.get("Rate Determinant"):
                continue

            # Collect rate constraints
            mapping = {
                "id": "id",
                "name": "Component Description",
                "rate": "Rate",
                "category": "Category",
                "unit": "Rate Determinant",
                "season": "Season",
                "start": "Start",
                "end": "End",
                "tou": "tou",
                "tier_freq": "Determinant",
                "applied_to": "applied_to"
            }
            kwags = {k: rate.get(v, "") for k,v in mapping.items() if rate.get(v) not in [None,""]}

            # Collect rates
            kwags["rate"] = pl.DataFrame(
                {
                    "month": [int(k.split("/")[0]) for k,_ in rate.items() if "/" in k],
                    "rate": [v for k,v in rate.items() if "/" in k]
                }
            ).sort("month")

            # Propogate metadata for customer counts
            if self.use_weights:
                kwags["metadata"] = self.metadata

            # Check for custom rate handling
            for ids, func in custom_rates.items():
                if kwags["id"] in ids:
                    kwags["custom"] = func
                    break

            rates.append(Rate(**kwags))
        
        return rates

    def annual_bill(self, load_orig):
        # Format load and optimize for non-demand and non-tou rates
        load = pl.DataFrame(load_orig)

        # Collect monthly totals
        monthly = []
        print("Processing consumption rates...")
        for r in tqdm(self.consumption_rates):
            if r.unit in ["60min kw", "30min kw", "kw"]:
                processed_bills = r.monthly_cost(load)
                monthly.append(processed_bills)
            elif eval(r.tou):
                temp = batch_agg_by_period(load, "1h")
                processed_bills = r.monthly_cost(temp)
                monthly.append(processed_bills)
            elif r.tier_freq=="day":
                temp = batch_agg_by_period(load, "1d")
                processed_bills = r.monthly_cost(temp)
                monthly.append(processed_bills)
            else:
                temp = batch_agg_by_period(load, "1mo")
                processed_bills = r.monthly_cost(temp)
                monthly.append(processed_bills)

        # Apply cost rates
        agg_costs = pl.concat(monthly)
        print("Processing cost-based riders...")
        percent_costs = [
            r.process_cost_riders(
                agg_costs
                .filter(pl.col("id").cast(str).is_in(r.applied_to.split(",")))
                .group_by("timestamp")
                .agg(select_quants.sum(),select_by_id.sum(),select_costs.sum())
            )
            for r in tqdm(self.cost_rates) if r.applied_to
        ]
        monthly += percent_costs

        # Apply weights if you want - changes total costs based on customer counts
        monthly = pl.concat(monthly)
        if self.use_weights and self.weight_col is not None:
            # Use adjust_load_profile to apply weights to calculate aggregations
            monthly = adjust_load_profile(monthly, self.metadata)['adjusted_load']
            # Adjust totals
            monthly = monthly.with_columns(
                pl.sum_horizontal([pl.col(c) for c in monthly.select(select_costs).columns]).round(9).alias("total_cost")
            )

        # Collect results
        results = {}

        results['load'] = monthly.group_by("timestamp").agg(select_by_id.first()).sort("timestamp")
        
        monthly = monthly.select(
            pl.col("id","rateName","category","timestamp"), select_quants,
            select_costs.name.map(lambda c: c.replace("_cost", ""))
        )

        # Annual Bill
        results["annual"] = (
            monthly.select(
                select_quants.sum().round(9),
                select_by_id.sum().round(9)).sum()
        )

        # Breakdown by Rate
        results["by_rate"] = (
            monthly
                .group_by(["id","rateName","category"],maintain_order=True)
                .agg(
                    select_quants.sum().round(9),
                    select_by_id.sum().round(9)
                )
                .sort([pl.col("id").cast(int),"rateName","category"])
        )
        
        # Final Monthly
        results["monthly"] = (
            monthly
                .group_by("timestamp",maintain_order=True)
                .agg(
                    select_quants.sum().round(9),
                    select_by_id.sum().round(9)
                )
                .sort(["timestamp"])
        )

        # Detailed monthly
        results["detailed"] = (
            monthly
                .group_by(["id","rateName","category","timestamp"],maintain_order=True)
                .agg(
                    select_quants.sum().round(9),
                    select_by_id.sum().round(9)
                )
                .sort([pl.col("id").cast(int),"rateName","category","timestamp"])
        )

        return Bill("electric" if "elec" in self.weight_col else "gas", "", results, self.metadata)
    

if __name__=="__main__":
    load = pl.read_parquet("/Volumes/CrucialX6/Coding/rates_analytics/outputs/load_profiles/NJ/Atlantic City Electric Co/elec_tot_0.parquet")
    test = pl.read_csv("/Volumes/CrucialX6/Coding/rates_analytics/outputs/electric_tariffs/Atlantic City Electric Co/Residential_A3.csv")
    import segments
    tariff = Tariff(test, segments.get_segment("NJ","Atlantic City Electric Co"),use_weights=True) # + kwargs, u know what they are
    results = tariff.annual_bill(load)
    results.by_rate.write_csv("dev/test_cost.csv")
    pass
