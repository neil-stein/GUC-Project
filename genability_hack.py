import requests
import os
import polars as pl
from datetime import datetime
import glom

def get_elec_tariff(elecTariff, utility, building, territory="", custom_path=""):
    if custom_path:
        return custom_path.rsplit('_',1)[0], pl.read_csv(custom_path)

    territoryId, territoryName = territory if territory else ("","")

    # --- CHANGED: build a safe dir path and ensure it exists
    dir_path = os.path.join("outputs", "electric_tariffs", utility)
    os.makedirs(dir_path, exist_ok=True)
    
    search = f"{territoryName}_{elecTariff}" if territoryName else f"{elecTariff}"
    files = [f for f in os.listdir(dir_path) if f.endswith(search+".csv") and not f.startswith("._")]
    if files:
        file_path = os.path.join(dir_path, files[0])
        df = pl.read_csv(file_path)
        df = df.fill_null("")
        return files[0].rsplit("_",1)[0], df

    app_id = "3df8e135-968d-4399-9879-2a1c6a3de30c"
    app_key = "e51974c7-996b-4698-9628-71950d223364"

    url = "https://api.genability.com/rest/v1/ondemand/calculate"
    params = {
        "masterTariffId": elecTariff,
        "fromDateTime": building["timestamp"].first().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "toDateTime": building["timestamp"].last().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "groupBy": "MONTH",
        "propertyInputs" : [{
            "keyName": "consumption",
            "unit": "kWh",
            "fromDateTime": building["timestamp"].first().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "duration": 900000, # 15 mins
            "dataSeries": building[:,1].to_list()
        }]
    }
    if territoryId:
        params["propertyInputs"].append({"keyName": "territoryId", "dataValue": territoryId})

    response = requests.post(url, auth=(app_id, app_key), json=params)
    data = response.json()["results"][0]
    tariff_name = data["tariffName"] + "-" + territoryName if territoryName else data["tariffName"]
    api_calc = pl.from_dicts(response.json()["results"][0]["items"])
    api_calc = api_calc.sort("rateName")

    # Call Tariff API
    url = "https://api.genability.com/rest/public/tariffs"

    params = {
        "masterTariffId": elecTariff,
        "fromDateTime": building["timestamp"].first().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "toDateTime": building["timestamp"].last().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "populateRates": True
    }

    response = requests.get(url, auth=(app_id, app_key), params=params)
    lseId = response.json()["results"][0]["lseId"]
    data = [d for d in response.json()["results"][0]["rates"] if d.get("territory",{}).get("territoryId",territoryId)==territoryId]
    from_tariff = pl.DataFrame({"rateName":[d["rateName"] for d in data],
                                "chargeClass": [d.get("chargeClass","") for d in data],
                                "blocks":[d["rateBands"] if len(d.get("rateBands"))>1 and any(x.get("hasConsumptionLimit",False) for x in d.get("rateBands",{})) else None for d in data ],
                                "touId": [d.get("timeOfUse",{}).get("touId") for d in data],
                                "seasonId": [d.get("season",{}).get("seasonId") for d in data]
                                })

    # Combine Tariff and Calculate API returns
    cols = [col for col in ["touId","seasonId"] if col in api_calc.columns]
    costs_breakdown = api_calc.join(from_tariff, on=["rateName","chargeClass"], how="left")
    costs_breakdown = costs_breakdown.with_columns([
        pl.coalesce(c,c+"_right").alias(c) for c in cols if c+"_right" in api_calc.columns
    ]).drop(pl.selectors.ends_with("_right"))
    costs_breakdown = (
        costs_breakdown
        .filter(pl.col("fromDateTime").str.contains(str(building["timestamp"].last().year)))
        .with_columns(fromDateTime = pl.col("fromDateTime").str.to_datetime(time_zone="UTC"))
    )
    cols = [c for c in ["rateName","chargeClass","chargeType","quantityKey","rateType","seasonId","touId","blocks"] if c in costs_breakdown.columns]
    costs_breakdown = costs_breakdown.filter(pl.col("fromDateTime").dt.year() == building["timestamp"].last().year).group_by(cols).agg(
        rates = pl.struct([pl.col("rateAmount"),pl.col("fromDateTime").dt.strftime("%m/%d/%Y")])
    ).sort("rateName")

    # Generate tariff
    rows = []
    months = [f"{str(m).zfill(2)}/01/{building['timestamp'].last().year}" for m in range(1,13)]
    for i,rate in enumerate(costs_breakdown.iter_rows(named=True)):
        rate_name = rate["rateName"]
        category = rate.get("chargeClass", "")
        determinant = ""

        # Add missing months with zero rate value (when applicable)
        rates = []
        it = iter(months)
        for month in it:
            r = rate.get("rates")
            if month in [m["fromDateTime"] for m in r]:
                rates.append([m for m in r if m["fromDateTime"] == month][0]["rateAmount"])
            else:
                rates.append(0)

        # Rate Determinant logic
        if rate.get("chargeType") == "FIXED_PRICE":
            rate_determinant = "per month"
            category = "DISTRIBUTION"
        elif rate.get("chargeType") == "QUANTITY" and rate.get("rateType")=="PERCENTAGE":
            rate_determinant = "percent"
            category = "COST"
        elif rate.get("chargeType") == "DEMAND_BASED":
            if "60min" in rate["quantityKey"]:
                rate_determinant = "per 60min kw"
            elif "ratchet" in rate["quantityKey"]:
                rate_determinant = "per kw"
            else:
                rate_determinant = "per 30min kw"
        elif rate.get("chargeType") == "CONSUMPTION_BASED":
            rate_determinant = "per kwh"
        else:
            continue

        # Season Logic
        season = ""
        if rate.get("seasonId"):
            url = f"https://api.genability.com/rest/public/seasons"
            params = {
                "lseId": lseId
            }

            response = requests.get(url, auth=(app_id, app_key), params=params).json()
            spec = ("results", ["seasons"])
            flat = glom.flatten(glom.glom(response,spec))
            for s in flat:
                if s["seasonId"]==rate.get("seasonId"):
                    season = f'{s["seasonFromMonth"]:02d}/{s["seasonFromDay"]:02d}-{s["seasonToMonth"]:02d}/{s["seasonToDay"]:02d}'
                    break
        
        # Time of Use Logic
        tou=[]
        tou_type = ""
        if rate.get("touId"):
            url = f"https://api.genability.com/rest/public/timeofuses/{rate.get('touId')}"

            response = requests.get(url, auth=(app_id, app_key)).json()
            
            for p in response["results"][0]["touPeriods"]:
                p["toHour"] = p["toHour"] if p["toHour"] else 24
                tou.append(([p["fromDayOfWeek"]+1,p["toDayOfWeek"]+1],[p["fromHour"], p["toHour"]]))
            tou_type = response["results"][0].get("touType", "OFF_PEAK")
        tou=str(tou)
        
        # Block Logic
        if rate.get("blocks"):
            bands = rate.get("blocks")
            if any(b.get("consumptionUpperLimit") for b in bands):
                prev_limit = None
                for band in bands:
                    start = prev_limit if prev_limit else ""
                    end = band.get("consumptionUpperLimit","")
                    credit = -1 if band.get("isCredit") else 1
                    rates = [credit*band.get("rateAmount") if r!=0 else 0 for r in rates]
                    determinant = "per month"
                    rows.append([i, tariff_name, rate_name, category, rate_determinant, start, end, determinant, season, tou, tou_type, ""] + rates)
                    prev_limit = end
                continue

        rows.append([i, tariff_name, rate_name, category, rate_determinant, "", "", determinant, season, tou, tou_type, ""] + rates)

    df = pl.DataFrame(rows, schema=[
        "id","tariff","Component Description", "Category", "Rate Determinant",
        "Start", "End", "Determinant", "Season", "tou", "period", "applied_to", *months
    ],orient="row")

    # Sort by charge types in order of demand, kwh, fixed, all else
    df = df.with_columns(
        sort_level = (
            pl.when(pl.col("Rate Determinant").str.contains(r"\bkw\b", literal=False))
            .then(1)
            .when(pl.col("Rate Determinant").str.contains(r"\bkwh\b", literal=False))
            .then(2)
            .when(pl.col("Rate Determinant").str.contains(r"day|month|year|bill", literal=False))
            .then(3)
            .otherwise(4)
        )
    )

    df = df.sort(["sort_level", "Component Description", "Category"])

    df = df.with_row_index("row_idx")

    group_ids = (
        df.group_by(["Component Description", "Category"])
        .agg(pl.col("row_idx").min().alias("id_new"))
        .with_columns((pl.col("id_new") + 1))
    )

    df = (
        df.drop("id")
        .join(group_ids, on=["Component Description", "Category"], how="left")
        .drop(["sort_level", "row_idx"])
        .rename({"id_new": "id"})
        .select(pl.col("id"),pl.all().exclude("id"))
    )


    df.write_csv(f"outputs/electric_tariffs/{utility}/{tariff_name}_{elecTariff}.csv")

    return tariff_name, df

if __name__=="__main__":
    get_elec_tariff("3479678","Commonwealth Edison Company",pl.read_parquet("outputs/load_profiles/NJ/_canonical/0/electricity/total.parquet")[:,:2].with_columns(timestamp = pl.col("timestamp").str.to_datetime(time_zone="UTC")))
    pass