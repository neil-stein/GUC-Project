import time
import os
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.webdriver.edge.service import Service
import polars as pl

# This function uses Selenium to automate web browser steps for downloading and processing the tariff data from RateAcuity.
def get_gas_tariff(state, utility, schedule):
    # make sure schedule name can be saved as filename
    clean = lambda x: (
        x.replace("/", "-")
                .replace("\\", "-")
                .replace(" ", "_")
                .replace(">", "greater_than")
                .replace("<", "less_than")
                .replace(":", "_")
                .replace('"', "_")
                .replace("|", "_")
                .replace("?", "_")
                .replace("*", "_")
                .replace("---", "")
                .strip()
    )
    # only download if not already done
    if os.path.exists(f"outputs/gas_tariffs/{state}-{utility}/{clean(schedule)}.csv"):
        return pl.read_csv(
            f"outputs/gas_tariffs/{state}-{utility}/{clean(schedule)}.csv"
        )

    download_path = os.path.join(os.getcwd(), "outputs", "gas_tariffs")

    # Configure Microsoft Edge WebDriver
    edge_options = Options()
    edge_options.use_chromium = True
    # edge_options.add_argument("--headless")  # Run in headless mode (remove for debugging)
    edge_options.add_argument("--disable-gpu")
    edge_options.add_argument("--no-sandbox")
    edge_options.add_argument("--disable-dev-shm-usage")
    edge_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36")
    edge_options.add_argument("--log-level=3")
    edge_options.add_experimental_option("useAutomationExtension", False)  # Disable automation extension
    edge_options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])  # Remove automation flag
    edge_options.add_experimental_option("prefs", {
    "download.default_directory": download_path,
    "download.prompt_for_download": False,
    "directory_upgrade": True
    })

    service = Service(log_path=os.devnull)

    # Initialize Edge WebDriver
    driver = webdriver.Edge(service=service,options=edge_options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    LINK = "https://secure.rateacuity.com/RateAcuityPortal/Account/Login"
    
    # Navigate to login page
    driver.get(LINK)

    # Check if the 'Log off' element exists and click it if present
    try:
        logoff_element = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, "//a[@href=\"javascript:document.getElementById('logoutForm').submit()\"]"))
        )
        logoff_element.click()
    except:
        pass  # If the element doesn't exist, continue without error
    
    # Login configuration
    EMAIL_ADDRESS = "al.qarooni@rmi.org"
    PASSWORD = "Power200"

    # Login to the page
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'UserName'))).send_keys(EMAIL_ADDRESS)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'Password'))).send_keys(PASSWORD)
    WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//input[@type='submit' and @value='Log in']"))
    ).click()

    # Click on 'Rate Acuity Gas Reports' link
    WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//a[contains(normalize-space(text()), 'Rate Acuity Gas Reports')]"))
    ).click()

    # Select History Option
    WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//input[@id='report' and @value='history']"))
    ).click()

    # Select a state from the dropdown
    state_dropdown = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'StateSelect')))
    select = Select(state_dropdown)
    select.select_by_value(state)

    # Get utilities list
    utility_dropdown = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'UtilitySelect')))
    utility_options = utility_dropdown.find_elements(By.TAG_NAME, 'option')
    option_texts = [option.text.strip() for option in utility_options]
    if utility not in option_texts:
        raise ValueError(f"Gas utility name invalid. Options are: {option_texts}")
    select = Select(utility_dropdown)
    select.select_by_visible_text(utility)

    # Get schedules list
    schedule_dropdown = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'ScheduleSelect')))
    schedule_options = schedule_dropdown.find_elements(By.TAG_NAME, 'option')
    option_texts = [option.text.strip() for option in schedule_options if 'res' in option.text.strip().lower() or 'multi' in option.text.strip().lower()]
    if schedule not in option_texts:
        raise ValueError(f"Choose one of the following schedules: {option_texts}")
    select = Select(schedule_dropdown)
    select.select_by_visible_text(schedule)

    ncomp = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "NComp")))
    ncomp.clear()
    ncomp.send_keys("12")

    nfreq = driver.find_element(By.ID,"NFreq")
    nfreq.clear()
    nfreq.send_keys("1")

    driver.find_element(By.XPATH, "//input[@type='submit' and @value='Search']").click()
    time.sleep(10)

    rate_file = [f for f in os.listdir(download_path) if f.endswith('.xlsx') and not f.startswith("~$")]

    if not rate_file:
        driver.quit()
        raise Exception("RateAcuity download failed, no file found in download directory.")

    rate_file_path = os.path.join(download_path, rate_file[0])

    directory_name = f"{state}-{utility}"
    target_dir = os.path.join(download_path, directory_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    # --- Load Excel file with Polars ---
    # --- Detect "Component Description" in the first column and set as header row ---
    raw_data = pl.read_excel(rate_file_path, engine="calamine", has_header=False)
    header_row_index = None

    for i, row in enumerate(raw_data.iter_rows()):
        if "Component Description" in row[0] or [utility in r for r in row[0]]:
            header_row_index = i
            break

    if header_row_index is None:
        raise Exception(f"Could not parse gas tariff {schedule}, downloaded excel file is in an unknown format.")
    
    # Organize Tariffs
    df = pl.read_excel(
        rate_file_path,
        engine="calamine",
        read_options={"header_row": header_row_index}
    )
    df = df.with_row_index("row_idx").rename({df.columns[0]: "Component Description"})

    # Add ids   
    group_ids = (
        df.group_by("Component Description")
        .agg(pl.col("row_idx").min().alias("id_new"))
        .with_columns((pl.col("id_new") + 1))
    )

    df = (
        df
        .join(group_ids, on="Component Description", how="left")
        .drop("row_idx")
        .rename({"id_new": "id"})
        .select(pl.col("id"),pl.all().exclude("id"))
    )

    df = df.select(
        ~pl.selectors.matches(r"/"),
        pl.lit("").alias("applied_to"),
        pl.selectors.matches(r"/")
    )
    if "Location" in df.columns:
        df = df.filter(pl.col("Location").is_null() | (pl.col("Location") == ""))

    df.write_csv(os.path.join(target_dir,clean(schedule) + ".csv"))
    os.remove(rate_file_path) # Remove the original file after processing

    return df

if __name__=="__main__":
    import sys, os
    sys.path.append(os.path.abspath("."))
    import polars as pl
    from utils.tariff_object import Tariff
    # building = pl.read_parquet("outputs/load_profiles/CA/128427-0.parquet")
    df = get_gas_tariff("IL", "Nicor Gas", "1-RESIDENTIAL SERVICE---")
    
    # tariff = Tariff(df)
    # annual_bill, costs_by_rate, monthly_totals, detailed_monthly = tariff.annual_bill(building.select(["timestamp","natural_gas.total"]))

    pass