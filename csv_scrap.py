from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time
import csv

# List of product page URLs
product_urls = [
    "https://www.hbkworld.com/en/products/instruments/handheld/sound-level-meters/type-2255/2255-noise-partner",
    "https://www.hbkworld.com/en/products/instruments/handheld/sound-level-meters/type-2255/2255-building-acoustics",
    "https://www.hbkworld.com/en/products/instruments/handheld/sound-level-meters/type-2255/2255-enviro-noise",
    "https://www.hbkworld.com/en/products/instruments/handheld/sound-level-meters/type-2255/2255-work-noise",
    "https://www.hbkworld.com/en/products/instruments/handheld/sound-level-meters/type-2255/product-noise"
]

def setup_driver():
    """Setup Chrome driver with optimal configuration"""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    try:
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(30)
        return driver
    except Exception as e:
        print(f"Error setting up driver: {e}")
        return None

def extract_product_info(driver, wait):
    """Extract basic product information"""
    product_name = "N/A"
    small_desc = "N/A"
    
    # Try multiple selectors for product name
    name_selectors = [
        "h1.cmp-title__text",
        "h1",
        ".product-title",
        ".page-title"
    ]
    
    for selector in name_selectors:
        try:
            element = driver.find_element(By.CSS_SELECTOR, selector)
            product_name = element.text.strip()
            if product_name:
                break
        except NoSuchElementException:
            continue
    
    # Extract description from div.cmp-text p - try multiple approaches
    desc_selectors = [
        "div.cmp-text p",  # Primary selector
        "main div.cmp-text p",  # More specific - in main content
        ".content div.cmp-text p",  # In content area
        "article div.cmp-text p"  # In article area
    ]
    
    for selector in desc_selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for element in elements:
                text = element.text.strip()
                # Skip if it's navigation text or too short
                if (text and 
                    len(text) > 20 and  # Must be substantial text
                    "HBKShop" not in text and 
                    "Training" not in text and 
                    "Resources" not in text and
                    "navigation" not in text.lower() and
                    "menu" not in text.lower()):
                    small_desc = text
                    break
            if small_desc != "N/A":
                break
        except NoSuchElementException:
            continue
    
    return product_name, small_desc

def click_tab(driver, wait, tab_text):
    """Click on a specific tab and wait for content to load"""
    tab_selectors = [
        f"//button[contains(text(), '{tab_text}')]",
        f"//a[contains(text(), '{tab_text}')]",
        f"//*[contains(@class, 'tab')][contains(text(), '{tab_text}')]",
        f"//*[contains(@data-tab, '{tab_text.lower()}')]"
    ]
    
    for selector in tab_selectors:
        try:
            tab_element = wait.until(EC.element_to_be_clickable((By.XPATH, selector)))
            driver.execute_script("arguments[0].click();", tab_element)
            time.sleep(2)  # Wait for content to load
            return True
        except TimeoutException:
            continue
    
    print(f"Could not find or click '{tab_text}' tab")
    return False

def extract_models(soup):
    """Extract model information from the models tab"""
    models = []
    
    # Look for the models table (usually has Code | Name structure)
    tables = soup.find_all("table")
    
    for table in tables:
        # Check if this looks like a models table
        headers = [th.text.strip() for th in table.find_all("th")]
        if not headers:
            # Try first row as headers
            first_row = table.find("tr")
            if first_row:
                headers = [td.text.strip() for td in first_row.find_all("td")]
        
        # Look for Code/Name pattern or product code pattern
        if any(header.lower() in ['code', 'model', 'name'] for header in headers):
            rows = table.find_all("tr")
            start_index = 1 if table.find_all("th") else 1  # Skip header row
            
            for row in rows[start_index:]:
                cells = row.find_all("td")
                if len(cells) >= 2:
                    code = cells[0].text.strip()
                    name = cells[1].text.strip()
                    if code and name and code != "Code" and name != "Name":
                        models.append({
                            "Model": code,
                            "Product Name": name
                        })
    
    # If no models found in typical structure, look for product code attributes
    if not models:
        model_rows = soup.find_all("tr", attrs={"data-product-code": True})
        for row in model_rows:
            try:
                code_element = row.find("a", class_="download_link")
                if code_element:
                    code = code_element.text.strip()
                    # Get last cell for name
                    cells = row.find_all("td")
                    name = cells[-1].text.strip() if cells else ""
                    if code:
                        models.append({
                            "Model": code,
                            "Product Name": name
                        })
            except:
                continue
    
    return models

def extract_specifications(soup):
    """Extract specifications from the specifications table"""
    spec_data = {}
    
    # Find all tables and look for the specifications table
    tables = soup.find_all("table")
    
    for table in tables:
        # Check if this looks like a specifications table (2 columns with parameter-value pairs)
        rows = table.find_all("tr")
        if len(rows) > 5:  # Specifications table should have multiple rows
            # Check if it's a 2-column table with specifications
            first_row = rows[0]
            cells = first_row.find_all(["td", "th"])
            
            if len(cells) == 2:  # 2-column table, likely specifications
                for row in rows:
                    cells = row.find_all("td")
                    if len(cells) == 2:
                        param = cells[0].text.strip()
                        value = cells[1].text.strip()
                        if param and value:
                            # Skip hidden rows (imperial units)
                            if "hidden" not in row.get("class", []):
                                spec_data[param] = value
                break
    
    return spec_data

def scrape_product_page(driver, wait, url):
    """Scrape a single product page"""
    print(f"Processing {url}...")
    
    try:
        driver.get(url)
        time.sleep(3)  # Initial page load wait
        
        # Extract basic product info
        product_name, small_desc = extract_product_info(driver, wait)
        print(f"  Product: {product_name}")
        print(f"  Description: {small_desc[:100]}...")
        
        # Initialize models list
        models = []
        
        # Try to click "All models" tab to get model information
        if click_tab(driver, wait, "All models"):
            soup = BeautifulSoup(driver.page_source, "html.parser")
            models = extract_models(soup)
            if models:
                print(f"  Found {len(models)} models")
            else:
                print("  No models found in All models tab")
        
        # If no models found, create a default entry based on the product
        if not models:
            models = [{"Model": "Standard", "Product Name": product_name}]
            print("  Using default model entry")
        
        # Try to click "Specifications" tab to get specifications
        specifications = {}
        if click_tab(driver, wait, "Specifications"):
            soup = BeautifulSoup(driver.page_source, "html.parser")
            specifications = extract_specifications(soup)
            if specifications:
                print(f"  Found {len(specifications)} specification parameters")
            else:
                print("  No specifications found in Specifications tab")
        
        # If no specs found, try extracting from current page
        if not specifications:
            soup = BeautifulSoup(driver.page_source, "html.parser")
            specifications = extract_specifications(soup)
            if specifications:
                print(f"  Found {len(specifications)} specifications on main page")
        
        # Create output data - one row per model with all specifications
        page_data = []
        for model in models:
            row_data = {
                "Product": product_name,
                "Small Description": small_desc,
                "Model Code": model["Model"],
                "Model Name": model["Product Name"]
            }
            
            # Add all specification parameters as separate columns
            for spec_name, spec_value in specifications.items():
                row_data[spec_name] = spec_value
            
            page_data.append(row_data)
        
        return page_data
        
    except Exception as e:
        print(f"  Error processing {url}: {e}")
        return []

def main():
    """Main scraping function"""
    driver = setup_driver()
    if not driver:
        print("Failed to setup driver")
        return
    
    wait = WebDriverWait(driver, 15)
    all_rows = []
    
    try:
        for url in product_urls:
            page_data = scrape_product_page(driver, wait, url)
            all_rows.extend(page_data)
            time.sleep(2)  # Be respectful to the server
        
        if all_rows:
            # Create DataFrame and save to CSV
            df = pd.DataFrame(all_rows)
            
            # Clean up column names
            df.columns = [col.strip() for col in df.columns]
            
            # Save to CSV
            filename = "hbk_2245_products.csv"
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"\nSaved {len(all_rows)} records to {filename}")
            
            # Also save as backup with more detailed CSV writer
            backup_filename = "hbk_2245_products_backup.csv"
            with open(backup_filename, 'w', newline='', encoding='utf-8') as f:
                if all_rows:
                    writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
                    writer.writeheader()
                    writer.writerows(all_rows)
            print(f"Backup saved to {backup_filename}")
            
            # Print summary
            print(f"\nScraping Summary:")
            print(f"Total products processed: {len(product_urls)}")
            print(f"Total records extracted: {len(all_rows)}")
            print(f"Columns: {list(df.columns)}")
            
        else:
            print("No data was extracted from any of the pages")
            
    except Exception as e:
        print(f"Error during scraping: {e}")
    
    finally:
        driver.quit()
        print("Driver closed")

if __name__ == "__main__":
    main()