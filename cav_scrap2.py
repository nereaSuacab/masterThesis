from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time
import csv
import re

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
    # Commented out headless mode for debugging - uncomment for production
    # options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    try:
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(30)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        return driver
    except Exception as e:
        print(f"Error setting up driver: {e}")
        return None

def extract_product_info(driver, wait):
    """Extract basic product information"""
    product_name = "N/A"
    small_desc = "N/A"
    
    # Try multiple selectors for product name with more specific targeting
    name_selectors = [
        "h1.cmp-title__text",
        "h1[data-cmp-hook-title='title']",
        "h1.product-title",
        "h1.page-title",
        "h1",
        ".hero-title h1",
        ".product-hero h1"
    ]
    
    for selector in name_selectors:
        try:
            element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
            product_name = element.text.strip()
            if product_name and len(product_name) > 3:
                break
        except (NoSuchElementException, TimeoutException):
            continue
    
    # Extract description with better targeting
    desc_selectors = [
        "div.cmp-text p",
        ".product-intro p",
        ".hero-content p",
        "main .cmp-text p",
        ".product-description p",
        ".intro-text p"
    ]
    
    for selector in desc_selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for element in elements:
                text = element.text.strip()
                # More refined filtering for description text
                if (text and 
                    len(text) > 30 and 
                    len(text) < 500 and
                    "HBKShop" not in text and
                    "Training" not in text and
                    "Resources" not in text and
                    "Cookie" not in text and
                    "navigation" not in text.lower() and
                    "menu" not in text.lower() and
                    not text.startswith("Contact") and
                    not re.match(r'^[\d\s\-\+\(\)]+$', text)):  # Not just numbers/phone
                    small_desc = text
                    break
            if small_desc != "N/A":
                break
        except NoSuchElementException:
            continue
    
    return product_name, small_desc

def wait_for_content_load(driver, timeout=10):
    """Wait for dynamic content to load"""
    try:
        # Wait for any loading indicators to disappear
        wait = WebDriverWait(driver, timeout)
        # Check for common loading indicators
        loading_selectors = [
            ".loading",
            ".spinner",
            "[data-loading='true']",
            ".cmp-loading"
        ]
        
        for selector in loading_selectors:
            try:
                wait.until(EC.invisibility_of_element_located((By.CSS_SELECTOR, selector)))
            except TimeoutException:
                pass
        
        time.sleep(2)  # Additional wait for content to stabilize
        return True
    except Exception:
        return False

def click_tab_advanced(driver, wait, tab_text):
    """Advanced tab clicking with multiple strategies"""
    print(f"  Attempting to click '{tab_text}' tab...")
    
    # Multiple strategies for finding and clicking tabs
    strategies = [
        # Strategy 1: Direct button/link with text
        (By.XPATH, f"//button[contains(normalize-space(text()), '{tab_text}')]"),
        (By.XPATH, f"//a[contains(normalize-space(text()), '{tab_text}')]"),
        
        # Strategy 2: Tab with class containing text
        (By.XPATH, f"//*[contains(@class, 'tab')][contains(normalize-space(text()), '{tab_text}')]"),
        
        # Strategy 3: Data attributes
        (By.XPATH, f"//*[contains(@data-tab, '{tab_text.lower()}')]"),
        (By.XPATH, f"//*[contains(@data-toggle, 'tab')][contains(normalize-space(text()), '{tab_text}')]"),
        
        # Strategy 4: Generic clickable elements with tab text
        (By.XPATH, f"//li[contains(normalize-space(text()), '{tab_text}')]//a"),
        (By.XPATH, f"//div[contains(@class, 'tab')][contains(normalize-space(text()), '{tab_text}')]"),
        
        # Strategy 5: Case insensitive and partial matches
        (By.XPATH, f"//*[contains(translate(normalize-space(text()), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{tab_text.lower()}')]"),
    ]
    
    for by_method, selector in strategies:
        try:
            elements = driver.find_elements(by_method, selector)
            for element in elements:
                try:
                    # Check if element is visible and clickable
                    if element.is_displayed() and element.is_enabled():
                        # Scroll element into view
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
                        time.sleep(1)
                        
                        # Try multiple click methods
                        click_methods = [
                            lambda e: e.click(),
                            lambda e: driver.execute_script("arguments[0].click();", e),
                            lambda e: driver.execute_script("arguments[0].dispatchEvent(new MouseEvent('click', {bubbles: true}));", e)
                        ]
                        
                        for click_method in click_methods:
                            try:
                                click_method(element)
                                wait_for_content_load(driver)
                                print(f"  Successfully clicked '{tab_text}' tab")
                                return True
                            except Exception as click_error:
                                continue
                                
                except (StaleElementReferenceException, Exception):
                    continue
                    
        except Exception:
            continue
    
    print(f"  Could not find or click '{tab_text}' tab")
    return False

def extract_models_advanced(soup, driver):
    """Advanced model extraction with multiple parsing strategies"""
    models = []
    
    # Strategy 1: Look for standard tables with model information
    tables = soup.find_all("table")
    for table in tables:
        # Check table headers
        headers = []
        header_elements = table.find_all("th")
        if header_elements:
            headers = [th.get_text(strip=True).lower() for th in header_elements]
        else:
            # Try first row as headers
            first_row = table.find("tr")
            if first_row:
                first_cells = first_row.find_all(["td", "th"])
                headers = [cell.get_text(strip=True).lower() for cell in first_cells]
        
        # Look for model/code patterns in headers
        model_indicators = ['code', 'model', 'part', 'product', 'type', 'name']
        if any(indicator in ' '.join(headers) for indicator in model_indicators):
            rows = table.find_all("tr")
            data_rows = rows[1:] if header_elements else rows[1:]  # Skip header
            
            for row in data_rows:
                cells = row.find_all("td")
                if len(cells) >= 2:
                    code_text = cells[0].get_text(strip=True)
                    name_text = cells[1].get_text(strip=True)
                    
                    # Filter out header repetitions and empty cells
                    if (code_text and name_text and 
                        code_text.lower() not in ['code', 'model', 'part'] and
                        name_text.lower() not in ['name', 'description', 'product'] and
                        len(code_text) > 2):
                        models.append({
                            "Model": code_text,
                            "Product Name": name_text
                        })
    
    # Strategy 2: Look for product variant lists or download links with model codes
    if not models:
        # Look for download links which often contain model codes
        download_links = soup.find_all("a", class_=re.compile(r"download"))
        for link in download_links:
            link_text = link.get_text(strip=True)
            if re.search(r'-\d{4}-[A-Z]-', link_text):  # Pattern like -2255-N-
                models.append({
                    "Model": link_text,
                    "Product Name": link_text
                })
    
    # Strategy 3: Look for specific HBK product code patterns in any text
    if not models:
        all_text = soup.get_text()
        # Look for HBK product codes (pattern: -NNNN-L-LL-)
        product_codes = re.findall(r'-\d{4}-[A-Z]-[A-Z0-9]+-?', all_text)
        for code in set(product_codes):  # Remove duplicates
            models.append({
                "Model": code,
                "Product Name": f"Model {code}"
            })
    
    # Strategy 4: Interactive elements - try to find and extract from dropdowns or selectors
    if not models:
        try:
            # Look for product selectors or dropdowns
            selectors = soup.find_all(["select", "datalist"])
            for selector in selectors:
                options = selector.find_all("option")
                for option in options:
                    option_text = option.get_text(strip=True)
                    option_value = option.get("value", "")
                    if option_text and option_text not in ["Select", "Choose", ""]:
                        models.append({
                            "Model": option_value if option_value else option_text,
                            "Product Name": option_text
                        })
        except Exception:
            pass
    
    return models

def extract_specifications_advanced(soup):
    """Advanced specifications extraction with multiple strategies and better filtering"""
    spec_data = {}
    
    # Remove cookie banners, navigation, and other unwanted elements first
    unwanted_selectors = [
        '[id*="cookie"]', '[class*="cookie"]',
        '[id*="privacy"]', '[class*="privacy"]',
        '[class*="navigation"]', '[class*="nav"]',
        '[class*="footer"]', '[class*="header"]',
        '[class*="banner"]', '[class*="consent"]'
    ]
    
    for selector in unwanted_selectors:
        for element in soup.select(selector):
            element.decompose()
    
    # Strategy 1: Standard specification tables (2-column parameter-value)
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        if len(rows) < 3:  # Skip small tables
            continue
            
        # Check if it's a 2-column specification table
        sample_row = None
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) == 2:
                sample_row = row
                break
        
        if sample_row:
            for row in rows:
                cells = row.find_all("td")
                if len(cells) == 2:
                    param = cells[0].get_text(strip=True)
                    value = cells[1].get_text(strip=True)
                    
                    # Enhanced filtering for unwanted content
                    unwanted_keywords = [
                        'privacy', 'cookie', 'consent', 'policy', 'google', 'analytics',
                        'ads', 'facebook', 'linkedin', 'microsoft', 'adobe', 'hubspot',
                        'cloudflare', 'magento', 'sitecore', 'zoominfo', 'months', 'years',
                        'browser', 'delete', 'functional', 'vendor', 'expiry', 'rollout'
                    ]
                    
                    # Clean up parameter names and values
                    if (param and value and len(param) < 100 and
                        not any(keyword in param.lower() for keyword in unwanted_keywords) and
                        not any(keyword in value.lower() for keyword in unwanted_keywords) and
                        'privacy policy' not in value.lower() and
                        not param.lower().startswith(('click', 'read', 'view', 'download')) and
                        not re.match(r'^-\d{4}-[A-Z]-', param)):  # Skip model codes as specs
                        
                        # Skip hidden rows (often imperial units)
                        row_classes = row.get("class", [])
                        if "hidden" not in row_classes:
                            # Clean parameter name
                            param_clean = re.sub(r'\s+', ' ', param)
                            value_clean = re.sub(r'\s+', ' ', value)
                            spec_data[param_clean] = value_clean
    
    # Strategy 2: Definition lists (dl, dt, dd)
    dl_elements = soup.find_all("dl")
    for dl in dl_elements:
        terms = dl.find_all("dt")
        descriptions = dl.find_all("dd")
        
        for term, desc in zip(terms, descriptions):
            param = term.get_text(strip=True)
            value = desc.get_text(strip=True)
            if param and value:
                spec_data[param] = value
    
    # Strategy 3: Divs with specification patterns
    spec_containers = soup.find_all("div", class_=re.compile(r"spec|specification|parameter"))
    for container in spec_containers:
        # Look for label-value pairs within the container
        labels = container.find_all(["label", "span", "div"], class_=re.compile(r"label|param|spec"))
        values = container.find_all(["span", "div"], class_=re.compile(r"value|data"))
        
        for label, value in zip(labels, values):
            param = label.get_text(strip=True)
            val = value.get_text(strip=True)
            if param and val:
                spec_data[param] = val
    
    # Strategy 4: Key-value pairs in structured content
    key_value_patterns = [
        (r"(\w+(?:\s+\w+)*)\s*[:]\s*([^\n\r]+)", soup.get_text()),  # "Parameter: Value"
        (r"(\w+(?:\s+\w+)*)\s*[-]\s*([^\n\r]+)", soup.get_text()),  # "Parameter - Value"
    ]
    
    for pattern, text in key_value_patterns:
        matches = re.findall(pattern, text)
        for param, value in matches:
            param = param.strip()
            value = value.strip()
            if (len(param) > 3 and len(param) < 50 and 
                len(value) > 1 and len(value) < 100 and
                not param.lower().startswith(('click', 'read', 'view', 'download'))):
                spec_data[param] = value
    
    return spec_data

def scrape_product_page(driver, wait, url):
    """Scrape a single product page with improved robustness"""
    print(f"\nProcessing {url}...")
    
    try:
        driver.get(url)
        wait_for_content_load(driver, 10)
        
        # Extract basic product info
        product_name, small_desc = extract_product_info(driver, wait)
        print(f"  Product: {product_name}")
        print(f"  Description: {small_desc[:100]}...")
        
        models = []
        specifications = {}
        
        # Try to get models from "All models" or "Models" tab
        model_tab_names = ["All models", "Models", "Variants", "Products"]
        models_found = False
        
        for tab_name in model_tab_names:
            if click_tab_advanced(driver, wait, tab_name):
                soup = BeautifulSoup(driver.page_source, "html.parser")
                models = extract_models_advanced(soup, driver)
                if models:
                    print(f"  Found {len(models)} models from '{tab_name}' tab")
                    models_found = True
                    break
        
        # If no models found, try extracting from main page
        if not models_found:
            soup = BeautifulSoup(driver.page_source, "html.parser")
            models = extract_models_advanced(soup, driver)
            if models:
                print(f"  Found {len(models)} models from main page")
        
        # Create default model if none found
        if not models:
            models = [{"Model": "Standard", "Product Name": product_name}]
            print("  Using default model entry")
        
        # Try to get specifications from "Specifications" or "Specs" tab
        spec_tab_names = ["Specifications", "Specs", "Technical", "Details"]
        specs_found = False
        
        for tab_name in spec_tab_names:
            if click_tab_advanced(driver, wait, tab_name):
                soup = BeautifulSoup(driver.page_source, "html.parser")
                specifications = extract_specifications_advanced(soup)
                if specifications:
                    print(f"  Found {len(specifications)} specifications from '{tab_name}' tab")
                    specs_found = True
                    break
        
        # If no specs found, try main page
        if not specs_found:
            soup = BeautifulSoup(driver.page_source, "html.parser")
            specifications = extract_specifications_advanced(soup)
            if specifications:
                print(f"  Found {len(specifications)} specifications from main page")
        
        # Create output data
        page_data = []
        for model in models:
            row_data = {
                "Product": product_name,
                "Small Description": small_desc,
                "Model Code": model["Model"],
                "Model Name": model["Product Name"]
            }
            
            # Add all specifications
            for spec_name, spec_value in specifications.items():
                row_data[spec_name] = spec_value
            
            page_data.append(row_data)
        
        print(f"  Created {len(page_data)} data rows")
        return page_data
        
    except Exception as e:
        print(f"  Error processing {url}: {str(e)}")
        return []

def main():
    """Main scraping function with improved error handling"""
    driver = setup_driver()
    if not driver:
        print("Failed to setup driver")
        return
    
    wait = WebDriverWait(driver, 20)
    all_rows = []
    
    try:
        for i, url in enumerate(product_urls, 1):
            print(f"\n{'='*60}")
            print(f"Processing page {i}/{len(product_urls)}")
            print(f"{'='*60}")
            
            page_data = scrape_product_page(driver, wait, url)
            all_rows.extend(page_data)
            
            # Be respectful to the server
            if i < len(product_urls):
                print(f"  Waiting 3 seconds before next page...")
                time.sleep(3)
        
        if all_rows:
            # Create DataFrame
            df = pd.DataFrame(all_rows)
            
            # Clean up column names
            df.columns = [col.strip() for col in df.columns]
            
            # Fill NaN values with empty strings for better CSV output
            df = df.fillna('')
            
            # Save main CSV
            filename = "hbk_2255_products_fixed.csv"
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"\n{'='*60}")
            print(f"SUCCESS: Saved {len(all_rows)} records to {filename}")
            
            # Save backup with all raw data
            backup_filename = "hbk_2255_products_backup.csv"
            with open(backup_filename, 'w', newline='', encoding='utf-8-sig') as f:
                if all_rows:
                    # Get all unique keys across all records
                    all_keys = set()
                    for row in all_rows:
                        all_keys.update(row.keys())
                    
                    writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
                    writer.writeheader()
                    for row in all_rows:
                        # Fill missing keys with empty strings
                        complete_row = {key: row.get(key, '') for key in all_keys}
                        writer.writerow(complete_row)
            
            print(f"Backup saved to {backup_filename}")
            
            # Print summary
            print(f"\nSCRAPING SUMMARY:")
            print(f"- Pages processed: {len(product_urls)}")
            print(f"- Records extracted: {len(all_rows)}")
            print(f"- Unique columns: {len(df.columns)}")
            print(f"- Column names: {list(df.columns)}")
            
            # Show sample of specifications found
            spec_columns = [col for col in df.columns if col not in ['Product', 'Small Description', 'Model Code', 'Model Name']]
            if spec_columns:
                print(f"\nSpecification columns found:")
                for col in spec_columns[:10]:  # Show first 10
                    non_empty = df[col].dropna().iloc[:1].values
                    sample = non_empty[0] if len(non_empty) > 0 else "N/A"
                    print(f"  - {col}: {sample}")
                if len(spec_columns) > 10:
                    print(f"  ... and {len(spec_columns) - 10} more")
        else:
            print("\nERROR: No data was extracted from any of the pages")
            
    except Exception as e:
        print(f"\nCritical error during scraping: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        driver.quit()
        print(f"\nDriver closed successfully")

if __name__ == "__main__":
    main()