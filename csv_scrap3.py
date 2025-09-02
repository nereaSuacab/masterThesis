# List of product page URLs
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

# URLs for HBK software apps
product_urls = [
    "https://www.hbkworld.com/en/products/instruments/handheld/hand-held-software/desktop-applications/dirac-room-acoustics-software-7841",
    "https://www.hbkworld.com/en/products/instruments/handheld/hand-held-software/desktop-applications/measurement-partner-suite-bz-5503"
]

def setup_driver():
    """Setup Chrome driver with optimal configuration"""
    options = Options()
    # Comment out headless for debugging
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

def handle_cookie_banners(driver):
    """Handle cookie consent banners"""
    cookie_selectors = [
        "button[id*='accept']",
        "button[class*='accept']",
        "button[id*='cookie']", 
        "button[class*='cookie']",
        ".cookie-consent button",
        ".cookie-banner button",
        "#cookie-consent button",
    ]
    
    for selector in cookie_selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for element in elements:
                try:
                    if element.is_displayed() and element.is_enabled():
                        driver.execute_script("arguments[0].click();", element)
                        time.sleep(1)
                        print("  Handled cookie banner")
                        return True
                except Exception:
                    continue
        except Exception:
            continue
    
    return False

def extract_product_name(driver, wait):
    """Extract product name with multiple strategies"""
    name_selectors = [
        "h1.cmp-title__text",
        "h1[data-cmp-hook-title='title']",
        "h1.product-title",
        "h1.page-title",
        "h1",
        ".hero-title h1",
        ".product-hero h1",
        ".page-header h1"
    ]
    
    for selector in name_selectors:
        try:
            element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
            product_name = element.text.strip()
            if product_name and len(product_name) > 3:
                return product_name
        except (NoSuchElementException, TimeoutException):
            continue
    
    return "N/A"

def extract_product_description(driver):
    """Extract product description"""
    desc_selectors = [
        "div.cmp-text p",
        ".product-intro p",
        ".hero-content p",
        "main .cmp-text p",
        ".product-description p",
        ".intro-text p",
        ".page-content p"
    ]
    
    for selector in desc_selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for element in elements:
                text = element.text.strip()
                # Filter for good description text
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
                    not re.match(r'^[\d\s\-\+\(\)]+$', text)):
                    return text
        except NoSuchElementException:
            continue
    
    return "N/A"

def click_tab(driver, wait, tab_text):
    """Try to click a tab with the given text"""
    print(f"  Looking for '{tab_text}' tab...")
    
    strategies = [
        (By.XPATH, f"//button[contains(normalize-space(text()), '{tab_text}')]"),
        (By.XPATH, f"//a[contains(normalize-space(text()), '{tab_text}')]"),
        (By.XPATH, f"//*[contains(@class, 'tab')][contains(normalize-space(text()), '{tab_text}')]"),
        (By.XPATH, f"//li[contains(normalize-space(text()), '{tab_text}')]//a"),
        (By.XPATH, f"//*[contains(translate(normalize-space(text()), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{tab_text.lower()}')]"),
    ]
    
    for by_method, selector in strategies:
        try:
            elements = driver.find_elements(by_method, selector)
            for element in elements:
                try:
                    if element.is_displayed() and element.is_enabled():
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
                        time.sleep(1)
                        driver.execute_script("arguments[0].click();", element)
                        time.sleep(2)  # Wait for content to load
                        print(f"  Successfully clicked '{tab_text}' tab")
                        return True
                except Exception:
                    continue
        except Exception:
            continue
    
    print(f"  '{tab_text}' tab not found")
    return False

def extract_models_from_page(soup):
    """Extract models from the current page content"""
    models = []
    
    # Strategy 1: Look for tables with model information
    tables = soup.find_all("table")
    for table in tables:
        headers = []
        header_elements = table.find_all("th")
        if header_elements:
            headers = [th.get_text(strip=True).lower() for th in header_elements]
        
        # Look for model-like headers
        if any(keyword in ' '.join(headers) for keyword in ['code', 'model', 'part', 'product', 'type']):
            rows = table.find_all("tr")
            data_rows = rows[1:] if header_elements else rows
            
            for row in data_rows:
                cells = row.find_all("td")
                if len(cells) >= 1:
                    # First cell is usually the model code
                    code_text = cells[0].get_text(strip=True)
                    name_text = cells[1].get_text(strip=True) if len(cells) > 1 else code_text
                    
                    # Filter valid model codes
                    if (code_text and 
                        len(code_text) > 2 and 
                        code_text.lower() not in ['code', 'model', 'part', 'product'] and
                        # Look for BZ codes or other HBK patterns
                        (re.search(r'BZ[- ]?\d+', code_text, re.IGNORECASE) or 
                         re.search(r'\d{4}', code_text))):
                        models.append({
                            "Model": code_text,
                            "Product Name": name_text
                        })
    
    # Strategy 2: Look for download links or product codes in links
    if not models:
        links = soup.find_all("a")
        for link in links:
            link_text = link.get_text(strip=True)
            href = link.get('href', '')
            
            # Look for BZ codes or model patterns
            bz_match = re.search(r'(BZ[- ]?\d+)', link_text, re.IGNORECASE)
            if bz_match:
                models.append({
                    "Model": bz_match.group(1),
                    "Product Name": link_text
                })
    
    # Strategy 3: Look for model codes in any text content
    if not models:
        all_text = soup.get_text()
        bz_codes = re.findall(r'(BZ[- ]?\d+)', all_text, re.IGNORECASE)
        for code in set(bz_codes):  # Remove duplicates
            models.append({
                "Model": code,
                "Product Name": f"Model {code}"
            })
    
    return models

def extract_model_from_url(url):
    """Extract model code from the URL as fallback"""
    # Look for BZ codes or model numbers in the URL
    bz_match = re.search(r'bz-(\d+)', url, re.IGNORECASE)
    if bz_match:
        return f"BZ-{bz_match.group(1)}"
    
    # Look for other number patterns
    number_match = re.search(r'/([a-z]+-)?(\d{4})[/-]', url)
    if number_match:
        return number_match.group(2)
    
    return "Standard"

def scrape_product_page(driver, wait, url):
    """Scrape basic product information from a single page"""
    print(f"\nProcessing {url}...")
    
    try:
        driver.get(url)
        time.sleep(3)  # Initial load
        
        # Handle cookie banners
        handle_cookie_banners(driver)
        
        # Extract basic product info
        product_name = extract_product_name(driver, wait)
        product_desc = extract_product_description(driver)
        
        print(f"  Product: {product_name}")
        print(f"  Description: {product_desc[:100]}...")
        
        models = []
        
        # Try to find models in "All models" tab first
        if click_tab(driver, wait, "All models"):
            soup = BeautifulSoup(driver.page_source, "html.parser")
            models = extract_models_from_page(soup)
            if models:
                print(f"  Found {len(models)} models from 'All models' tab")
        
        # If no models found, try "Models" tab
        if not models and click_tab(driver, wait, "Models"):
            soup = BeautifulSoup(driver.page_source, "html.parser")
            models = extract_models_from_page(soup)
            if models:
                print(f"  Found {len(models)} models from 'Models' tab")
        
        # If still no models, try extracting from main page
        if not models:
            soup = BeautifulSoup(driver.page_source, "html.parser")
            models = extract_models_from_page(soup)
            if models:
                print(f"  Found {len(models)} models from main page")
        
        # If still no models, create default from URL or product name
        if not models:
            model_code = extract_model_from_url(url)
            models = [{"Model": model_code, "Product Name": product_name}]
            print(f"  Using fallback model: {model_code}")
        
        # Create output data
        page_data = []
        for model in models:
            row_data = {
                "Product": product_name,
                "Small Description": product_desc,
                "Model Code": model["Model"],
                "Model Name": model["Product Name"]
            }
            page_data.append(row_data)
        
        print(f"  Created {len(page_data)} data rows")
        return page_data
        
    except Exception as e:
        print(f"  Error processing {url}: {str(e)}")
        # Return at least basic info even if there's an error
        return [{
            "Product": "Error - Could not extract",
            "Small Description": f"Error processing {url}",
            "Model Code": extract_model_from_url(url),
            "Model Name": "Error"
        }]

def main():
    """Main scraping function"""
    driver = setup_driver()
    if not driver:
        print("Failed to setup driver")
        return
    
    wait = WebDriverWait(driver, 20)
    all_rows = []
    
    try:
        for i, url in enumerate(product_urls, 1):
            print(f"\n{'='*80}")
            print(f"Processing page {i}/{len(product_urls)}")
            print(f"{'='*80}")
            
            page_data = scrape_product_page(driver, wait, url)
            all_rows.extend(page_data)
            
            # Be respectful to the server
            if i < len(product_urls):
                print(f"  Waiting 3 seconds before next page...")
                time.sleep(3)
        
        if all_rows:
            # Create DataFrame
            df = pd.DataFrame(all_rows)
            
            # Save to CSV
            filename = "hbk_software_apps_basic.csv"
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"\n{'='*80}")
            print(f"SUCCESS: Saved {len(all_rows)} records to {filename}")
            
            # Save backup
            backup_filename = "hbk_software_apps_backup.csv"
            df.to_csv(backup_filename, index=False, encoding='utf-8-sig')
            
            # Print summary
            print(f"\nSCRAPING SUMMARY:")
            print(f"- Pages processed: {len(product_urls)}")
            print(f"- Records extracted: {len(all_rows)}")
            print(f"- Columns: {list(df.columns)}")
            
            # Show sample of data
            print(f"\nSAMPLE DATA:")
            print(df.head().to_string(index=False, max_colwidth=50))
            
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