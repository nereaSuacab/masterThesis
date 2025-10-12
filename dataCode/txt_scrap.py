from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import re
import os

# URLs for HBK software apps
product_urls = [
   "https://www.hbkworld.com/en/products/instruments/handheld/vibration-meters/3656-a"
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

def extract_product_name(driver):
    """Extract product name from the page"""
    product_name = "Unknown_Product"
    
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
            name = element.text.strip()
            if name:
                product_name = name
                break
        except NoSuchElementException:
            continue
    
    return product_name

def sanitize_filename(filename):
    """Clean filename to make it valid for file system"""
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    # Remove extra underscores
    filename = re.sub(r'_+', '_', filename)
    # Remove leading/trailing underscores
    filename = filename.strip('_')
    return filename

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
            time.sleep(3)  # Wait for content to load
            return True
        except TimeoutException:
            continue
    
    print(f"Could not find or click '{tab_text}' tab")
    return False

def extract_overview_content(driver, soup):
    """Extract overview content from the page"""
    overview_content = []
    
    # First, try to find overview content in common containers
    overview_selectors = [
        "div.cmp-text",
        ".overview-content",
        ".product-overview",
        "main .cmp-text",
        "article .cmp-text",
        ".content .cmp-text"
    ]
    
    for selector in overview_selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for element in elements:
                text = element.text.strip()
                # Filter out navigation and short text
                if (text and 
                    len(text) > 30 and  # Must be substantial text
                    "HBKShop" not in text and 
                    "Training" not in text and 
                    "Resources" not in text and
                    "navigation" not in text.lower() and
                    "menu" not in text.lower() and
                    "cookie" not in text.lower() and
                    "accept" not in text.lower()):
                    overview_content.append(text)
        except NoSuchElementException:
            continue
    
    # Also try to extract from BeautifulSoup for better text handling
    text_divs = soup.find_all("div", class_="cmp-text")
    for div in text_divs:
        text = div.get_text(separator='\n', strip=True)
        if (text and 
            len(text) > 30 and
            text not in overview_content and  # Avoid duplicates
            "HBKShop" not in text and 
            "Training" not in text and 
            "Resources" not in text):
            overview_content.append(text)
    
    # Look for paragraphs and other text elements
    paragraphs = soup.find_all("p")
    for p in paragraphs:
        text = p.get_text(strip=True)
        if (text and 
            len(text) > 50 and  # Longer threshold for paragraphs
            text not in overview_content and
            "HBKShop" not in text and 
            "cookie" not in text.lower()):
            overview_content.append(text)
    
    return overview_content

def scrape_product_overview(driver, wait, url):
    """Scrape overview content from a single product page"""
    print(f"Processing {url}...")
    
    try:
        driver.get(url)
        time.sleep(3)  # Initial page load wait
        
        # Extract product name for filename
        product_name = extract_product_name(driver)
        print(f"  Product: {product_name}")
        
        # Get initial page content (usually contains overview by default)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        overview_content = extract_overview_content(driver, soup)
        
        # Try to click "Overview" tab if it exists
        if click_tab(driver, wait, "Overview"):
            print("  Successfully clicked Overview tab")
            soup = BeautifulSoup(driver.page_source, "html.parser")
            tab_overview_content = extract_overview_content(driver, soup)
            # Merge content, avoiding duplicates
            for content in tab_overview_content:
                if content not in overview_content:
                    overview_content.append(content)
        else:
            print("  No Overview tab found, using default page content")
        
        # If still no content, try clicking other tabs that might contain overview
        if not overview_content:
            for tab_name in ["Product", "Description", "About"]:
                if click_tab(driver, wait, tab_name):
                    print(f"  Trying {tab_name} tab for content")
                    soup = BeautifulSoup(driver.page_source, "html.parser")
                    tab_content = extract_overview_content(driver, soup)
                    overview_content.extend(tab_content)
                    if overview_content:
                        break
        
        if overview_content:
            print(f"  Found {len(overview_content)} content sections")
            
            # Create filename from product name
            safe_filename = sanitize_filename(product_name)
            filename = f"{safe_filename}.txt"
            
            # Ensure we don't overwrite files
            counter = 1
            base_filename = safe_filename
            while os.path.exists(filename):
                safe_filename = f"{base_filename}_{counter}"
                filename = f"{safe_filename}.txt"
                counter += 1
            
            # Write content to file in continuous format
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Product: {product_name}\n")
                f.write(f"URL: {url}\n")
                f.write("=" * 80 + "\n\n")
                
                # Join all content with double line breaks for readability
                continuous_text = '\n\n'.join(overview_content)
                f.write(continuous_text)
                f.write("\n")
            
            print(f"  Saved overview to: {filename}")
            return filename
        else:
            print("  No overview content found")
            return None
            
    except Exception as e:
        print(f"  Error processing {url}: {e}")
        return None

def main():
    """Main scraping function"""
    driver = setup_driver()
    if not driver:
        print("Failed to setup driver")
        return
    
    wait = WebDriverWait(driver, 15)
    saved_files = []
    
    try:
        print(f"Starting to scrape {len(product_urls)} product pages...\n")
        
        for url in product_urls:
            filename = scrape_product_overview(driver, wait, url)
            if filename:
                saved_files.append(filename)
            time.sleep(2)  # Be respectful to the server
            print()  # Add spacing between products
        
        # Print summary
        print("=" * 60)
        print("SCRAPING SUMMARY")
        print("=" * 60)
        print(f"Total products processed: {len(product_urls)}")
        print(f"Successfully saved files: {len(saved_files)}")
        
        if saved_files:
            print("\nSaved files:")
            for filename in saved_files:
                print(f"  - {filename}")
        else:
            print("\nNo files were saved. Check if the pages are accessible and contain overview content.")
            
    except Exception as e:
        print(f"Error during scraping: {e}")
    
    finally:
        driver.quit()
        print("\nDriver closed")

if __name__ == "__main__":
    main()