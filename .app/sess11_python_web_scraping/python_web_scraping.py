"""
================================================================================
Python script to demonstrate scraping a website and storing data to a file
================================================================================


Requirements
--------------------------
 pip install requests beautifulsoup4

Author : Nyanjui
Date   : 25 September 2026
"""
# ================================================================================
# SECTION 0: Import the required modules
# ================================================================================
import csv
from asyncio import timeout

import requests
import time
from bs4 import BeautifulSoup
from pathlib import Path


# ================================================================================
# SECTION 1: Set up a polite user agent string
# ================================================================================
headers = {
   "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; X64) Nyanjui-Learning-Scraper/1.0"
}

# ================================================================================
# SECTION 2: Specify the target URL (Website you wish to scrape)
# ================================================================================
url = "https://books.toscrape.com"

# ================================================================================
# SECTION 3: Make the request
# ================================================================================
try:
   response = requests.get(url, headers=headers, timeout=10)
   response.raise_for_status()
except requests.exceptions.RequestException as err:
   print(f"Request failed due to:\n{err}")
   exit(1)

# ================================================================================
# SECTION 5: Parse the HTML using beautifulsoup
# ================================================================================
soup = BeautifulSoup(response.text, "html.parser")

# ================================================================================
# SECTION 6: 😊 Fun part now. Extract the data
# ================================================================================
books = soup.find_all("h3")
prices = soup.find_all("p", class_="price_color")

# ================================================================================
# SECTION 7: Display the book titles then their prices
# ================================================================================
print("First 8 book titles")
print("-" * 55)
for book in books[:8]:
   title = book.a["title"]
   print(f"* {title}")
print("Prices of the first 8 books")
print("-" * 55)
for price in prices[:8]:
   print(f"* {price.get_text().strip().replace('Â', '')}")

# ================================================================================
# SECTION 8: Combine the prices and titles safely
# ================================================================================
book_data = []
for book, price in zip(books, prices):
    title = book.a["title"].strip()
    price_text = price.get_text().strip().replace('Â', '')
    book_data.append([title, price_text])

# ================================================================================
# SECTION 9: 💾Save the data to a CSV file (can be used for further analysis)
# ================================================================================
# a. Build the path to the files folder where we will save our scraped data
files_dir = Path.cwd().parent / "files"
files_dir.mkdir(exist_ok=True)

csv_path = files_dir / "scraped_books.csv"

with open(csv_path, "w", newline="", encoding="utf-8-sig") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Title", "Price"])
    writer.writerows(book_data)

# Inform the user the data has been saved
print(f"Saved {len(book_data)} books/records to {csv_path}")

time.sleep(2) # polite delay
