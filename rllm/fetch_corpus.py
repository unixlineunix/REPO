import urllib.request
import urllib.parse
import json
import os
import time

def fetch_url(url):
    print(f"Fetching {url}...")
    time.sleep(1.0) # sleep 1 second to respect rate limits
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            return response.read().decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return ""

def fetch_wikipedia_article(title):
    print(f"Fetching Wikipedia article: {title}...")
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles={urllib.parse.quote(title)}&format=json"
    try:
        data = fetch_url(url)
        if not data:
            return ""
        js = json.loads(data)
        pages = js.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            return page.get("extract", "")
    except Exception as e:
        print(f"Failed to fetch Wikipedia article {title}: {e}")
    return ""

def main():
    # 1. Start with original Shakespeare corpus if available
    original_path = "../corpus.txt"
    combined_text = ""
    if os.path.exists(original_path):
        print("Reading original Shakespeare corpus...")
        with open(original_path, "r", encoding="utf-8", errors="ignore") as f:
            combined_text += f.read() + "\n\n"
    
    # 2. Gutenberg Books
    gutenberg_books = [
        ("Alice in Wonderland", "https://www.gutenberg.org/cache/epub/11/pg11.txt"),
        ("Frankenstein", "https://www.gutenberg.org/cache/epub/84/pg84.txt"),
        ("Sherlock Holmes", "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"),
        ("The Time Machine", "https://www.gutenberg.org/cache/epub/35/pg35.txt")
    ]
    
    for name, url in gutenberg_books:
        text = fetch_url(url)
        if text:
            combined_text += f"\n\n--- BOOK: {name} ---\n\n" + text + "\n\n"
            
    # 3. Wikipedia Articles about Computer Systems & CS
    wiki_articles = [
        "Computer system",
        "Operating system",
        "Central processing unit",
        "Computer architecture",
        "Random-access memory",
        "Machine learning",
        "Vulkan (API)",
        "CUDA",
        "Linux",
        "Linux kernel",
        "Bash (Unix shell)",
        "C (programming language)",
        "C++",
        "Rust (programming language)"
    ]
    
    for article in wiki_articles:
        text = fetch_wikipedia_article(article)
        if text:
            combined_text += f"\n\n--- ARTICLE: {article} ---\n\n" + text + "\n\n"
            
    # 4. Save to local corpus.txt (replacing symlink)
    local_path = "corpus.txt"
    if os.path.islink(local_path):
        os.remove(local_path)
        
    print(f"Saving combined corpus to {local_path}...")
    with open(local_path, "w", encoding="utf-8") as f:
        f.write(combined_text)
        
    print(f"Corpus size: {len(combined_text)} characters. Done!")

if __name__ == "__main__":
    main()
