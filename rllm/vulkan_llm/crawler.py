#!/usr/bin/env python3
"""
Web crawler to fetch and save training data for the LLM.
Supports crawling websites, extracting text, and saving as .trdata files.
"""

import os
import sys
import time
import json
import urllib.request
import urllib.parse
import urllib.error
from urllib.robotparser import RobotFileParser
from html.parser import HTMLParser
from pathlib import Path
from datetime import datetime
import difflib
from collections import deque
import random

class TextExtractor(HTMLParser):
    """Extract text from HTML while ignoring scripts, styles, and metadata."""
    
    def __init__(self):
        super().__init__()
        self.text = []
        self.skip = False
        self.skip_tags = {'script', 'style', 'head', 'meta', 'noscript', 'nav'}
    
    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.skip_tags:
            self.skip = True
    
    def handle_endtag(self, tag):
        if tag.lower() in self.skip_tags:
            self.skip = False
    
    def handle_data(self, data):
        if not self.skip:
            text = data.strip()
            if text:
                self.text.append(text)
    
    def get_text(self):
        return '\n'.join(self.text)


class WebCrawler:
    """Crawl websites and extract text for training data."""
    
    def __init__(self, output_dir='TrainingData', rate_limit=2.0):
        self.output_dir = output_dir
        self.rate_limit = rate_limit
        self.visited = set()
        os.makedirs(output_dir, exist_ok=True)
    
    def _get_safe_filename(self, url):
        """Convert URL to a safe filename."""
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.replace('www.', '').split('.')[0]
        path = parsed.path.strip('/').replace('/', '_')[:50]
        timestamp = datetime.now().strftime('%s')
        return f"{domain}_{path}_{timestamp}.trdata" if path else f"{domain}_{timestamp}.trdata"
    
    def _extract_text(self, html):
        """Extract text from HTML content."""
        parser = TextExtractor()
        try:
            parser.feed(html)
            return parser.get_text()
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            return ""
    
    def _can_fetch(self, url):
        """Check if URL should be fetched (respecting robots.txt)."""
        try:
            parsed = urllib.parse.urlparse(url)
            robot_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            rp = RobotFileParser()
            rp.set_url(robot_url)
            rp.read()
            return rp.can_fetch("*", url)
        except:
            return True
    
    def fetch_url(self, url, timeout=15):
        """Fetch and extract text from a URL."""
        if url in self.visited:
            print(f"Already visited: {url}")
            return None
        
        if not self._can_fetch(url):
            print(f"robots.txt disallows: {url}")
            return None
        
        print(f"Fetching: {url}")
        self.visited.add(url)
        
        try:
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (Training Data Crawler)'}
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                html = response.read().decode('utf-8', errors='ignore')
                text = self._extract_text(html)
                
                if text and len(text) > 100:
                    filename = self._get_safe_filename(url)
                    filepath = os.path.join(self.output_dir, filename)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(text)
                    print(f"Saved: {filepath} ({len(text)} chars)")
                    time.sleep(self.rate_limit)
                    return filepath
                else:
                    print(f"No significant text extracted from {url}")
                    return None
        
        except urllib.error.URLError as e:
            print(f"URL Error: {url} - {e}")
        except urllib.error.HTTPError as e:
            print(f"HTTP Error: {url} - {e.code}")
        except Exception as e:
            print(f"Error fetching {url}: {e}")
        
        return None
    
    def crawl_urls(self, urls):
        """Crawl a list of URLs."""
        results = []
        for url in urls:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            result = self.fetch_url(url)
            if result:
                results.append(result)
        return results
    
    def crawl_wikipedia(self, topics):
        """Crawl Wikipedia topics with fuzzy matching and depth-limited link traversal.

        For each provided topic this performs a fuzzy search to find the best-matching
        Wikipedia page (title similarity). If a match is found the crawler does a
        breadth-first traversal of links up to depth 2 and saves pages whose title
        or content is similar to the original topic. Topics without an acceptable
        initial match are reported at the end as "not accepted".
        """

        api_url = "https://en.wikipedia.org/w/api.php"
        results = []
        rejected = []

        def _similar(a, b):
            return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

        def _token_overlap(topic, text):
            toks = [t for t in topic.lower().split() if t]
            if not toks:
                return 0.0
            count = 0
            lt = text.lower()
            for t in toks:
                if t in lt:
                    count += 1
            return count / len(toks)

        def _urlopen_with_retries(req, timeout=15, max_retries=5):
            backoff = 1.0
            for attempt in range(max_retries):
                try:
                    return urllib.request.urlopen(req, timeout=timeout)
                except urllib.error.HTTPError as e:
                    code = getattr(e, 'code', None)
                    if code in (429, 502, 503, 504):
                        sleep = backoff + random.random() * 0.5
                        print(f"HTTP {code} from {req.full_url}, retrying in {sleep:.1f}s (attempt {attempt+1})")
                        time.sleep(sleep)
                        backoff *= 2
                        continue
                    raise
                except urllib.error.URLError as e:
                    sleep = backoff + random.random() * 0.5
                    print(f"URL error {e} for {getattr(req, 'full_url', str(req))}, retrying in {sleep:.1f}s (attempt {attempt+1})")
                    time.sleep(sleep)
                    backoff *= 2
                    continue
            raise Exception(f"Max retries exceeded for {getattr(req, 'full_url', str(req))}")

        def fetch_extract(title):
            params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts',
                'explaintext': 1,
            }
            q = f"{api_url}?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(q, headers={'User-Agent': 'Mozilla/5.0 (Training Data Crawler)'} )
            with _urlopen_with_retries(req, timeout=15) as resp:
                return json.loads(resp.read().decode('utf-8'))

        def fetch_links(title, limit=50):
            # Fetch links from a page (first `limit` links)
            params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'links',
                'pllimit': min(limit, 500)
            }
            q = f"{api_url}?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(q, headers={'User-Agent': 'Mozilla/5.0 (Training Data Crawler)'} )
            with _urlopen_with_retries(req, timeout=15) as resp:
                return json.loads(resp.read().decode('utf-8'))

        def fallback_random_crawl(attempts=3, wait_between=3):
            """When primary wiki requests fail, do a small number of random web fetches
            (using a small seed list) to avoid getting completely blocked, then return."""
            seed_urls = [
                'https://en.wikipedia.org/wiki/Special:Random',
                'https://www.bbc.com/',
                'https://arxiv.org/',
                'https://www.nature.com/',
                'https://www.sciencedaily.com/'
            ]
            print(f"Fallback: performing {attempts} random web fetches to cool down...")
            for i in range(attempts):
                url = random.choice(seed_urls)
                try:
                    self.fetch_url(url)
                except Exception as e:
                    print(f"Fallback fetch error: {e}")
                time.sleep(wait_between)

        for topic in topics:
            print(f"\nSearching Wikipedia for topic: {topic}")

            # Try searching, with a single fallback to random crawling on failure
            search_attempts = 0
            max_search_attempts = 2
            data = None
            while search_attempts < max_search_attempts:
                try:
                    # use search API to find candidate titles
                    params = {
                        'action': 'query',
                        'list': 'search',
                        'srsearch': topic,
                        'srlimit': 8,
                        'format': 'json'
                    }
                    q = f"{api_url}?" + urllib.parse.urlencode(params)
                    req = urllib.request.Request(q, headers={'User-Agent': 'Mozilla/5.0 (Training Data Crawler)'} )
                    with _urlopen_with_retries(req, timeout=15) as resp:
                        data = json.loads(resp.read().decode('utf-8'))
                    break
                except Exception as e:
                    search_attempts += 1
                    print(f"Search attempt {search_attempts} failed for '{topic}': {e}")
                    if search_attempts < max_search_attempts:
                        # perform fallback random crawling to cooldown before retrying
                        try:
                            fallback_random_crawl(attempts=3, wait_between=2)
                        except Exception as fe:
                            print(f"Fallback crawling also failed: {fe}")
                        print("Retrying Wikipedia search after fallback...")
                    else:
                        print(f"Giving up on topic '{topic}' after {search_attempts} attempts")

                if not data:
                    candidates = []
                else:
                    candidates = [r.get('title') for r in data.get('query', {}).get('search', [])]

                # pick best candidate by title similarity
                best = None
                best_score = 0.0
                for cand in candidates:
                    score = _similar(topic, cand)
                    if score > best_score:
                        best_score = score
                        best = cand

                title_threshold = 0.55
                if not best or best_score < title_threshold:
                    print(f"No close Wikipedia title match for '{topic}' (best='{best}' score={best_score:.2f})")
                    rejected.append(topic)
                    continue

                print(f"Using root page: '{best}' (score={best_score:.2f})")

                # BFS up to depth 2
                max_depth = 2
                max_links_per_page = 40
                visited_titles = set()
                q = deque()
                q.append((best, 0))

                while q:
                    cur_title, depth = q.popleft()
                    if cur_title in visited_titles:
                        continue
                    visited_titles.add(cur_title)

                    # fetch extract
                    try:
                        data = fetch_extract(cur_title)
                        pages = data.get('query', {}).get('pages', {})
                        for pid, page in pages.items():
                            if 'missing' in page:
                                continue
                            text = page.get('extract', '') or ''
                            title = page.get('title', cur_title)

                            title_sim = _similar(topic, title)
                            token_ov = _token_overlap(topic, text)

                            accept = False
                            # accept if title is similar or token overlap is significant
                            if title_sim >= 0.6 or token_ov >= 0.2 or (title_sim >= 0.5 and token_ov >= 0.1):
                                accept = True

                            if accept and text and len(text) > 100:
                                safe = title.replace(' ', '_')[:60]
                                filename = f"wikipedia_{safe}.trdata"
                                filepath = os.path.join(self.output_dir, filename)
                                with open(filepath, 'w', encoding='utf-8') as f:
                                    f.write(text)
                                print(f"Saved: {filepath} ({len(text)} chars) [depth={depth} title_sim={title_sim:.2f} tok_ov={token_ov:.2f}]")
                                results.append(filepath)
                                time.sleep(self.rate_limit)

                        # enqueue links if depth < max_depth
                        if depth < max_depth:
                            link_data = fetch_links(cur_title, limit=max_links_per_page)
                            pages = link_data.get('query', {}).get('pages', {})
                            for pid, page in pages.items():
                                for link in page.get('links', [])[:max_links_per_page]:
                                    ltitle = link.get('title')
                                    if ltitle and ltitle not in visited_titles:
                                        q.append((ltitle, depth + 1))

                    except Exception as e:
                        print(f"Error processing page '{cur_title}': {e}")
                        continue

        if rejected:
            print("\nNot accepted topics:")
            for t in rejected:
                print(f"  - {t}")

        return results
    
    def crawl_subreddit(self, subreddit, limit=50):
        """Crawl Reddit subreddit posts (note: Reddit may rate-limit bots)."""
        print(f"\nCrawling subreddit: r/{subreddit}")
        results = []
        try:
            url = f"https://www.reddit.com/r/{subreddit}/top.json?t=all&limit={min(limit, 100)}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            req = urllib.request.Request(url, headers=headers)
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with urllib.request.urlopen(req, timeout=15) as response:
                        data = json.loads(response.read().decode('utf-8'))
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 403 and attempt < max_retries - 1:
                        print(f"  Rate limited, retrying in 5s...")
                        time.sleep(5)
                    else:
                        raise
            
            posts = data.get('data', {}).get('children', [])
            combined_text = f"Reddit: r/{subreddit}\n\n"
            
            for post in posts:
                post_data = post.get('data', {})
                title = post_data.get('title', '')
                selftext = post_data.get('selftext', '')
                score = post_data.get('score', 0)
                
                if title and (selftext or score > 100):
                    combined_text += f"[{score} upvotes] {title}\n"
                    if selftext:
                        combined_text += f"{selftext}\n"
                    combined_text += "\n---\n\n"
            
            if len(combined_text) > 500:
                filename = f"reddit_r{subreddit}.trdata"
                filepath = os.path.join(self.output_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(combined_text)
                print(f"Saved: {filepath} ({len(combined_text)} chars)")
                results.append(filepath)
                time.sleep(self.rate_limit)
            else:
                print(f"Not enough content from r/{subreddit}")
        
        except Exception as e:
            print(f"Error crawling r/{subreddit}: {e} (Reddit may be blocking bot access)")
        
        return results
    
    def crawl_hackernews(self, limit=30):
        """Crawl HackerNews top stories (more crawler-friendly)."""
        print(f"\nCrawling HackerNews (top {limit} stories)")
        results = []
        try:
            url = "https://news.ycombinator.com/rss"
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                xml = response.read().decode('utf-8')
            
            import re
            # Extract titles and links from RSS
            titles = re.findall(r'<title>([^<]+)</title>', xml)
            combined_text = f"HackerNews Top Stories\n\n"
            
            for i, title in enumerate(titles[1:limit+1], 1):
                combined_text += f"{i}. {title}\n"
            
            if len(combined_text) > 200:
                filename = "hackernews_top.trdata"
                filepath = os.path.join(self.output_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(combined_text)
                print(f"Saved: {filepath} ({len(combined_text)} chars)")
                results.append(filepath)
        
        except Exception as e:
            print(f"Error crawling HackerNews: {e}")
        
        return results
    
    def crawl_stackoverflow(self, tag, limit=20):
        """Crawl Stack Overflow questions and answers."""
        print(f"\nCrawling Stack Overflow: tag '{tag}'")
        results = []
        try:
            # Use Stack Exchange API
            url = f"https://api.stackexchange.com/2.3/questions?order=desc&sort=votes&tagged={tag}&site=stackoverflow&pagesize={min(limit, 100)}"
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            combined_text = f"Stack Overflow: {tag}\n\n"
            
            for q in data.get('items', [])[:limit]:
                title = q.get('title', '')
                score = q.get('score', 0)
                # Unescape HTML entities
                title = title.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
                combined_text += f"[{score} votes] {title}\n"
            
            if len(combined_text) > 200:
                filename = f"stackoverflow_{tag}.trdata"
                filepath = os.path.join(self.output_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(combined_text)
                print(f"Saved: {filepath} ({len(combined_text)} chars)")
                results.append(filepath)
                time.sleep(self.rate_limit)
        
        except Exception as e:
            print(f"Error crawling Stack Overflow: {e}")
        
        return results

        """Crawl ArXiv paper summaries."""
        print(f"\nSearching ArXiv for: {query}")
        results = []
        try:
            encoded_query = urllib.parse.quote(query)
            search_url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={max_papers}"
            req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                xml = response.read().decode('utf-8')
            
            # Simple XML parsing for summaries
            import re
            summaries = re.findall(r'<summary>(.*?)</summary>', xml, re.DOTALL)
            
            if summaries:
                combined_text = f"ArXiv Papers on {query}:\n\n"
                for i, summary in enumerate(summaries, 1):
                    clean = re.sub(r'\n\s+', ' ', summary.strip())
                    combined_text += f"{i}. {clean}\n\n"
                
                filename = f"arxiv_{query.replace(' ', '_')}.trdata"
                filepath = os.path.join(self.output_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(combined_text)
                print(f"Saved: {filepath} ({len(combined_text)} chars)")
                results.append(filepath)
                time.sleep(self.rate_limit)
        
        except Exception as e:
            print(f"Error crawling ArXiv: {e}")
        
        return results
    
    def status(self):
        """Print status of downloaded data."""
        files = list(Path(self.output_dir).glob('*.trdata'))
        total_size = sum(f.stat().st_size for f in files)
        print(f"\nTrainingData Status:")
        print(f"  Files: {len(files)}")
        print(f"  Total size: {total_size / 1024 / 1024:.2f} MB")
        for f in sorted(files):
            print(f"    {f.name}: {f.stat().st_size / 1024:.1f} KB")


def main():
    """Interactive crawler CLI."""
    crawler = WebCrawler()
    
    print("=" * 60)
    print("LLM Training Data Web Crawler")
    print("=" * 60)
    
    while True:
        print("\nOptions:")
        print("1. Fetch URL(s)")
        print("2. Crawl Wikipedia topics")
        print("3. Crawl ArXiv papers")
        print("4. Crawl Reddit subreddits")
        print("5. Crawl HackerNews")
        print("6. Crawl Stack Overflow")
        print("7. Status")
        print("8. Exit")
        
        choice = input("\nEnter choice (1-8): ").strip()
        
        if choice == '1':
            urls = input("Enter URLs (comma-separated): ").split(',')
            urls = [u.strip() for u in urls if u.strip()]
            crawler.crawl_urls(urls)
        
        elif choice == '2':
            topics = input("Enter Wikipedia topics (comma-separated): ").split(',')
            topics = [t.strip() for t in topics if t.strip()]
            crawler.crawl_wikipedia(topics)
        
        elif choice == '3':
            query = input("Enter ArXiv search query: ").strip()
            if query:
                max_papers = input("Max papers (default 5): ").strip()
                max_papers = int(max_papers) if max_papers else 5
                crawler.crawl_arxiv_summaries(query, max_papers)
        
        elif choice == '4':
            subreddits = input("Enter subreddits (comma-separated, no r/ prefix): ").split(',')
            subreddits = [s.strip() for s in subreddits if s.strip()]
            if subreddits:
                limit = input("Posts per subreddit (default 50): ").strip()
                limit = int(limit) if limit else 50
                crawler.crawl_subreddit_list(subreddits, limit)
        
        elif choice == '5':
            limit = input("Number of stories (default 30): ").strip()
            limit = int(limit) if limit else 30
            crawler.crawl_hackernews(limit)
        
        elif choice == '6':
            tag = input("Enter Stack Overflow tag: ").strip()
            if tag:
                limit = input("Number of questions (default 20): ").strip()
                limit = int(limit) if limit else 20
                crawler.crawl_stackoverflow(tag, limit)
        
        elif choice == '7':
            crawler.status()
        
        elif choice == '8':
            print("Done!")
            break
        
        else:
            print("Invalid choice")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # CLI mode for batch operations
        if sys.argv[1] == 'url' and len(sys.argv) > 2:
            crawler = WebCrawler()
            crawler.crawl_urls(sys.argv[2:])
        elif sys.argv[1] == 'wiki' and len(sys.argv) > 2:
            crawler = WebCrawler()
            crawler.crawl_wikipedia(sys.argv[2:])
        elif sys.argv[1] == 'arxiv' and len(sys.argv) > 2:
            crawler = WebCrawler()
            query = ' '.join(sys.argv[2:])
            crawler.crawl_arxiv_summaries(query)
        elif sys.argv[1] == 'sub' and len(sys.argv) > 2:
            crawler = WebCrawler()
            crawler.crawl_subreddits(sys.argv[2:])
        elif sys.argv[1] == 'hn':
            crawler = WebCrawler()
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 30
            crawler.crawl_hackernews(limit)
        elif sys.argv[1] == 'so' and len(sys.argv) > 2:
            crawler = WebCrawler()
            for tag in sys.argv[2:]:
                crawler.crawl_stackoverflow(tag)
        else:
            print("Usage:")
            print("  python crawler.py                           # Interactive mode")
            print("  python crawler.py url <url1> <url2> ...     # Fetch URLs")
            print("  python crawler.py wiki <topic1> <topic2> ...# Wikipedia")
            print("  python crawler.py arxiv <query>             # ArXiv papers")
            print("  python crawler.py sub <sub1> <sub2> ...     # Reddit subreddits")
            print("  python crawler.py hn [limit]                # HackerNews (default 30)")
            print("  python crawler.py so <tag1> <tag2> ...      # Stack Overflow")
    else:
        main()
