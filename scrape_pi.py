import os
import re
import sys
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.pi.website/"
START_URLS = [
    "https://www.pi.website/",
    "https://www.pi.website/blog",
    # You can add more known research URLs here if needed
]

# Where to store PDFs (change if you want a different folder)
DOWNLOAD_DIR = os.path.join(os.path.expanduser("~"), "Downloads", "pi_papers")

# Simple pattern for PDF links used on the site, e.g. /download/rlt.pdf, /download/Mem.pdf
PDF_EXT_RE = re.compile(r"\.pdf$", re.IGNORECASE)

# Match arXiv abstract URLs so we can rewrite them to direct PDF links
ARXIV_ABS_RE = re.compile(r"^https?://arxiv\.org/abs/(?P<id>[^?#]+)/?$", re.IGNORECASE)


def rewrite_arxiv_abs_to_pdf(url):
    """Rewrite https://arxiv.org/abs/<id> -> https://arxiv.org/pdf/<id>.pdf.

    Returns the original URL if it doesn't match the arXiv abs pattern.
    """
    m = ARXIV_ABS_RE.match(url)
    if not m:
        return url
    arxiv_id = m.group("id").rstrip("/")
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def ensure_download_dir():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    print(f"Downloading PDFs into: {DOWNLOAD_DIR}")


def fetch(url):
    try:
        print(f"Fetching: {url}")
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return resp
    except Exception as e:
        print(f"Failed to fetch {url}: {e}", file=sys.stderr)
        return None


def is_same_domain(url, base=BASE_URL):
    try:
        u = urlparse(url)
        b = urlparse(base)
        return (u.netloc == "" or u.netloc == b.netloc)
    except Exception:
        return False


def find_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    page_links = set()
    pdf_links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()

        # Normalize to absolute
        full_url = urljoin(base_url, href)

        # Skip obvious mailto etc.
        if full_url.startswith("mailto:"):
            continue

        # Rewrite arXiv abstract pages to their direct PDF URLs
        full_url = rewrite_arxiv_abs_to_pdf(full_url)

        if PDF_EXT_RE.search(full_url):
            pdf_links.add(full_url)
        elif is_same_domain(full_url):
            page_links.add(full_url)

    return page_links, pdf_links


def download_pdf(url):
    filename = os.path.basename(urlparse(url).path) or "paper.pdf"
    # Avoid empty or duplicate names
    local_path = os.path.join(DOWNLOAD_DIR, filename)

    # If file already exists, skip
    if os.path.exists(local_path):
        print(f"Already exists, skipping: {local_path}")
        return

    try:
        print(f"Downloading PDF: {url}")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print(f"Saved: {local_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}", file=sys.stderr)


def crawl_for_pdfs(start_urls):
    ensure_download_dir()

    visited_pages = set()
    pdfs_found = set()
    to_visit = list(start_urls)

    while to_visit:
        url = to_visit.pop()
        if url in visited_pages:
            continue
        visited_pages.add(url)

        resp = fetch(url)
        if resp is None:
            continue

        page_links, pdf_links = find_links(resp.text, url)

        # Add new page links to frontier
        for link in page_links:
            if link not in visited_pages and link.startswith(BASE_URL):
                to_visit.append(link)

        # Collect PDFs
        for pdf in pdf_links:
            if pdf not in pdfs_found:
                pdfs_found.add(pdf)

    print("\nFound PDFs:")
    for pdf in pdfs_found:
        print(" -", pdf)

    print("\nDownloading all PDFs...")
    for pdf in pdfs_found:
        download_pdf(pdf)

    print("\nDone.")


if __name__ == "__main__":
    crawl_for_pdfs(START_URLS)
