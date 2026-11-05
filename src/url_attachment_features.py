import re
import urllib.parse

RISKY_EXTENSIONS = {'.exe', '.vbs', '.js', '.scr', '.bat', '.cmd', '.zip', '.docm', '.xlsm', '.iso'}

def check_ip_in_url(url):
    """Detects if a URL uses a raw IP address instead of a domain name."""
    ip_pattern = r'http[s]?://(?:\d{1,3}\.){3}\d{1,3}'
    return 1 if re.search(ip_pattern, url) else 0

def check_text_url_mismatch(display_text, actual_url):
    """Detects if visible link text differs from the actual hyperlink destination."""
    if not display_text or not actual_url:
        return 0
    display_domain = urllib.parse.urlparse(display_text).netloc.lower()
    actual_domain = urllib.parse.urlparse(actual_url).netloc.lower()
    if display_domain and actual_domain and (display_domain != actual_domain):
        return 1
    return 0

def extract_url_features(text):
    """Extracts URL counts, IP usage, and redirect markers from text body."""
    urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', text)
    has_ip_url = any(check_ip_in_url(u) for u in urls)
    has_redirect = any(u.count("http://") + u.count("https://") > 1 or "//" in u[8:] for u in urls)
    
    return {
        "url_count": len(urls),
        "has_ip_in_url": int(has_ip_url),
        "has_redirect_in_url": int(has_redirect)
    }

def check_risky_attachments(attachment_filenames):
    """Identifies if any attachment has an executable or high-risk extension."""
    if isinstance(attachment_filenames, str):
        attachment_filenames = [a.strip() for a in attachment_filenames.split(',')]
    
    for filename in attachment_filenames:
        ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
        if ext in RISKY_EXTENSIONS:
            return 1
    return 0