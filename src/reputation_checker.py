import os
import re
import socket
from urllib.parse import urlparse

# Default threat feed domains (auto-saved to data/blacklists/threat_feed.txt)
DEFAULT_THREAT_FEED = [
    "secure-login-verify-account.com",
    "paypal-security-update.xyz",
    "microsoft-auth-alert.top",
    "amazon-billing-confirm.tk",
    "bankofamerica-verify-user.ml",
]

SUSPICIOUS_TLDS = {
    ".xyz", ".top", ".tk", ".ml", ".ga", ".cf", ".gq",
    ".zip", ".click", ".work", ".support", ".info", ".icu"
}

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "rb.gy",
    "is.gd", "ow.ly", "buff.ly", "cutt.ly"
}

PHISHING_URL_KEYWORDS = {
    "login", "signin", "verify", "secure", "update", "account",
    "banking", "confirm", "wallet", "suspend", "password", "credential"
}


def load_local_blacklist():
    """
    Loads external/cached threat feed domains from data/blacklists/threat_feed.txt.
    Creates a starter threat feed file automatically if missing.
    """
    feed_dir = os.path.join("data", "blacklists")
    feed_path = os.path.join(feed_dir, "threat_feed.txt")

    if not os.path.exists(feed_path):
        os.makedirs(feed_dir, exist_ok=True)
        with open(feed_path, "w", encoding="utf-8") as f:
            f.write("# Cached External Threat Intelligence Feed\n")
            for domain in DEFAULT_THREAT_FEED:
                f.write(domain + "\n")

    blacklisted_domains = set()
    with open(feed_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = line.strip().lower()
            if entry and not entry.startswith("#"):
                blacklisted_domains.add(entry)
    return blacklisted_domains


def extract_urls(text):
    """
    Extracts all HTTP/HTTPS and www URLs from raw email text.
    """
    if not text:
        return []
    pattern = r"(?:https?://|www\.)[^\s<>\"'\]\)]+"
    raw_urls = re.findall(pattern, str(text), flags=re.IGNORECASE)
    cleaned = []
    for u in raw_urls:
        u = u.rstrip(".,;:!?")
        if not u.lower().startswith(("http://", "https://")):
            u = "http://" + u
        cleaned.append(u)
    return list(dict.fromkeys(cleaned))


def check_spamhaus_dnsbl(domain_or_ip, timeout=1.5):
    """
    Queries Spamhaus DBL (domains) or ZEN (IPv4) DNS blocklists.
    Ignores public resolver rate-limit codes (127.255.255.x) to prevent false positives.
    """
    target = domain_or_ip.strip().lower()
    if not target:
        return False, None

    is_ipv4 = bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", target))
    if is_ipv4:
        reversed_ip = ".".join(reversed(target.split(".")))
        query = f"{reversed_ip}.zen.spamhaus.org"
    else:
        query = f"{target}.dbl.spamhaus.org"

    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    try:
        result_ip = socket.gethostbyname(query)
        # Valid Spamhaus listing codes start with 127.0.0.x or 127.0.1.x
        if result_ip.startswith(("127.0.0.", "127.0.1.")):
            return True, f"Spamhaus DNSBL hit ({result_ip})"
        return False, None
    except (socket.gaierror, socket.timeout, OSError):
        return False, None
    finally:
        socket.setdefaulttimeout(old_timeout)


def evaluate_url_reputation(text_or_urls, enable_live_dnsbl=True):
    """
    Evaluates URLs in email content against blacklists and reputation heuristics.
    Returns a structured dictionary with reputation_score [0.0, 1.0] and triggered flags.
    """
    if isinstance(text_or_urls, list):
        urls = text_or_urls
    else:
        urls = extract_urls(text_or_urls)

    local_blacklist = load_local_blacklist()

    if not urls:
        return {
            "reputation_score": 0.0,
            "blacklisted": False,
            "urls_found": [],
            "domains_checked": [],
            "flags": []
        }

    max_risk = 0.0
    is_blacklisted = False
    domains_checked = []
    flags = []

    for url in urls:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        path_query = (parsed.path + "?" + parsed.query).lower()

        if not host:
            continue
        domains_checked.append(host)

        url_risk = 0.0

        # 1. Local / Cached Threat Feed Check
        if host in local_blacklist or any(host.endswith("." + b) for b in local_blacklist):
            is_blacklisted = True
            url_risk = 1.0
            flags.append(f"Blacklist feed match: {host}")

        # 2. External Spamhaus DNSBL Check
        if enable_live_dnsbl and not is_blacklisted:
            hit, reason = check_spamhaus_dnsbl(host)
            if hit:
                is_blacklisted = True
                url_risk = 1.0
                flags.append(f"{reason} on {host}")

        # 3. IP-Based URL Check
        if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host):
            url_risk += 0.45
            flags.append(f"Direct IP address URL: {host}")

        # 4. Suspicious / High-Abuse TLD Check
        if any(host.endswith(tld) for tld in SUSPICIOUS_TLDS):
            url_risk += 0.25
            flags.append(f"High-risk TLD on {host}")

        # 5. URL Shortener Obfuscation Check
        if host in URL_SHORTENERS:
            url_risk += 0.30
            flags.append(f"URL shortener detected: {host}")

        # 6. Punycode / Homograph or '@' Credential Trick
        if "xn--" in host or "@" in parsed.netloc:
            url_risk += 0.40
            flags.append(f"Deceptive domain/homograph encoding: {host}")

        # 7. Excessive Subdomains (>= 4 parts)
        if len(host.split(".")) >= 4:
            url_risk += 0.20
            flags.append(f"Excessive subdomains: {host}")

        # 8. Phishing Lure Keywords in URL
        matched_kw = [kw for kw in PHISHING_URL_KEYWORDS if kw in host or kw in path_query]
        if matched_kw:
            url_risk += min(0.30, 0.12 * len(matched_kw))
            flags.append(f"Suspicious URL keywords ({', '.join(matched_kw[:3])}) in {host}")

        max_risk = max(max_risk, min(1.0, url_risk))

    # Deduplicate flags while preserving order
    unique_flags = list(dict.fromkeys(flags))

    return {
        "reputation_score": round(max_risk, 4),
        "blacklisted": is_blacklisted,
        "urls_found": urls,
        "domains_checked": list(dict.fromkeys(domains_checked)),
        "flags": unique_flags
    }


if __name__ == "__main__":
    samples = [
        "Hi team, please review the agenda at https://github.com/Anshika59/Phishing before 4 PM.",
        "URGENT: Your account is suspended! Verify immediately at http://192.168.10.45/login/verify-account",
        "Security Alert: Confirm your credentials at https://paypal-security-update.xyz/signin"
    ]

    print("\n=== URL REPUTATION & BLACKLIST CHECKER TEST ===")
    for idx, text in enumerate(samples, 1):
        res = evaluate_url_reputation(text)
        print(f"\nSample {idx}:")
        print(f"  Domains Checked : {res['domains_checked']}")
        print(f"  Blacklisted     : {res['blacklisted']}")
        print(f"  Reputation Score: {res['reputation_score']}")
        print(f"  Flags           : {res['flags']}")