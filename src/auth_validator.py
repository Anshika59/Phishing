import re
from email.utils import parseaddr

PROTECTED_BRANDS = {
    "paypal", "microsoft", "apple", "amazon", "google", "netflix",
    "bankofamerica", "chase", "wellsfargo", "citibank", "dhl", "fedex"
}


def extract_domain(email_addr):
    """
    Extracts lowercase domain from an email address string.
    """
    _, addr = parseaddr(str(email_addr or ""))
    if "@" in addr:
        return addr.split("@")[-1].strip().lower()
    return ""


def parse_raw_headers(raw_email):
    """
    Extracts key authentication and routing headers from a raw email string.
    """
    headers = {
        "from": "",
        "reply-to": "",
        "return-path": "",
        "authentication-results": "",
        "received-spf": "",
        "dkim-signature": "",
    }
    if not raw_email:
        return headers

    # Unfold multi-line RFC 5322 headers
    unfolded = re.sub(r"\r?\n[ \t]+", " ", str(raw_email))
    for line in unfolded.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        k = key.strip().lower()
        if k in headers:
            headers[k] = (headers[k] + " " + val.strip()).strip()
    return headers


def evaluate_authentication(email_input):
    """
    Evaluates SPF, DKIM, DMARC, and header alignment rules.
    Accepts either a raw email string or a dictionary of header/metadata fields.
    Returns auth_risk_score [0.0, 1.0], hard_auth_fail, status breakdown, and flags.
    """
    if isinstance(email_input, dict):
        headers = {k.lower(): str(v) for k, v in email_input.items()}
    else:
        headers = parse_raw_headers(email_input)

    auth_res = headers.get("authentication-results", "").lower()
    rec_spf = headers.get("received-spf", "").lower()
    dkim_sig = headers.get("dkim-signature", "").lower()

    spf_status = headers.get("spf", "").lower() or "unknown"
    dkim_status = headers.get("dkim", "").lower() or "unknown"
    dmarc_status = headers.get("dmarc", "").lower() or "unknown"

    # Extract statuses from Authentication-Results / Received-SPF headers if not explicitly passed
    if spf_status == "unknown":
        m_spf = re.search(r"spf=(pass|fail|softfail|neutral|none|temperror|permerror)", auth_res)
        if m_spf:
            spf_status = m_spf.group(1)
        elif rec_spf:
            m_rec = re.match(r"^(pass|fail|softfail|neutral|none|temperror|permerror)", rec_spf)
            if m_rec:
                spf_status = m_rec.group(1)

    if dkim_status == "unknown":
        m_dkim = re.search(r"dkim=(pass|fail|neutral|none|temperror|permerror)", auth_res)
        if m_dkim:
            dkim_status = m_dkim.group(1)
        elif dkim_sig:
            dkim_status = "present"

    if dmarc_status == "unknown":
        m_dmarc = re.search(r"dmarc=(pass|fail|bestguesspass|none|temperror|permerror)", auth_res)
        if m_dmarc:
            dmarc_status = m_dmarc.group(1)

    risk = 0.0
    hard_fail = False
    flags = []

    # 1. SPF Rules
    if spf_status in {"fail", "permerror"}:
        risk += 0.35
        hard_fail = True
        flags.append(f"SPF authentication failed ({spf_status})")
    elif spf_status in {"softfail", "none", "temperror"}:
        risk += 0.15
        flags.append(f"Weak/missing SPF record ({spf_status})")

    # 2. DKIM Rules
    if dkim_status in {"fail", "permerror"}:
        risk += 0.35
        hard_fail = True
        flags.append(f"DKIM signature verification failed ({dkim_status})")
    elif dkim_status in {"none", "temperror"}:
        risk += 0.15
        flags.append(f"Missing or invalid DKIM signature ({dkim_status})")

    # 3. DMARC Rules
    if dmarc_status in {"fail", "permerror"}:
        risk += 0.40
        hard_fail = True
        flags.append(f"DMARC policy check failed ({dmarc_status})")
    elif dmarc_status == "none":
        risk += 0.10
        flags.append("No DMARC policy enforced (none)")

    # 4. Sender Header Alignment (From vs Reply-To / Return-Path)
    from_raw = headers.get("from", "")
    display_name, from_addr = parseaddr(from_raw)
    from_domain = extract_domain(from_addr)
    reply_domain = extract_domain(headers.get("reply-to", ""))
    return_domain = extract_domain(headers.get("return-path", ""))

    if from_domain and reply_domain and from_domain != reply_domain:
        risk += 0.25
        flags.append(f"Reply-To domain mismatch ({from_domain} vs {reply_domain})")

    if from_domain and return_domain and from_domain != return_domain:
        risk += 0.20
        flags.append(f"Return-Path domain mismatch ({from_domain} vs {return_domain})")

    # 5. Display Name Brand Spoofing Check
    disp_clean = re.sub(r"[^a-z0-9]", "", display_name.lower())
    for brand in PROTECTED_BRANDS:
        if brand in disp_clean and from_domain and brand not in from_domain:
            risk += 0.35
            hard_fail = True
            flags.append(f"Display name spoofing '{brand}' from unrelated domain {from_domain}")
            break

    return {
        "auth_risk_score": round(min(1.0, risk), 4),
        "hard_auth_fail": hard_fail,
        "spf": spf_status,
        "dkim": dkim_status,
        "dmarc": dmarc_status,
        "flags": list(dict.fromkeys(flags)),
    }


if __name__ == "__main__":
    legit_email = """From: GitHub <noreply@github.com>
Authentication-Results: mx.google.com; dkim=pass; spf=pass; dmarc=pass
Subject: New commit pushed
"""

    spoofed_email = """From: PayPal Security <alert@random-mailer.xyz>
Reply-To: support@collect-credentials.top
Return-Path: <bounce@unknown-server.info>
Authentication-Results: mx.google.com; dkim=fail; spf=fail; dmarc=fail
Subject: Urgent Account Verification
"""

    print("\n=== EMAIL AUTHENTICATION VALIDATOR TEST ===")
    for label, sample in [("Legitimate Email", legit_email), ("Spoofed Email", spoofed_email)]:
        res = evaluate_authentication(sample)
        print(f"\n{label}:")
        print(f"  SPF / DKIM / DMARC : {res['spf']} / {res['dkim']} / {res['dmarc']}")
        print(f"  Hard Auth Fail     : {res['hard_auth_fail']}")
        print(f"  Auth Risk Score    : {res['auth_risk_score']}")
        print(f"  Flags              : {res['flags']}")