import re

def extract_header_features(raw_email):
    """
    Module written by Arushi: Extracts SPF, DKIM, DMARC status,
    detects Reply-To mismatches, and checks typosquatted domains.
    """
    email_text = str(raw_email).lower()
    
    # 1. Extract SPF, DKIM, DMARC status from headers
    spf_pass = 1 if ("spf=pass" in email_text or "received-spf: pass" in email_text) else 0
    dkim_pass = 1 if ("dkim=pass" in email_text or "header.i=" in email_text) else 0
    dmarc_pass = 1 if ("dmarc=pass" in email_text) else 0
    
    # 2. Detect Reply-To vs From domain mismatches
    reply_to_mismatch = 0
    from_match = re.search(r'from:\s*.*?<([^>]+)>', email_text)
    reply_match = re.search(r'reply-to:\s*.*?<([^>]+)>', email_text)
    
    if from_match and reply_match:
        from_domain = from_match.group(1).split('@')[-1]
        reply_domain = reply_match.group(1).split('@')[-1]
        if from_domain != reply_domain:
            reply_to_mismatch = 1

    # 3. Detect typosquatted / look-alike domains
    typosquat_flag = 0
    suspicious_patterns = ["paypa1", "g00gle", "micros0ft", "verify-bank", "secure-update", "0utlook"]
    if any(pattern in email_text for pattern in suspicious_patterns):
        typosquat_flag = 1
        
    return {
        "spf_pass": spf_pass,
        "dkim_pass": dkim_pass,
        "dmarc_pass": dmarc_pass,
        "reply_to_mismatch": reply_to_mismatch,
        "typosquat_flag": typosquat_flag
    }