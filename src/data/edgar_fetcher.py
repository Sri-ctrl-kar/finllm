"""
edgar_fetcher.py
------------------------------------------------------------
Fetches real 10-K/10-Q filing text from SEC EDGAR's public API.

NOTE: this needs outbound internet access, which this sandbox
doesn't have — so this file is written correctly but not executed
here. Run it on your own machine (or in Antigravity) where the
network is open. tests/test_pipeline.py exercises the rest of the
pipeline against a bundled sample filing instead, so you can verify
everything downstream of fetching works today.

SEC EDGAR basics (worth understanding, not just calling):
  - Every filer (company) has a CIK (Central Index Key) number.
  - `https://data.sec.gov/submissions/CIK{cik:010d}.json` lists
    all of a company's filings with accession numbers.
  - The actual filing document is fetched from
    `https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filename}`
  - SEC requires a descriptive User-Agent header identifying who's
    making the request (with contact info) — requests without one
    get blocked. This is a real, easy-to-miss gotcha.
------------------------------------------------------------
"""

import re
import time
from typing import Optional

import requests

SEC_BASE = "https://www.sec.gov"
SEC_DATA_BASE = "https://data.sec.gov"

# SEC asks that automated requests identify a real contact — replace
# with your own name/email before running at any real volume.
USER_AGENT = "RideSync-Capstone-Project research@example.com"


class EdgarFetcher:
    def __init__(self, user_agent: str = USER_AGENT, request_delay_s: float = 0.2):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.request_delay_s = request_delay_s  # SEC rate-limits; be polite

    def get_cik(self, ticker: str) -> Optional[str]:
        """Look up a company's CIK from its ticker symbol."""
        resp = self.session.get(f"{SEC_BASE}/files/company_tickers.json")
        resp.raise_for_status()
        data = resp.json()
        ticker = ticker.upper()
        for entry in data.values():
            if entry["ticker"] == ticker:
                return f"{entry['cik_str']:010d}"
        return None

    def list_filings(self, cik: str, form_type: str = "10-K", limit: int = 5) -> list[dict]:
        """List recent filings of a given form type for a company."""
        resp = self.session.get(f"{SEC_DATA_BASE}/submissions/CIK{cik}.json")
        resp.raise_for_status()
        data = resp.json()

        recent = data["filings"]["recent"]
        results = []
        for i, form in enumerate(recent["form"]):
            if form == form_type:
                results.append(
                    {
                        "accessionNumber": recent["accessionNumber"][i],
                        "filingDate": recent["filingDate"][i],
                        "primaryDocument": recent["primaryDocument"][i],
                        "form": form,
                    }
                )
            if len(results) >= limit:
                break
        return results

    def fetch_filing_text(self, cik: str, accession_number: str, primary_document: str) -> str:
        """Download a filing document and strip HTML tags down to plain text."""
        accession_no_dashes = accession_number.replace("-", "")
        url = f"{SEC_BASE}/Archives/edgar/data/{int(cik)}/{accession_no_dashes}/{primary_document}"

        time.sleep(self.request_delay_s)
        resp = self.session.get(url)
        resp.raise_for_status()

        return self._strip_html(resp.text)

    @staticmethod
    def _strip_html(html: str) -> str:
        """Minimal HTML-to-text: good enough for a Phase 1 baseline.
        Upgrade path: use `beautifulsoup4` for more robust extraction
        (handles tables, nested tags, etc. more gracefully)."""
        text = re.sub(r"<script.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def fetch_latest_10k(self, ticker: str) -> dict:
        """Convenience: ticker in, {company, filing_type, source_id, text} out —
        ready to hand straight to chunker.chunk_text()."""
        cik = self.get_cik(ticker)
        if not cik:
            raise ValueError(f"Ticker '{ticker}' not found in SEC's ticker list.")

        filings = self.list_filings(cik, form_type="10-K", limit=1)
        if not filings:
            raise ValueError(f"No 10-K filings found for '{ticker}'.")

        filing = filings[0]
        text = self.fetch_filing_text(cik, filing["accessionNumber"], filing["primaryDocument"])

        return {
            "company": ticker.upper(),
            "filing_type": "10-K",
            "source_id": filing["accessionNumber"],
            "filing_date": filing["filingDate"],
            "text": text,
        }


if __name__ == "__main__":
    # Example usage — run this file directly (with internet access) to try it:
    #   python -m src.data.edgar_fetcher
    fetcher = EdgarFetcher()
    result = fetcher.fetch_latest_10k("AAPL")
    print(f"Fetched {result['company']} {result['filing_type']} dated {result['filing_date']}")
    print(f"Document length: {len(result['text'])} characters")
    print(result["text"][:500])
