# bot/news_module.py
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


@dataclass
class NewsItem:
    source: str
    title: str
    link: str
    published_ts: float
    matched_assets: List[str]
    matched_keywords: List[str]
    score: float


def _now_ts() -> float:
    return float(time.time())


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()


def _norm(s: str) -> str:
    return (s or "").strip()


def _lower(s: str) -> str:
    return _norm(s).lower()


def _parse_time_to_ts(s: str) -> Optional[float]:
    s = _norm(s)
    if not s:
        return None

    # RFC822 (common in RSS <pubDate>)
    try:
        dt = parsedate_to_datetime(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        pass

    # ISO8601 (common in Atom <updated>/<published>)
    # Normalize Z to +00:00
    try:
        ss = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ss)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _fetch_url(url: str, timeout_sec: int = 12) -> str:
    req = Request(
        url=url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; KrakenBotNews/1.0; +https://example.local)",
            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
        },
        method="GET",
    )
    with urlopen(req, timeout=timeout_sec) as resp:
        data = resp.read()
    # RSS is usually UTF-8; if not, ignore bad bytes safely
    return data.decode("utf-8", errors="ignore")


def _text_from_elem(elem: Optional[ET.Element]) -> str:
    if elem is None:
        return ""
    # concatenate text + child tail text
    parts: List[str] = []
    if elem.text:
        parts.append(elem.text)
    for ch in list(elem):
        if ch.text:
            parts.append(ch.text)
        if ch.tail:
            parts.append(ch.tail)
    return _norm(" ".join(parts))


def _first_nonempty(*vals: str) -> str:
    for v in vals:
        vv = _norm(v)
        if vv:
            return vv
    return ""


def _strip_html(s: str) -> str:
    # very lightweight HTML stripping for RSS descriptions
    s = _norm(s)
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _split_words(s: str) -> List[str]:
    s = _lower(s)
    # keep words + tickers; treat separators as boundaries
    return re.findall(r"[a-z0-9\-\+\.]+", s)


class NewsModule:
    """
    RSS/Atom news fetcher with:
      - hourly refresh (refresh_sec)
      - dedupe via persisted seen-set (state_path)
      - relevance filter: portfolio assets + high impact keywords
      - prints only NEW items each refresh
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get("enabled", False))
        self.refresh_sec = int(self.cfg.get("refresh_sec", 3600))
        self.max_new_items = int(self.cfg.get("max_new_items", 10))
        self.max_state_items = int(self.cfg.get("max_state_items", 4000))
        self.print_when_no_new = bool(self.cfg.get("print_when_no_new", False))
        self.include_general_macro_news = bool(self.cfg.get("include_general_macro_news", True))

        self.state_path = Path(self.cfg.get("state_path", "data/news_state.json"))
        self.sources = self.cfg.get("sources") or []
        self.high_impact_keywords = [str(x) for x in (self.cfg.get("high_impact_keywords") or [])]

        self._last_fetch_ts: float = 0.0
        self._seen: Dict[str, float] = {}  # id -> first_seen_ts

        self._load_state()

    def _load_state(self) -> None:
        try:
            if not self.state_path.exists():
                return
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._last_fetch_ts = _safe_float(data.get("last_fetch_ts"), 0.0)
            self._seen = {str(k): _safe_float(v, _now_ts()) for k, v in (data.get("seen") or {}).items()}
        except Exception:
            self._last_fetch_ts = 0.0
            self._seen = {}

    def _save_state(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            # trim seen to keep file bounded
            if len(self._seen) > self.max_state_items:
                items = sorted(self._seen.items(), key=lambda kv: kv[1], reverse=True)[: self.max_state_items]
                self._seen = dict(items)

            data = {
                "last_fetch_ts": self._last_fetch_ts,
                "seen": self._seen,
            }
            self.state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _parse_feed(self, xml_text: str, source_name: str) -> List[Tuple[str, str, float, str]]:
        """
        Returns list of (title, link, published_ts, raw_text_blob)
        Works for RSS 2.0 and Atom.
        """
        out: List[Tuple[str, str, float, str]] = []
        if not xml_text:
            return out

        # Some feeds include leading junk; try to find the first '<'
        idx = xml_text.find("<")
        if idx > 0:
            xml_text = xml_text[idx:]

        try:
            root = ET.fromstring(xml_text)
        except Exception:
            return out

        tag = _lower(root.tag)

        # Atom: <feed><entry>...
        if "feed" in tag and "atom" in tag or tag.endswith("feed"):
            for entry in root.findall(".//{*}entry"):
                title = _text_from_elem(entry.find("{*}title"))
                link = ""
                for l in entry.findall("{*}link"):
                    href = l.attrib.get("href")
                    rel = _lower(l.attrib.get("rel", ""))
                    if href and (rel in ("", "alternate")):
                        link = href
                        break
                updated = _first_nonempty(
                    _text_from_elem(entry.find("{*}published")),
                    _text_from_elem(entry.find("{*}updated")),
                )
                ts = _parse_time_to_ts(updated) or _now_ts()
                summary = _strip_html(_text_from_elem(entry.find("{*}summary")))
                content = _strip_html(_text_from_elem(entry.find("{*}content")))
                blob = f"{title}\n{summary}\n{content}"
                if title and link:
                    out.append((title, link, ts, blob))
            return out

        # RSS: <rss><channel><item>...
        for item in root.findall(".//item"):
            title = _text_from_elem(item.find("title"))
            link = _text_from_elem(item.find("link"))
            guid = _text_from_elem(item.find("guid"))
            pub = _text_from_elem(item.find("pubDate"))
            ts = _parse_time_to_ts(pub) or _now_ts()
            desc = _strip_html(_text_from_elem(item.find("description")))
            blob = f"{title}\n{desc}"
            link = _first_nonempty(link, guid)
            if title and link:
                out.append((title, link, ts, blob))
        return out

    def _asset_aliases(self, assets: List[str]) -> Dict[str, List[str]]:
        # basic aliases; extend if you want
        aliases: Dict[str, List[str]] = {}
        for a in assets:
            aa = a.strip().upper()
            if not aa:
                continue
            if aa == "BTC":
                aliases[aa] = ["btc", "bitcoin", "satoshi", "₿"]
            elif aa == "ETH":
                aliases[aa] = ["eth", "ethereum", "ether", "vitalik"]
            elif aa == "SOL":
                aliases[aa] = ["sol", "solana"]
            elif aa == "XMR":
                aliases[aa] = ["xmr", "monero"]
            elif aa == "XRP":
                aliases[aa] = ["xrp", "ripple"]
            elif aa == "ATOM":
                aliases[aa] = ["atom", "cosmos"]
            elif aa == "ADA":
                aliases[aa] = ["ada", "cardano"]
            elif aa == "AVAX":
                aliases[aa] = ["avax", "avalanche"]
            elif aa == "DOT":
                aliases[aa] = ["dot", "polkadot"]
            elif aa == "MATIC":
                aliases[aa] = ["matic", "polygon"]
            elif aa == "LINK":
                aliases[aa] = ["link", "chainlink"]
            elif aa == "UNI":
                aliases[aa] = ["uni", "uniswap"]
            elif aa == "DOGE":
                aliases[aa] = ["doge", "dogecoin", "shiba"]
            elif aa == "LTC":
                aliases[aa] = ["ltc", "litecoin"]
            elif aa == "BCH":
                aliases[aa] = ["bch", "bitcoin cash"]
            elif aa == "ALGO":
                aliases[aa] = ["algo", "algorand"]
            elif aa == "NEAR":
                aliases[aa] = ["near", "near protocol"]
            elif aa == "FTM":
                aliases[aa] = ["ftm", "fantom"]
            elif aa == "ICP":
                aliases[aa] = ["icp", "internet computer"]
            elif aa == "APT":
                aliases[aa] = ["apt", "aptos"]
            elif aa == "ARB":
                aliases[aa] = ["arb", "arbitrum"]
            elif aa == "OP":
                aliases[aa] = ["op", "optimism"]
            elif aa == "TRX":
                aliases[aa] = ["trx", "tron"]
            elif aa == "TON":
                aliases[aa] = ["ton", "toncoin", "telegram"]
            else:
                aliases[aa] = [aa.lower()]
        return aliases

    def _match(self, text_blob: str, assets: List[str]) -> Tuple[List[str], List[str], float]:
        """
        Returns (matched_assets, matched_keywords, score)
        """
        blob = _lower(text_blob)
        words = set(_split_words(blob))

        aliases = self._asset_aliases(assets)
        matched_assets: List[str] = []
        for asset, als in aliases.items():
            hit = False
            for al in als:
                # token and substring checks (title text often includes punctuation)
                if al in words or re.search(rf"\b{re.escape(al)}\b", blob):
                    hit = True
                    break
            if hit:
                matched_assets.append(asset)

        matched_keywords: List[str] = []
        for kw in self.high_impact_keywords:
            k = _lower(kw)
            if not k:
                continue
            if k in words or re.search(rf"\b{re.escape(k)}\b", blob):
                matched_keywords.append(kw)

        # score (simple + robust):
        # - assets are more important than generic keywords
        # - title relevance tends to be higher (blob includes title first)
        score = 0.0
        score += 2.5 * float(len(matched_assets))
        score += 1.0 * float(len(matched_keywords))
        # small boost for really high-impact items
        if any(k.lower() in ("hack", "exploit", "sec", "etf", "liquidation", "outage") for k in matched_keywords):
            score += 1.0

        return matched_assets, matched_keywords, float(score)

    def poll_and_get_new(self, portfolio_assets: List[str]) -> List[NewsItem]:
        """
        Fetches feeds at most once per refresh_sec.
        Returns NEW NewsItem(s) (not seen before).
        """
        if not self.enabled:
            return []

        now = _now_ts()
        if self._last_fetch_ts and (now - self._last_fetch_ts) < float(self.refresh_sec):
            return []

        print(f"[NEWS] Fetching news for {len(portfolio_assets)} tracked assets...")
        self._last_fetch_ts = now

        assets = [a.strip().upper() for a in (portfolio_assets or []) if a and a.strip()]
        assets = sorted(set(assets))
        print(f"[NEWS] Tracking assets: {', '.join(assets[:10])}{'...' if len(assets) > 10 else ''}")

        new_items: List[NewsItem] = []

        for src in self.sources:
            name = str((src or {}).get("name") or "Source").strip()
            url = str((src or {}).get("url") or "").strip()
            if not url:
                continue

            try:
                xml = _fetch_url(url)
                rows = self._parse_feed(xml, name)
                for title, link, ts, blob in rows:
                    item_id = _sha1(link)
                    if item_id in self._seen:
                        continue

                    matched_assets, matched_kws, score = self._match(blob, assets)

                    # relevance filter:
                    # - include coin-matching items
                    # - optionally include macro/high-impact even if no coin match
                    if matched_assets:
                        pass
                    else:
                        if not self.include_general_macro_news:
                            continue
                        if not matched_kws:
                            continue
                        # if only macro, require a minimum score
                        if score < 1.5:
                            continue

                    # mark seen immediately to avoid duplicates across feeds
                    self._seen[item_id] = now

                    new_items.append(
                        NewsItem(
                            source=name,
                            title=_norm(title),
                            link=_norm(link),
                            published_ts=float(ts),
                            matched_assets=matched_assets,
                            matched_keywords=matched_kws,
                            score=float(score),
                        )
                    )
            except Exception:
                continue

        # sort: newest first, then score
        new_items.sort(key=lambda x: (x.published_ts, x.score), reverse=True)

        # cap output
        if self.max_new_items > 0:
            new_items = new_items[: self.max_new_items]

        self._save_state()
        return new_items

    def print_new_items(self, items: List[NewsItem]) -> None:
        now_local = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not items:
            if self.print_when_no_new:
                print(f"[NEWS] {now_local} | no new items (refresh={self.refresh_sec}s)")
            return

        print(f"\n[NEWS] {now_local} | NEW items={len(items)} (refresh={self.refresh_sec}s)")
        # quick “trending” summary: keywords/coins mentions
        kw_counts: Dict[str, int] = {}
        a_counts: Dict[str, int] = {}
        for it in items:
            for k in it.matched_keywords:
                kk = k.upper()
                kw_counts[kk] = kw_counts.get(kk, 0) + 1
            for a in it.matched_assets:
                aa = a.upper()
                a_counts[aa] = a_counts.get(aa, 0) + 1

        if kw_counts or a_counts:
            top_k = ", ".join([f"{k}({v})" for k, v in sorted(kw_counts.items(), key=lambda kv: kv[1], reverse=True)[:6]])
            top_a = ", ".join([f"{k}({v})" for k, v in sorted(a_counts.items(), key=lambda kv: kv[1], reverse=True)[:6]])
            line = " | ".join([x for x in [top_a and f"assets: {top_a}", top_k and f"themes: {top_k}"] if x])
            if line:
                print(f"[NEWS][TREND] {line}")

        for it in items:
            ts = datetime.fromtimestamp(it.published_ts).strftime("%H:%M")
            assets = ",".join(it.matched_assets) if it.matched_assets else "-"
            themes = ",".join([k.upper() for k in it.matched_keywords[:5]]) if it.matched_keywords else "-"
            print(f"  - ({it.source}, {ts}) {it.title} | assets={assets} | themes={themes}")
            print(f"    {it.link}")
        print("")
