# bot/sentiment_analyzer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import re


@dataclass
class SentimentScore:
    """Sentiment score for an asset based on news analysis."""
    asset: str
    score: float  # -1.0 (very bearish) to +1.0 (very bullish)
    news_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    recent_headlines: List[str]


class SentimentAnalyzer:
    """
    Analyzes sentiment from news items to provide a sentiment score for assets.
    Uses keyword-based approach with positive/negative/high-impact word lists.
    """

    # Positive sentiment keywords
    POSITIVE_KEYWORDS = [
        "bullish", "rally", "surge", "soar", "gain", "growth", "profit",
        "upgrade", "breakthrough", "adoption", "partnership", "acquisition",
        "positive", "optimistic", "strong", "robust", "milestone", "success",
        "boost", "jump", "climb", "rise", "expand", "innovation", "approve",
        "approval", "recovery", "breakout", "momentum", "buy", "accumulate",
        "moon", "pump", "green", "ath", "record high", "all-time high",
        "support", "bullish trend", "golden cross", "oversold bounce"
    ]

    # Negative sentiment keywords
    NEGATIVE_KEYWORDS = [
        "bearish", "crash", "plunge", "fall", "drop", "decline", "loss",
        "downgrade", "hack", "exploit", "scam", "fraud", "lawsuit", "ban",
        "negative", "pessimistic", "weak", "vulnerable", "risk", "warning",
        "concern", "fear", "sell", "liquidation", "dump", "red", "collapse",
        "outage", "delay", "postpone", "reject", "cancel", "investigation",
        "regulatory", "resistance", "death cross", "overbought", "correction",
        "panic", "capitulation", "fud", "sell-off", "breach", "uncertainty"
    ]

    # High-impact events (neutral but significant)
    HIGH_IMPACT_KEYWORDS = [
        "sec", "etf", "fed", "regulation", "interest rate", "cpi", "inflation",
        "merger", "ipo", "earnings", "halving", "fork", "mainnet", "launch",
        "announcement", "conference", "summit", "vote", "election", "policy"
    ]

    def __init__(self):
        self._positive_pattern = self._compile_pattern(self.POSITIVE_KEYWORDS)
        self._negative_pattern = self._compile_pattern(self.NEGATIVE_KEYWORDS)
        self._high_impact_pattern = self._compile_pattern(self.HIGH_IMPACT_KEYWORDS)

    def _compile_pattern(self, keywords: List[str]) -> re.Pattern:
        """Compile a regex pattern from keywords for efficient matching."""
        # Sort by length (longest first) to match phrases before individual words
        sorted_kw = sorted(keywords, key=len, reverse=True)
        escaped = [re.escape(kw) for kw in sorted_kw]
        pattern = r'\b(?:' + '|'.join(escaped) + r')\b'
        return re.compile(pattern, re.IGNORECASE)

    def _analyze_text(self, text: str) -> tuple[int, int, int]:
        """
        Analyze a piece of text and return (positive_count, negative_count, high_impact_count).
        """
        text_lower = text.lower()
        
        positive_matches = len(self._positive_pattern.findall(text_lower))
        negative_matches = len(self._negative_pattern.findall(text_lower))
        high_impact_matches = len(self._high_impact_pattern.findall(text_lower))
        
        return positive_matches, negative_matches, high_impact_matches

    def analyze_asset_sentiment(
        self,
        asset: str,
        news_items: List,
        lookback_hours: float = 24.0
    ) -> SentimentScore:
        """
        Analyze sentiment for a specific asset based on news items.
        
        Args:
            asset: The asset symbol (e.g., "BTC", "ETH")
            news_items: List of NewsItem objects from news_module
            lookback_hours: Only consider news from the last N hours
            
        Returns:
            SentimentScore object with aggregated sentiment
        """
        import time
        
        asset_upper = asset.upper().strip()
        now = time.time()
        cutoff_ts = now - (lookback_hours * 3600)
        
        relevant_items = []
        for item in news_items:
            # Check if this news item mentions the asset
            if asset_upper not in [a.upper() for a in item.matched_assets]:
                continue
            
            # Check if within lookback period
            if item.published_ts < cutoff_ts:
                continue
                
            relevant_items.append(item)
        
        if not relevant_items:
            return SentimentScore(
                asset=asset_upper,
                score=0.0,
                news_count=0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                recent_headlines=[]
            )
        
        total_positive = 0
        total_negative = 0
        total_high_impact = 0
        headlines = []
        
        for item in relevant_items[:10]:  # Limit to 10 most recent
            text = f"{item.title}"
            pos, neg, high = self._analyze_text(text)
            
            total_positive += pos
            total_negative += neg
            total_high_impact += high
            
            if len(headlines) < 5:
                headlines.append(item.title[:80])
        
        # Calculate overall sentiment score
        # Range: -1.0 (very bearish) to +1.0 (very bullish)
        if total_positive + total_negative == 0:
            score = 0.0
        else:
            # Simple ratio with boost for high-impact news
            net_sentiment = total_positive - total_negative
            total_keywords = total_positive + total_negative + total_high_impact
            
            # Normalize to -1 to +1 range
            if total_keywords > 0:
                score = net_sentiment / max(total_keywords, 1)
                # Apply sigmoid-like smoothing to avoid extreme values
                score = score * 0.8  # Scale down a bit
            else:
                score = 0.0
        
        # Clamp to valid range
        score = max(-1.0, min(1.0, score))
        
        # Categorize news items
        positive_count = sum(1 for item in relevant_items 
                           if self._analyze_text(item.title)[0] > self._analyze_text(item.title)[1])
        negative_count = sum(1 for item in relevant_items 
                           if self._analyze_text(item.title)[1] > self._analyze_text(item.title)[0])
        neutral_count = len(relevant_items) - positive_count - negative_count
        
        return SentimentScore(
            asset=asset_upper,
            score=float(score),
            news_count=len(relevant_items),
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            recent_headlines=headlines
        )

    def get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to a human-readable label."""
        if score >= 0.5:
            return "VERY_BULLISH"
        elif score >= 0.2:
            return "BULLISH"
        elif score >= -0.2:
            return "NEUTRAL"
        elif score >= -0.5:
            return "BEARISH"
        else:
            return "VERY_BEARISH"

    def get_sentiment_emoji(self, score: float) -> str:
        """Get an emoji representation of sentiment."""
        if score >= 0.5:
            return "🚀"
        elif score >= 0.2:
            return "📈"
        elif score >= -0.2:
            return "➡️"
        elif score >= -0.5:
            return "📉"
        else:
            return "⚠️"
