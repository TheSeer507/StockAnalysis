#!/usr/bin/env python3
"""
Test script for the Sentiment Analyzer

This script demonstrates how the sentiment analyzer works with sample news items.
Run this to understand sentiment scoring before using it in live trading.
"""

from bot.sentiment_analyzer import SentimentAnalyzer, SentimentScore
from bot.news_module import NewsItem
import time


def create_sample_news() -> list:
    """Create sample news items for testing."""
    current_ts = time.time()
    
    return [
        NewsItem(
            source="TestSource",
            title="Bitcoin surges to new all-time high as institutional adoption grows",
            link="https://example.com/1",
            published_ts=current_ts - 3600,  # 1 hour ago
            matched_assets=["BTC"],
            matched_keywords=["adoption"],
            score=5.0
        ),
        NewsItem(
            source="TestSource",
            title="Major exchange reports security breach affecting Ethereum wallets",
            link="https://example.com/2",
            published_ts=current_ts - 7200,  # 2 hours ago
            matched_assets=["ETH"],
            matched_keywords=["hack", "breach"],
            score=4.0
        ),
        NewsItem(
            source="TestSource",
            title="Solana network upgrade expected to boost performance and scalability",
            link="https://example.com/3",
            published_ts=current_ts - 10800,  # 3 hours ago
            matched_assets=["SOL"],
            matched_keywords=["upgrade"],
            score=3.5
        ),
        NewsItem(
            source="TestSource",
            title="Bitcoin ETF sees record inflows amid bullish market sentiment",
            link="https://example.com/4",
            published_ts=current_ts - 14400,  # 4 hours ago
            matched_assets=["BTC"],
            matched_keywords=["etf", "bullish"],
            score=4.5
        ),
        NewsItem(
            source="TestSource",
            title="Crypto market crash as SEC announces new regulatory crackdown",
            link="https://example.com/5",
            published_ts=current_ts - 18000,  # 5 hours ago
            matched_assets=["BTC", "ETH", "SOL"],
            matched_keywords=["sec", "regulatory", "crash"],
            score=5.0
        ),
        NewsItem(
            source="TestSource",
            title="Ripple wins lawsuit settlement with SEC in major victory",
            link="https://example.com/6",
            published_ts=current_ts - 21600,  # 6 hours ago
            matched_assets=["XRP"],
            matched_keywords=["sec", "settlement"],
            score=4.0
        ),
        NewsItem(
            source="TestSource",
            title="Monero faces regulatory pressure amid privacy concerns",
            link="https://example.com/7",
            published_ts=current_ts - 25200,  # 7 hours ago
            matched_assets=["XMR"],
            matched_keywords=["regulatory"],
            score=3.0
        ),
        NewsItem(
            source="TestSource",
            title="Cardano announces major partnership with tech giant",
            link="https://example.com/8",
            published_ts=current_ts - 28800,  # 8 hours ago
            matched_assets=["ADA"],
            matched_keywords=["partnership"],
            score=3.5
        ),
    ]


def print_sentiment_analysis(analyzer: SentimentAnalyzer, asset: str, news_items: list):
    """Print detailed sentiment analysis for an asset."""
    sentiment = analyzer.analyze_asset_sentiment(asset, news_items, lookback_hours=24.0)
    label = analyzer.get_sentiment_label(sentiment.score)
    emoji = analyzer.get_sentiment_emoji(sentiment.score)
    
    print(f"\n{'='*70}")
    print(f"Asset: {sentiment.asset}")
    print(f"{'='*70}")
    print(f"Sentiment Score:  {sentiment.score:+.3f} / 1.000")
    print(f"Sentiment Label:  {label} {emoji}")
    print(f"News Articles:    {sentiment.news_count}")
    print(f"  - Positive:     {sentiment.positive_count}")
    print(f"  - Neutral:      {sentiment.neutral_count}")
    print(f"  - Negative:     {sentiment.negative_count}")
    
    if sentiment.recent_headlines:
        print(f"\nRecent Headlines:")
        for i, headline in enumerate(sentiment.recent_headlines, 1):
            print(f"  {i}. {headline}")
    
    print(f"{'='*70}\n")


def main():
    """Run sentiment analyzer test."""
    print("\n" + "="*70)
    print("SENTIMENT ANALYZER TEST")
    print("="*70)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    print(f"\n✓ Sentiment Analyzer initialized")
    print(f"  - Positive keywords: {len(analyzer.POSITIVE_KEYWORDS)}")
    print(f"  - Negative keywords: {len(analyzer.NEGATIVE_KEYWORDS)}")
    print(f"  - High-impact keywords: {len(analyzer.HIGH_IMPACT_KEYWORDS)}")
    
    # Create sample news
    news_items = create_sample_news()
    print(f"\n✓ Created {len(news_items)} sample news items")
    
    # Analyze sentiment for different assets
    assets_to_analyze = ["BTC", "ETH", "SOL", "XRP", "XMR", "ADA"]
    
    print("\n" + "="*70)
    print("SENTIMENT ANALYSIS RESULTS")
    print("="*70)
    
    for asset in assets_to_analyze:
        print_sentiment_analysis(analyzer, asset, news_items)
    
    # Summary
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("""
Sentiment Score Range:
  +0.50 to +1.00  🚀 VERY_BULLISH   - Strong positive news, good time to buy
  +0.20 to +0.50  📈 BULLISH        - Moderate positive news
  -0.20 to +0.20  ➡️  NEUTRAL        - Mixed or minimal news coverage
  -0.50 to -0.20  📉 BEARISH        - Moderate negative news, caution advised
  -1.00 to -0.50  ⚠️  VERY_BEARISH   - Strong negative news, avoid buying

Configuration Examples:
  - Conservative:  min_score_for_buy: 0.0   (only buy neutral+)
  - Balanced:      min_score_for_buy: -0.3  (avoid strong negative)
  - Aggressive:    min_score_for_buy: 0.3   (only buy strong positive)
    """)
    
    print("="*70)
    print("Test completed! Now you understand how sentiment scoring works.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
