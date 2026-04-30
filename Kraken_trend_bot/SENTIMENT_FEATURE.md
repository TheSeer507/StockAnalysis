# Sentiment Analysis Feature

## Overview

The sentiment analysis feature adds news-based sentiment scoring to your ML-powered trading bot. It analyzes news headlines and content to determine market sentiment for each asset, helping you make more informed trading decisions.

## How It Works

### 1. News Collection
The bot already collects news from multiple RSS feeds (CoinDesk, Cointelegraph, CryptoSlate, etc.). The sentiment analyzer extends this by:
- Caching recent news items (last 200 items)
- Analyzing news for positive, negative, and high-impact keywords
- Calculating sentiment scores for each asset

### 2. Sentiment Scoring
Each asset receives a sentiment score ranging from **-1.0 (very bearish)** to **+1.0 (very bullish)**:

- **🚀 VERY_BULLISH** (≥0.5): Strong positive news
- **📈 BULLISH** (0.2 to 0.5): Moderately positive news
- **➡️ NEUTRAL** (-0.2 to 0.2): Mixed or minimal news
- **📉 BEARISH** (-0.5 to -0.2): Moderately negative news
- **⚠️ VERY_BEARISH** (≤-0.5): Strong negative news

### 3. Integration Points

#### A. ML Recommendations
- Sentiment is analyzed for each potential trade recommendation
- Filters out trades with sentiment below the threshold (default: -0.3)
- Boosts recommendation scores based on positive sentiment
- Displays sentiment alongside ML predictions

Example output:
```
[ENGINE] Top 5 recommendations (quote=USD):
   1. BTC/USD      price=95234.50 24h=+2.34% trend=BULL sig=LONG -> BUY | ML TP50=96500.00 TP80=97800.00 TP90=98500.00 | SENTIMENT: BULLISH (+0.42, 5 news)
   2. ETH/USD      price=3456.78 24h=+1.89% trend=BULL sig=LONG -> BUY | ML TP50=3520.00 TP80=3580.00 TP90=3620.00 | SENTIMENT: NEUTRAL (+0.15, 3 news)
```

#### B. Portfolio Scanner
- Shows sentiment for all held assets during portfolio scans
- Helps you understand the current market perception of your holdings
- Updates every time the scanner runs

Example output:
```
[PORTFOLIO] Scanning signals for held assets...
  BTC      BTC/USD     qty=0.05000000 price=95234.50 trend=BULL signal=LONG | ML TP50=96500.00 TP80=97800.00 TP90=98500.00 | SENTIMENT: BULLISH (+0.42, 5 news)
  ETH      ETH/USD     qty=0.50000000 price=3456.78 trend=BEAR signal=EXIT | ML TP50=3420.00 TP80=3380.00 TP90=3350.00 | SENTIMENT: BEARISH (-0.35, 4 news)
```

## Configuration

Add this section to your `config/config.yaml`:

```yaml
sentiment:
  enabled: true                        # enable sentiment analysis from news
  use_for_recommendations: true        # use sentiment to filter recommendations
  min_score_for_buy: -0.3              # minimum sentiment score (-1 to +1) to allow BUY action
  score_weight: 0.1                    # weight of sentiment in recommendation scoring (0-1)
  lookback_hours: 24.0                 # analyze news from last N hours
```

### Configuration Parameters

- **enabled**: Turn sentiment analysis on/off
- **use_for_recommendations**: If true, filters out buy recommendations with poor sentiment
- **min_score_for_buy**: Minimum sentiment score required to allow BUY/COMPOUND actions (range: -1.0 to 1.0)
  - `-1.0`: Allow all trades regardless of sentiment
  - `-0.3` (default): Block trades with strong negative sentiment
  - `0.0`: Only allow trades with neutral or positive sentiment
  - `0.3`: Only allow trades with strong positive sentiment
- **score_weight**: How much sentiment affects recommendation ranking (0.0 to 1.0)
  - `0.0`: Sentiment doesn't affect ranking, only filtering
  - `0.1` (default): Small boost for positive sentiment
  - `0.5`: Strong influence on recommendation order
- **lookback_hours**: How far back to analyze news (default: 24 hours)

## Sentiment Keywords

The analyzer uses predefined keyword lists to classify news sentiment:

### Positive Keywords (Examples)
- bullish, rally, surge, gain, growth, profit
- upgrade, breakthrough, adoption, partnership
- buy, accumulate, breakout, momentum
- ath (all-time high), golden cross

### Negative Keywords (Examples)
- bearish, crash, plunge, decline, loss
- hack, exploit, scam, fraud, lawsuit
- sell, liquidation, dump, collapse
- death cross, panic, capitulation

### High-Impact Events (Neutral but Significant)
- SEC, ETF, regulation, interest rate
- merger, IPO, earnings, halving
- mainnet launch, fork

## Use Cases

### 1. Risk Management
Avoid entering positions when sentiment is very negative:
```yaml
sentiment:
  min_score_for_buy: 0.0  # Only buy when sentiment is neutral or positive
```

### 2. Aggressive Trading
Focus on assets with strong positive sentiment:
```yaml
sentiment:
  min_score_for_buy: 0.3   # Only buy with strong positive sentiment
  score_weight: 0.5        # Heavily weight sentiment in rankings
```

### 3. Conservative Approach
Use sentiment as a tiebreaker only:
```yaml
sentiment:
  min_score_for_buy: -0.5  # Block only extreme negative sentiment
  score_weight: 0.05       # Minimal influence on rankings
```

### 4. Analysis Only
Monitor sentiment without affecting trades:
```yaml
sentiment:
  enabled: true
  use_for_recommendations: false  # Display sentiment but don't filter trades
```

## Files Modified/Created

### New Files
1. **`bot/sentiment_analyzer.py`**: Core sentiment analysis engine
   - `SentimentAnalyzer` class with keyword matching
   - `SentimentScore` dataclass for results
   - Configurable lookback periods

### Modified Files
1. **`bot/top5_trader.py`**:
   - Added sentiment fields to `Recommendation` dataclass
   - Integrated sentiment analysis in `top_recommendations()`
   - Added sentiment display in `scan_portfolio_signals()`
   - Added sentiment caching in `maybe_print_news()`

2. **`config/config.yaml`**:
   - Added `sentiment` configuration section

## Technical Details

### Sentiment Calculation Algorithm

1. **Text Analysis**: Each news item is scanned for positive, negative, and high-impact keywords
2. **Score Calculation**:
   ```
   net_sentiment = positive_keywords - negative_keywords
   total_keywords = positive_keywords + negative_keywords + high_impact_keywords
   score = (net_sentiment / total_keywords) * 0.8  # scaled and smoothed
   ```
3. **Clamping**: Final score is clamped to [-1.0, +1.0] range

### News Cache Management
- Stores last 200 news items in memory
- Updates automatically when new news is fetched
- Uses 24-hour lookback by default
- Filters news items by matched assets

### Performance Considerations
- Sentiment analysis is fast (regex-based pattern matching)
- Minimal overhead: ~1-2ms per asset
- News cache prevents redundant API calls
- Only analyzes news for relevant assets

## Troubleshooting

### No Sentiment Showing
**Problem**: Sentiment not displaying in output  
**Solution**: 
1. Check that `sentiment.enabled: true` in config.yaml
2. Ensure `news.enabled: true` (sentiment requires news)
3. Wait for news to be fetched (happens once per hour by default)
4. Check that assets have recent news coverage

### Sentiment Always Neutral
**Problem**: All sentiment scores are close to 0.0  
**Solution**:
1. News might not contain strong positive/negative keywords
2. Increase `lookback_hours` to capture more news
3. Check that news sources are working (RSS feeds accessible)
4. Verify assets have actual news coverage

### Blocking Too Many Trades
**Problem**: Sentiment filtering prevents desired trades  
**Solution**:
1. Lower `min_score_for_buy` (e.g., from -0.3 to -0.5)
2. Set `use_for_recommendations: false` to disable filtering
3. Check sentiment scores manually to understand market conditions

### Sentiment Seems Wrong
**Problem**: Sentiment doesn't match your perception of news  
**Solution**:
1. Remember sentiment is keyword-based, not AI-powered
2. Check `recent_headlines` in sentiment output
3. Keyword lists can be customized in `sentiment_analyzer.py`
4. Consider news might include both positive and negative aspects

## Future Enhancements

Potential improvements for future versions:
- Integration with sentiment API services (FinBERT, etc.)
- Historical sentiment tracking and visualization
- Asset-specific keyword weighting
- Social media sentiment (Twitter/Reddit)
- Sentiment momentum indicators
- Alert system for sudden sentiment changes

## Example Workflow

1. **Bot starts up**: Sentiment analyzer initialized if enabled
2. **News fetched**: Every hour (configurable), new news items collected
3. **Sentiment cached**: News items stored in memory for analysis
4. **Recommendations generated**: 
   - ML predictions calculated
   - Sentiment analyzed for each candidate
   - Scores combined based on weights
   - Filtered by minimum sentiment threshold
5. **Portfolio scanned**: Sentiment displayed for held assets
6. **Trades executed**: Only if sentiment passes threshold

## Best Practices

1. **Start Conservative**: Begin with `min_score_for_buy: -0.3` and adjust based on results
2. **Monitor Output**: Watch sentiment scores for a few days before heavy reliance
3. **Combine with ML**: Use sentiment as a filter, not a replacement for ML predictions
4. **Stay Updated**: News keywords and patterns evolve; update lists periodically
5. **Backtest**: Test different sentiment parameters in paper mode first

## Support

For issues or questions:
1. Check logs for sentiment-related messages: `[SENTIMENT]` prefix
2. Verify news module is working: `[NEWS]` output
3. Review configuration in `config/config.yaml`
4. Check that news RSS feeds are accessible

---

**Note**: Sentiment analysis is a supplementary tool. Always combine it with technical analysis, ML predictions, and your own market research for best results.
