# News & Sentiment in Recommendations

## Summary of Changes

The bot now displays **news headlines and sentiment analysis** directly when recommendations are generated. This provides immediate context for trading decisions.

## What Was Changed

### 1. **Recommendation Data Structure**
- Added `news_headlines` field to store recent news articles related to the asset

### 2. **Sentiment Calculation Enhanced**
- Now captures not just the sentiment score, but also the actual news headlines
- Headlines are extracted from the last 24 hours of news for each recommended asset

### 3. **Display Format**
When recommendations hit, the output now shows:
- **Basic Info**: Symbol, price, 24h change, trend, signal, action
- **ML Predictions**: TP50, TP80, TP90 target prices (if available)
- **Sentiment**: Label (BULLISH/NEUTRAL/BEARISH), score (-1 to +1), news count
- **📰 News Headlines**: Top 3 recent headlines related to the asset

## Example Output

```
[ENGINE] Top 5 recommendations (quote=USD):
   1. BTC/USD      price=87665.000000 24h=+2.34% trend=BULL sig=LONG -> BUY | ML TP50=89087.590000 TP80=90565.370000 TP90=91252.070000 | SENTIMENT: BULLISH (+0.45, 8 news)
      📰 Recent News for BTC:
         1. Bitcoin ETF Sees Record Inflows as Institutional Demand Surges
         2. MicroStrategy Announces Additional Bitcoin Purchase of $500M
         3. Bitcoin Network Hash Rate Reaches All-Time High Amid Price Rally
         ... and 5 more

   2. ETH/USD      price=3245.120000 24h=+1.87% trend=BULL sig=LONG -> BUY | ML TP50=3312.450000 TP80=3389.870000 TP90=3421.560000 | SENTIMENT: NEUTRAL (+0.12, 5 news)
      📰 Recent News for ETH:
         1. Ethereum Developers Announce Successful Testnet Upgrade
         2. Major DeFi Protocol Launches on Ethereum Layer 2
         3. Ethereum Gas Fees Drop to Lowest Levels in Months

   3. SOL/USD      price=98.45000000 24h=+3.12% trend=BULL sig=LONG -> BUY | ML TP50=101.230000 TP80=104.560000 TP90=106.120000 | SENTIMENT: BULLISH (+0.67, 12 news)
      📰 Recent News for SOL:
         1. Solana Foundation Announces Partnership with Major Payment Processor
         2. Solana Network Sees 40% Increase in Daily Active Users
         3. New Solana-Based NFT Marketplace Launches with Zero Fees

   4. AVAX/USD     price=42.15000000 24h=+0.95% trend=BULL sig=LONG -> HOLD | ML TP50=43.120000 TP80=44.230000 TP90=44.890000 | SENTIMENT: NEUTRAL (-0.05, 3 news)
      📰 Recent News for AVAX:
         1. Avalanche Subnet Adoption Grows as Enterprise Projects Launch
         2. Avalanche Foundation Announces $100M Developer Fund
         3. Technical Analysis: AVAX Forms Bullish Pattern

   5. MATIC/USD    price=0.89500000 24h=+1.23% trend=BULL sig=LONG -> HOLD | SENTIMENT: BEARISH (-0.32, 4 news)
      📰 Recent News for MATIC:
         1. Polygon Team Addresses Network Congestion Concerns
         2. Competition Heats Up as Rival Layer 2 Solutions Gain Traction
         3. Polygon Daily Transactions Show Slight Decline This Week
         4. Regulatory Concerns Impact Sentiment for Polygon Token
```

## Features

### Sentiment Score Integration
- **Score Range**: -1.0 (very bearish) to +1.0 (very bullish)
- **Labels**: 
  - `BULLISH` (+0.3 to +1.0)
  - `NEUTRAL` (-0.3 to +0.3)
  - `BEARISH` (-1.0 to -0.3)
- **News Count**: Shows how many news articles were analyzed

### News Display
- Shows **top 3 headlines** for each recommendation
- If more than 3 articles exist, shows count: "... and X more"
- Only displays when relevant news is available (last 24 hours)
- Empty line after news for better readability

### Sentiment Impact on Trading
The sentiment is already used to:
1. **Filter recommendations**: Assets with very negative sentiment can be filtered out
2. **Score boost**: Positive sentiment increases recommendation priority
3. **Visual context**: Traders can see WHY an asset is being recommended

## Configuration

Sentiment behavior is controlled in `config/config.yaml`:

```yaml
sentiment:
  use_for_recommendations: true      # Apply sentiment to recommendations
  min_score_for_buy: -0.3            # Minimum sentiment score to allow BUY action
  score_weight: 0.1                  # How much sentiment affects ranking (0-1)
  lookback_hours: 24.0               # Hours of news to analyze
```

## Benefits

1. **Informed Decisions**: See the news driving market sentiment
2. **Context Awareness**: Understand WHY an asset is being recommended
3. **Risk Management**: Negative news can help avoid risky trades
4. **Sentiment Validation**: ML predictions combined with news sentiment
5. **Real-time Updates**: News refreshes every cycle (typically 5-10 minutes)

## News Sources

The bot aggregates news from:
- CoinDesk
- Cointelegraph
- CryptoSlate
- CryptoPotato
- The Defiant
- Decrypt

All sources are monitored continuously, with new articles analyzed for sentiment within minutes of publication.
