from calculation import analyze_long_term_stocks

if __name__ == "__main__":
    my_watchlist = [
        "AMD", "NVDA", "AAL", "MU", "AVGO",
        "TSM", "TSLA", "ORCL", "MSFT", "AMZN",
        "META", "AAPL", "GOOGL", "MSTR"
    ]

    analysis_results = analyze_long_term_stocks(
        my_watchlist,
        resolution="DAY",
        ma_short=20,
        ma_long=50,
        rsi_window=21
    )

    # Save results to file
    with open("watchlist_report.txt", "w") as f:
        f.write("Top Long-Term Investment Candidates:\n")
        for stock in analysis_results[:5]:
            f.write(f"{stock['rank']}. {stock['epic']} ({stock['priority']}) - "
                    f"Score: {stock['score']}/100\n")