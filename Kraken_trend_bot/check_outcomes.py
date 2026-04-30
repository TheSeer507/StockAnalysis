import json

data = json.load(open('data/ml_predictions.json'))
records = data['records']
with_outcomes = [r for r in records if r.get('outcome_recorded')]

print(f"Total records: {len(records):,}")
print(f"With recorded outcomes: {len(with_outcomes):,}")

if with_outcomes:
    print("\nFirst 5 with outcomes:")
    for i, r in enumerate(with_outcomes[:5]):
        print(f"\n{i+1}. {r['symbol']}:")
        print(f"   Predicted TP80: {r.get('pred_ret80', 0):.2%}")
        print(f"   Actual peak: {r.get('actual_peak_return', 0):.2%}")
        print(f"   Hit TP80: {r.get('actual_peak_return', 0) >= r.get('pred_ret80', 0)}")
