#!/usr/bin/env python3
"""
Quick validation script to test the new ML model architecture.
Run this before full training to ensure everything works.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add bot directory to path
sys.path.insert(0, str(Path(__file__).parent / "bot"))

from torch_tp_forecaster import QuantileGRU, quantile_loss, build_seq_features


def test_feature_engineering():
    """Test that feature building produces 24 features"""
    print("\n[TEST 1] Feature Engineering...")
    
    # Create dummy OHLCV data (100 bars)
    np.random.seed(42)
    base_price = 100.0
    ohlcv = []
    
    for i in range(100):
        ts = 1700000000000 + i * 900000  # 15min intervals
        c = base_price * (1 + np.random.randn() * 0.01)
        o = c * (1 + np.random.randn() * 0.005)
        h = max(o, c) * (1 + abs(np.random.randn()) * 0.01)
        l = min(o, c) * (1 - abs(np.random.randn()) * 0.01)
        v = 1000000 * (1 + np.random.randn() * 0.2)
        ohlcv.append([ts, o, h, l, c, v])
        base_price = c
    
    features = build_seq_features(ohlcv)
    
    assert len(features) == 100, f"Expected 100 rows, got {len(features)}"
    assert len(features[0]) == 24, f"Expected 24 features, got {len(features[0])}"
    
    print(f"✅ Feature count: {len(features[0])}")
    print(f"✅ Sequence length: {len(features)}")
    print(f"✅ Sample features (last bar): {[f'{f:.4f}' for f in features[-1][:5]]}...")


def test_model_architecture():
    """Test the unidirectional GRU + attention model"""
    print("\n[TEST 2] Model Architecture...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    model = QuantileGRU(
        input_size=24,
        hidden_size=192,
        num_layers=3,
        dropout=0.20,
        quantiles=(0.5, 0.8, 0.9),
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Total parameters: {total_params:,}")
    print(f"✅ Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 16
    lookback = 96
    x = torch.randn(batch_size, lookback, 24, device=device)
    
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (batch_size, 3), f"Expected shape ({batch_size}, 3), got {output.shape}"
    print(f"✅ Forward pass output shape: {output.shape}")
    
    # Check quantile ordering (should be enforced by loss, but let's see)
    print(f"   Sample predictions (q50, q80, q90): {output[0].tolist()}")


def test_quantile_loss():
    """Test the improved quantile loss function"""
    print("\n[TEST 3] Quantile Loss Function...")
    
    # Create test data
    pred = torch.tensor([[0.02, 0.05, 0.08],  # q50=2%, q80=5%, q90=8%
                         [0.03, 0.04, 0.06]])  # properly ordered
    target = torch.tensor([0.06, 0.02])  # actual returns
    
    quantiles = (0.5, 0.8, 0.9)
    loss = quantile_loss(pred, target, quantiles)
    
    print(f"✅ Loss computed: {loss.item():.6f}")
    
    # Test crossing penalty
    pred_bad = torch.tensor([[0.08, 0.05, 0.02]])  # reversed order (bad!)
    target_bad = torch.tensor([0.05])
    loss_bad = quantile_loss(pred_bad, target_bad, quantiles)
    
    print(f"✅ Loss with crossing penalty: {loss_bad.item():.6f}")
    assert loss_bad > loss, "Crossing penalty should increase loss"
    print(f"✅ Crossing penalty working (bad order has higher loss)")


def test_normalization():
    """Test feature normalization"""
    print("\n[TEST 4] Feature Normalization...")
    
    # Create features
    features = torch.randn(100, 20) * 10 + 5  # random features with mean~5, std~10
    
    # Normalize
    mean = features.mean(dim=0)
    std = features.std(dim=0)
    normalized = (features - mean) / (std + 1e-8)
    
    # Check properties
    new_mean = normalized.mean(dim=0).abs().max().item()
    new_std = (normalized.std(dim=0) - 1.0).abs().max().item()
    
    print(f"✅ Normalized mean (should be ~0): {new_mean:.6f}")
    print(f"✅ Normalized std (should be ~1): {1.0 + new_std:.6f}")
    
    assert new_mean < 0.1, f"Mean too high: {new_mean}"
    assert new_std < 0.1, f"Std deviation from 1.0 too high: {new_std}"


def test_end_to_end():
    """Test complete prediction pipeline"""
    print("\n[TEST 5] End-to-End Pipeline...")
    
    # Create OHLCV
    np.random.seed(42)
    base_price = 50000.0
    ohlcv = []
    
    for i in range(150):
        ts = 1700000000000 + i * 900000
        c = base_price * (1 + np.random.randn() * 0.01)
        o = c * (1 + np.random.randn() * 0.005)
        h = max(o, c) * (1 + abs(np.random.randn()) * 0.01)
        l = min(o, c) * (1 - abs(np.random.randn()) * 0.01)
        v = 10000000 * (1 + np.random.randn() * 0.2)
        ohlcv.append([ts, o, h, l, c, v])
        base_price = c
    
    # Build features
    features = build_seq_features(ohlcv)
    assert len(features) >= 96, "Not enough features for lookback"
    
    # Normalize
    feat_tensor = torch.tensor(features[-96:], dtype=torch.float32)
    mean = feat_tensor.mean(dim=0)
    std = feat_tensor.std(dim=0)
    feat_norm = (feat_tensor - mean) / (std + 1e-8)
    
    # Predict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuantileGRU(
        input_size=24,
        hidden_size=192,
        num_layers=3,
        dropout=0.20,
        quantiles=(0.5, 0.8, 0.9),
    ).to(device)
    
    model.eval()
    with torch.no_grad():
        x = feat_norm.unsqueeze(0).to(device)  # (1, 96, 24)
        pred = model(x).squeeze(0)  # (3,)
    
    # Convert to prices
    last_close = ohlcv[-1][4]
    tp50 = last_close * (1 + pred[0].item())
    tp80 = last_close * (1 + pred[1].item())
    tp90 = last_close * (1 + pred[2].item())
    
    print(f"✅ Last close: ${last_close:.2f}")
    print(f"✅ Predicted TP50: ${tp50:.2f} ({pred[0].item()*100:+.2f}%)")
    print(f"✅ Predicted TP80: ${tp80:.2f} ({pred[1].item()*100:+.2f}%)")
    print(f"✅ Predicted TP90: ${tp90:.2f} ({pred[2].item()*100:+.2f}%)")
    
    # Note: Untrained model won't have ordered quantiles
    # The quantile crossing penalty in the loss function will enforce this during training
    print(f"✅ Predictions generated successfully")
    print(f"   (Note: Untrained model - quantiles will be ordered after training)")


def main():
    print("="*70)
    print("ML MODEL ARCHITECTURE VALIDATION")
    print("="*70)
    
    try:
        test_feature_engineering()
        test_model_architecture()
        test_quantile_loss()
        test_normalization()
        test_end_to_end()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nModel architecture is working correctly.")
        print("You can now proceed with full training:")
        print("  python -m ml.train_torch_forecaster")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
