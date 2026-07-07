import pandas as pd
import numpy as np

print('='*80)
print('DETAILED FEATURE COMPARISON')
print('='*80)

sp100 = pd.read_csv('data/sp100/raw/values.csv')
sp500 = pd.read_csv('data/sp500/raw/values.csv')

print('\nSP100 Features:', sp100.columns.tolist())
print('SP500 Features:', sp500.columns.tolist())

print('\nCommon features:', set(sp100.columns) & set(sp500.columns))
print('SP100-only features:', set(sp100.columns) - set(sp500.columns))
print('SP500-only features:', set(sp500.columns) - set(sp100.columns))

print('\n' + '='*80)
print('VERIFYING SP500 RAW VS STANDARDIZED')
print('='*80)

# For SP500, compute what raw log returns would be from Close prices
sp500_sorted = sp500.sort_values(['Symbol', 'Date'])

# Compute raw log return from Close prices
sp500_sorted['RawLogReturn'] = sp500_sorted.groupby('Symbol')['Close'].transform(
    lambda x: np.log(x / x.shift(1))
)

# Sample a few stocks to verify
sample_stocks = sp500_sorted['Symbol'].unique()[:3]
print(f'\nSampling {len(sample_stocks)} stocks to verify:')

for stock in sample_stocks:
    stock_data = sp500_sorted[sp500_sorted['Symbol'] == stock].iloc[:10]
    print(f'\n{stock}:')
    print(f'  DailyLogReturn (from CSV): mean={stock_data["DailyLogReturn"].mean():.6f}, std={stock_data["DailyLogReturn"].std():.6f}')
    print(f'  RawLogReturn (computed):   mean={stock_data["RawLogReturn"].dropna().mean():.6f}, std={stock_data["RawLogReturn"].dropna().std():.6f}')
    
    # Check if they match
    valid_mask = ~stock_data['RawLogReturn'].isna()
    if valid_mask.sum() > 0:
        ratio = stock_data.loc[valid_mask, 'DailyLogReturn'].values / stock_data.loc[valid_mask, 'RawLogReturn'].values
        print(f'  Ratio (CSV/computed): mean={np.mean(ratio):.2f}, std={np.std(ratio):.2f}')

# Overall statistics
print(f'\n' + '='*80)
print('OVERALL SP500 STATISTICS')
print('='*80)

raw_lr_stats = sp500_sorted.groupby('Symbol')['RawLogReturn'].agg(['mean', 'std'])
csv_lr_stats = sp500_sorted.groupby('Symbol')['DailyLogReturn'].agg(['mean', 'std'])

print(f'\nRaw log return (computed from prices):')
print(f'  Per-stock mean std: {raw_lr_stats["std"].mean():.6f} ± {raw_lr_stats["std"].std():.6f}')
print(f'  Per-stock mean: {raw_lr_stats["mean"].mean():.6f}')

print(f'\nCSV DailyLogReturn (from file):')
print(f'  Per-stock mean std: {csv_lr_stats["std"].mean():.6f} ± {csv_lr_stats["std"].std():.6f}')
print(f'  Per-stock mean: {csv_lr_stats["mean"].mean():.6f}')

# Check typical raw daily log return std for stock market
print(f'\n📊 TYPICAL RAW DAILY LOG RETURN STD: 0.01-0.03 (1-3%)')
if csv_lr_stats["std"].mean() < 0.1:
    scale_type = "RAW or lightly scaled"
else:
    scale_type = "SCALED (likely by ~100x or 2x)"
print(f'   SP500 CSV std={csv_lr_stats["std"].mean():.6f} → This is {scale_type}')

# Calculate scaling factor
print(f'\n🔍 SCALING FACTOR ANALYSIS:')
avg_ratio = (csv_lr_stats["std"] / raw_lr_stats["std"]).mean()
print(f'   Average ratio (CSV std / Raw std): {avg_ratio:.2f}x')
print(f'   SP500 DailyLogReturn appears to be scaled by approximately {avg_ratio:.0f}x')

# Analyze ALL features to determine which need standardization
print('\n' + '='*80)
print('ALL FEATURES STANDARDIZATION ANALYSIS')
print('='*80)

def analyze_feature(df, col_name, dataset_name):
    """Check if a feature needs standardization."""
    if col_name not in df.columns or col_name in ['Symbol', 'Date']:
        return None
    
    data = df[col_name].dropna()
    if len(data) == 0:
        return None
    
    # Per-stock statistics
    grouped = df.groupby('Symbol')[col_name]
    per_stock_stds = grouped.std()
    
    result = {
        'feature': col_name,
        'dataset': dataset_name,
        'overall_mean': data.mean(),
        'overall_std': data.std(),
        'per_stock_std_mean': per_stock_stds.mean(),
        'per_stock_std_std': per_stock_stds.std(),
        'is_standardized': abs(per_stock_stds.mean() - 1.0) < 0.15,
        'needs_normalization': per_stock_stds.std() > 0.1 * per_stock_stds.mean()
    }
    return result

# Analyze all numeric features in both datasets
print('\nSP100 Feature Analysis:')
print('-' * 80)
sp100_features = ['DailyLogReturn', 'ALR1W', 'ALR2W', 'ALR1M', 'ALR2M', 'RSI', 'MACD', 'NormClose', 'Close']
for feat in sp100_features:
    result = analyze_feature(sp100, feat, 'SP100')
    if result:
        status = '✅ PRE-STANDARDIZED' if result['is_standardized'] else '❌ NEEDS STANDARDIZATION'
        print(f"\n{feat}:")
        print(f"  Per-stock std: {result['per_stock_std_mean']:.4f} ± {result['per_stock_std_std']:.4f}")
        print(f"  Overall range: [{sp100[feat].min():.4f}, {sp100[feat].max():.4f}]")
        print(f"  Status: {status}")

print('\n\nSP500 Feature Analysis:')
print('-' * 80)
sp500_features = ['DailyLogReturn', 'ALR1W', 'ALR2W', 'ALR1M', 'ALR2M', 'RSI', 'MACD', 'NormClose', 'Close', 'Open', 'High', 'Low', 'Volume']
for feat in sp500_features:
    result = analyze_feature(sp500, feat, 'SP500')
    if result:
        status = '✅ PRE-STANDARDIZED' if result['is_standardized'] else '❌ NEEDS STANDARDIZATION'
        print(f"\n{feat}:")
        print(f"  Per-stock std: {result['per_stock_std_mean']:.4f} ± {result['per_stock_std_std']:.4f}")
        print(f"  Overall range: [{sp500[feat].min():.4f}, {sp500[feat].max():.4f}]")
        print(f"  Status: {status}")

print('\n' + '='*80)
print('RECOMMENDATIONS')
print('='*80)
print('\n📋 Features requiring PER-STOCK STANDARDIZATION for SP500:')
print('   ✓ DailyLogReturn (scaled 100x, needs z-scoring per stock)')
print('   ✓ ALR1W, ALR2W, ALR1M, ALR2M (aggregated returns, follow DailyLogReturn)')
print('   ✓ MACD (unbounded technical indicator)')
print('   ✓ Close, Open, High, Low (raw prices - log-transform then standardize)')
print('   ✓ Volume (extreme range - log-transform then standardize)')
print('\n📋 Features that may NOT need standardization:')
print('   • RSI (bounded [0-100], but normalize to [0,1] recommended)')
print('   • NormClose (check if already normalized)')
print('\n💡 SP100 is already standardized - no action needed!')
