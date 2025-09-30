import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import numpy as np

# Load the data
file_path = 'Harmon Foods Data.xlsx'
df = pd.read_excel(file_path)

# Plot each variable over time
variables = ['Sales', 'DA', 'CP', 'SeasIndx']
for var in variables:
  plt.figure(figsize=(10, 4))
  plt.plot(df['TIME'], df[var], marker='o')
  plt.title(f'{var} over Time')
  plt.xlabel('Time')
  plt.ylabel(var)
  plt.grid(True)
  plt.tight_layout()
  # plt.show()
  plt.savefig(f'outputs/{var.lower()}_over_time.png')

# Autocorrelation for DA
plt.figure(figsize=(8, 4))
plot_acf(df['DA'], lags=20)
plt.title('Autocorrelation of DA')
plt.tight_layout()
# plt.show()
plt.savefig('outputs/autocorrelation_da.png')

# Autocorrelation for CP
plt.figure(figsize=(8, 4))
plot_acf(df['CP'], lags=20)
plt.title('Autocorrelation of CP')
plt.tight_layout()
# plt.show()
plt.savefig('outputs/autocorrelation_cp.png')

# Autocorrelation for SeasIndx
plt.figure(figsize=(8, 4))
plot_acf(df['SeasIndx'], lags=20)
plt.title('Autocorrelation of SeasIndx')
plt.tight_layout()
# plt.show()
plt.savefig('outputs/autocorrelation_seasindx.png')

# ==========================
# CORRELATION MATRIX ANALYSIS
# ==========================

# Select variables for correlation analysis
correlation_variables = ['Sales', 'DA', 'CP', 'SeasIndx']

# Add lagged variables if they exist in the dataset
if 'CP(t-1)' in df.columns:
    correlation_variables.extend(['CP(t-1)', 'CP(t-2)', 'DA(t-1)', 'DA(t-2)'])

# Compute correlation matrix
correlation_matrix = df[correlation_variables].corr()

# Round to 4 decimal places for better readability
correlation_matrix_rounded = correlation_matrix.round(4)

print("===== Correlation Matrix =====")
print(correlation_matrix_rounded)

# Export correlation matrix to Excel
try:
    with pd.ExcelWriter('outputs/correlation_matrix.xlsx', engine='openpyxl') as writer:
        correlation_matrix_rounded.to_excel(writer, sheet_name='Correlation Matrix', index=True)
        
        # Also create a summary statistics sheet
        summary_stats = df[correlation_variables].describe()
        summary_stats.to_excel(writer, sheet_name='Summary Statistics', index=True)
        
    print(f"\nCorrelation matrix exported to 'outputs/correlation_matrix.xlsx'")
except ImportError:
    print("\nWarning: openpyxl not available. Saving as CSV instead.")
    correlation_matrix_rounded.to_csv('outputs/correlation_matrix.csv')
    df[correlation_variables].describe().to_csv('outputs/summary_statistics.csv')
    print("Correlation matrix saved as 'outputs/correlation_matrix.csv'")
    print("Summary statistics saved as 'outputs/summary_statistics.csv'")

# Create correlation heatmap visualization
plt.figure(figsize=(12, 10))
try:
    import seaborn as sns
    # Use seaborn for a nicer heatmap if available
    sns.heatmap(correlation_matrix_rounded, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .8},
                fmt='.3f')
    plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold', pad=20)
except ImportError:
    # Fallback to matplotlib if seaborn is not available
    im = plt.imshow(correlation_matrix_rounded, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.xticks(range(len(correlation_variables)), correlation_variables, rotation=45, ha='right')
    plt.yticks(range(len(correlation_variables)), correlation_variables)
    plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold', pad=20)
    
    # Add correlation values as text
    for i in range(len(correlation_variables)):
        for j in range(len(correlation_variables)):
            plt.text(j, i, f'{correlation_matrix_rounded.iloc[i, j]:.3f}', 
                    ha='center', va='center', color='black' if abs(correlation_matrix_rounded.iloc[i, j]) < 0.5 else 'white')

plt.tight_layout()
plt.savefig('outputs/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("Correlation heatmap saved as 'outputs/correlation_heatmap.png'")
plt.show()