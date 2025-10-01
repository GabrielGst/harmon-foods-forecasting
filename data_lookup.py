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
  # Plot Sales vs each other variable in separate charts
  if var != 'Sales':
    # Plot original scale
    plt.figure(figsize=(8, 4))
    plt.scatter(df[var], df['Sales'], alpha=0.7, color='teal')
    plt.title(f'Sales vs {var}')
    plt.xlabel(var)
    plt.ylabel('Sales')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'outputs/data_lookup/sales/sales_vs_{var.lower()}.png')
    plt.close()

    # Plot log1p scale
    plt.figure(figsize=(8, 4))
    plt.scatter(np.log1p(df[var]), df['Sales'], alpha=0.7, color='purple')
    plt.title(f'Sales vs Log({var})')
    plt.xlabel(f'Log({var})')
    plt.ylabel('Sales')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'outputs/data_lookup/sales/sales_vs_log_{var.lower()}.png')
    plt.close()
  
  plt.figure(figsize=(10, 4))
  plt.plot(df['TIME'], np.log1p(df[var]), marker='o', color='orange')
  plt.title(f'Log of {var} over Time')
  plt.xlabel('Time')
  plt.ylabel(f'Log({var})')
  plt.grid(True)
  plt.tight_layout()
  plt.savefig(f'outputs/data_lookup/time_series/log_{var.lower()}_over_time.png')
  plt.close()
  
  plt.figure(figsize=(10, 4))
  plt.plot(df['TIME'], df[var], marker='o')
  plt.title(f'{var} over Time')
  plt.xlabel('Time')
  plt.ylabel(var)
  plt.grid(True)
  plt.tight_layout()
  # plt.show()
  plt.savefig(f'outputs/data_lookup/time_series/{var.lower()}_over_time.png')
  plt.close()

# Autocorrelation for DA
plt.figure(figsize=(8, 4))
plot_acf(df['DA'], lags=20)
plt.title('Autocorrelation of DA')
plt.tight_layout()
# plt.show()
plt.savefig('outputs/data_lookup/correlation/autocorrelation_da.png')
plt.close()

# Autocorrelation for CP
plt.figure(figsize=(8, 4))
plot_acf(df['CP'], lags=20)
plt.title('Autocorrelation of CP')
plt.tight_layout()
# plt.show()
plt.savefig('outputs/data_lookup/correlation/autocorrelation_cp.png')
plt.close()

# Autocorrelation for SeasIndx
plt.figure(figsize=(8, 4))
plot_acf(df['SeasIndx'], lags=20)
plt.title('Autocorrelation of SeasIndx')
plt.tight_layout()
# plt.show()
plt.savefig('outputs/data_lookup/correlation/autocorrelation_seasindx.png')
plt.close()

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
    with pd.ExcelWriter('outputs/data_lookup/correlation/correlation_matrix.xlsx', engine='openpyxl') as writer:
        correlation_matrix_rounded.to_excel(writer, sheet_name='Correlation Matrix', index=True)
        
        # Also create a summary statistics sheet
        summary_stats = df[correlation_variables].describe()
        summary_stats.to_excel(writer, sheet_name='Summary Statistics', index=True)
        
    print(f"\nCorrelation matrix exported to 'outputs/data_lookup/correlation/correlation_matrix.xlsx'")
except ImportError:
    print("\nWarning: openpyxl not available. Saving as CSV instead.")
    correlation_matrix_rounded.to_csv('outputs/data_lookup/correlation/correlation_matrix.csv')
    df[correlation_variables].describe().to_csv('outputs/data_lookup/correlation/summary_statistics.csv')
    print("Correlation matrix saved as 'outputs/data_lookup/correlation/correlation_matrix.csv'")
    print("Summary statistics saved as 'outputs/data_lookup/correlation/summary_statistics.csv'")

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
plt.savefig('outputs/data_lookup/correlation/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Correlation heatmap saved as 'outputs/data_lookup/correlation_heatmap.png'")
plt.show()