import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

# Create outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

@dataclass
class ModelConfig:
    """Configuration for model variables and targets"""
    # Model predictor columns
    sales_v1_cols: List[str]  # Lag-only model
    
    # Target columns
    sales_target: str = 'Sales'

@dataclass 
class ForecastResults:
    """Container for storing forecast results"""
    # Predictions and actuals
    sales_pred_v1: List[float]
    sales_true: List[float]
    
    # R² values
    r2_values: Dict[str, List[float]]
    # Adjusted R² values
    adj_r2_values: Dict[str, List[float]]
    
    # January 1988 predictions from each iteration
    jan88_predictions: Dict[str, List[float]]
    
    # Cumulative performance metrics
    cumulative_rmse: List[float]
    cumulative_mape: List[float]
    
    # Model coefficient statistics (for last iteration)
    final_model_stats: Dict[str, Dict]

class HarmonFoodsForecaster:
    """Main forecasting class for Harmon Foods analysis"""
    
    def __init__(self, file_path: str, initial_train: int = 24):
        self.file_path = file_path
        self.initial_train = initial_train
        self.df = None
        self.config = None
        self.results = None
        
    def exclude_rows(self, df: pd.DataFrame, rows_to_exclude: List[int]) -> pd.DataFrame:
        """Exclude specified rows from the dataframe"""
        if rows_to_exclude:
            print(f"Excluding rows: {rows_to_exclude}")
            print("Rows excluded:")
            print(df.iloc[rows_to_exclude])
            df = df.drop(index=rows_to_exclude, errors='ignore').reset_index(drop=True)
            
        return df
    
    def load_and_prepare_data(self, rows_to_exclude: List[int] = None) -> pd.DataFrame:
        """Load and prepare the data with transformations"""
        df = pd.read_excel(self.file_path)
        df.columns = df.columns.str.strip()
        
        # Exclude specified rows if provided
        if rows_to_exclude is None:
            rows_to_exclude = [12, 24, 38, 40, 41, 44]  # Default exclusion list
        df = self.exclude_rows(df, rows_to_exclude)
        
        # Convert to float
        cols_to_float = ['Sales', 'CP', 'CP(t-1)', 'CP(t-2)', 'DA', 'DA(t-1)', 'DA(t-2)', 'SeasIndx']
        df[cols_to_float] = df[cols_to_float].astype(float)
        
        # Add January 1988 row for prediction
        jan_88_row = {
            'CP': 100000,
            'CP(t-1)': df['CP'].iloc[-1],
            'CP(t-2)': df['CP(t-1)'].iloc[-1],
            'DA': 500000,
            'DA(t-1)': df['DA'].iloc[-1],
            'DA(t-2)': df['DA(t-1)'].iloc[-1],
            'SeasIndx': df['SeasIndx'].iloc[-12],  # Use actual value from 12 months ago
            'lag_12': df['SeasIndx'].iloc[-12],
            'Sales': np.nan
        }
        
        # Append January 1988 row
        df = pd.concat([df, pd.DataFrame([jan_88_row])], ignore_index=True)
        
        self.df = df
        return df
    
    def setup_model_config(self) -> ModelConfig:
        """Setup model configuration"""
        config = ModelConfig(
            # All variables
            sales_v1_cols=['CP', 'CP(t-1)', 'CP(t-2)', 'DA', 'DA(t-1)', 'DA(t-2)', 'SeasIndx']
            
            # Adjusted model
            # sales_v1_cols=['CP', 'CP(t-1)', 'DA', 'SeasIndx']
        )
        self.config = config
        return config
    
    def initialize_results(self) -> ForecastResults:
        """Initialize results container"""
        results = ForecastResults(
            sales_pred_v1=[], sales_true=[],
            r2_values={'sales_v1': []},
            adj_r2_values={'sales_v1': []},
            jan88_predictions={'v1': []},
            cumulative_rmse=[],
            cumulative_mape=[],
            final_model_stats={}
        )
        self.results = results
        return results
    
    def extract_model_statistics(self, model, model_name: str) -> Dict:
        """Extract comprehensive statistics from a fitted model"""
        # Get summary with 80% confidence intervals (alpha=0.20)
        summary = model.summary2(alpha=0.20)
        coeff_table = summary.tables[1]
        
        # Extract coefficient statistics
        coefficients = {}
        for var_name in coeff_table.index:
            coefficients[var_name] = {
                'coefficient': coeff_table.loc[var_name, 'Coef.'],
                'std_error': coeff_table.loc[var_name, 'Std.Err.'],
                't_score': coeff_table.loc[var_name, 't'],
                'p_value': coeff_table.loc[var_name, 'P>|t|'],
                'conf_lower': coeff_table.loc[var_name, '[0.1'],  # 80% CI lower bound
                'conf_upper': coeff_table.loc[var_name, '0.9]']   # 80% CI upper bound
            }
        
        return {
            'model_name': model_name,
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'aic': model.aic,
            'bic': model.bic,
            'f_statistic': model.fvalue,
            'f_pvalue': model.f_pvalue,
            'n_observations': model.nobs,
            'coefficients': coefficients,
            'summary_text': str(summary)
        }
    
    def predict_with_model(self, model, X_test: pd.DataFrame) -> float:
        """Helper to make predictions"""
        X_test_const = sm.add_constant(X_test, has_constant='add')
        return model.predict(X_test_const).iloc[0]
    
    def fit_and_predict_step(self, t: int) -> Tuple[Dict, Dict]:
        """Perform one iteration of fitting and prediction"""
        # Split data - exclude January 1988 row from training
        train = self.df.iloc[12:t].copy()  # Start from row 12, end at t (exclusive)
        test = self.df.iloc[t:t+1].copy()
        jan88_row = self.df.iloc[-1:].copy()  # Last row is January 1988
        
        models = {}
        predictions = {}
        
        # 1. Sales Model V1 (using actual values)
        X1_train = sm.add_constant(train[self.config.sales_v1_cols], has_constant='add')
        y1_train = train[self.config.sales_target]
        sales_model_v1 = sm.OLS(y1_train, X1_train).fit()
        models['sales_v1'] = sales_model_v1
        
        X1_test = sm.add_constant(test[self.config.sales_v1_cols], has_constant='add')
        y1_pred = sales_model_v1.predict(X1_test).iloc[0]
        predictions['sales_v1'] = y1_pred
        
        # 2. January 1988 predictions with this iteration's models
        jan88_predictions = self.predict_jan88(models, jan88_row)
        predictions.update(jan88_predictions)
        
        return models, predictions
    
    def predict_jan88(self, models: Dict, jan88_row: pd.DataFrame) -> Dict:
        """Predict January 1988 values using current models"""
        jan88_row = jan88_row.copy()
        
        # Predict Sales V1 for Jan 1988 (using actual SeasIndx value)
        X_sales_v1_jan88 = sm.add_constant(jan88_row[self.config.sales_v1_cols], has_constant='add')
        sales_pred_v1_jan88 = models['sales_v1'].predict(X_sales_v1_jan88).iloc[0]
        
        return {
            'jan88_sales_v1': sales_pred_v1_jan88
        }
    
    def calculate_cumulative_metrics(self) -> None:
        """Calculate cumulative RMSE and MAPE"""
        if len(self.results.sales_true) > 0:
            sales_true_arr = np.array(self.results.sales_true)
            sales_pred_v1_arr = np.array(self.results.sales_pred_v1)
            
            # RMSE
            rmse_v1 = np.sqrt(np.mean((sales_true_arr - sales_pred_v1_arr)**2))
            
            # MAPE with zero protection
            mask = sales_true_arr != 0
            if np.any(mask):
                mape_v1 = np.mean(np.abs((sales_true_arr[mask] - sales_pred_v1_arr[mask]) / sales_true_arr[mask])) * 100
            else:
                mape_v1 = 0.0
            
            self.results.cumulative_rmse.append(rmse_v1)
            self.results.cumulative_mape.append(mape_v1)
    
    def run_rolling_forecast(self) -> ForecastResults:
        """Main rolling forecast loop"""
        print("Starting rolling forecast...")
        
        # Get data length (excluding Jan 1988 row)
        n = len(self.df) - 1
        final_models = None  # Store the last iteration's models for detailed stats
        
        for t in tqdm(range(self.initial_train, n), desc="Rolling Forecast"):
            models, predictions = self.fit_and_predict_step(t)
            
            # Store regular predictions
            test_row = self.df.iloc[t]
            self.results.sales_pred_v1.append(predictions['sales_v1'])
            self.results.sales_true.append(test_row[self.config.sales_target])
            
            # Store R² and Adjusted R² values
            self.results.r2_values['sales_v1'].append(models['sales_v1'].rsquared)
            self.results.adj_r2_values['sales_v1'].append(models['sales_v1'].rsquared_adj)
            
            # Store January 1988 predictions
            self.results.jan88_predictions['v1'].append(predictions['jan88_sales_v1'])
            
            # Update cumulative metrics
            self.calculate_cumulative_metrics()
            
            # Store final models for detailed statistics (last iteration)
            final_models = models
        
        # Extract detailed statistics from final models
        if final_models:
            self.results.final_model_stats = {
                'sales_v1': self.extract_model_statistics(final_models['sales_v1'], 'Sales V1 Model')
            }
        
        return self.results
    
    def calculate_final_model_predictions(self) -> Dict:
        """Calculate predictions using the final iteration's model on all test data"""
        if not self.results.final_model_stats:
            print("No final model available. Run the forecast first.")
            return {}
        
        # Get the final model (from last iteration)
        final_models = None
        n = len(self.df) - 1  # Exclude Jan 1988 row
        
        # Re-run the last iteration to get the final model
        t = n - 1  # Last iteration
        train = self.df.iloc[12:t].copy()
        
        # Fit final model
        X1_train = sm.add_constant(train[self.config.sales_v1_cols], has_constant='add')
        y1_train = train[self.config.sales_target]
        final_sales_model = sm.OLS(y1_train, X1_train).fit()
        
        # Calculate predictions for all test periods
        test_predictions = []
        test_actuals = []
        test_periods = []
        
        for i in range(self.initial_train, n):
            test_row = self.df.iloc[i:i+1].copy()
            X_test = sm.add_constant(test_row[self.config.sales_v1_cols], has_constant='add')
            pred = final_sales_model.predict(X_test).iloc[0]
            actual = test_row[self.config.sales_target].iloc[0]
            
            test_predictions.append(pred)
            test_actuals.append(actual)
            test_periods.append(i)
        
        # January 1988 prediction
        jan88_row = self.df.iloc[-1:].copy()
        X_jan88 = sm.add_constant(jan88_row[self.config.sales_v1_cols], has_constant='add')
        jan88_pred = final_sales_model.predict(X_jan88).iloc[0]
        
        # Calculate final metrics
        test_predictions_arr = np.array(test_predictions)
        test_actuals_arr = np.array(test_actuals)
        
        final_rmse = np.sqrt(np.mean((test_actuals_arr - test_predictions_arr)**2))
        mask = test_actuals_arr != 0
        final_mape = np.mean(np.abs((test_actuals_arr[mask] - test_predictions_arr[mask]) / test_actuals_arr[mask])) * 100 if np.any(mask) else 0.0
        
        return {
            'final_model': final_sales_model,
            'test_predictions': test_predictions,
            'test_actuals': test_actuals,
            'test_periods': test_periods,
            'jan88_prediction': jan88_pred,
            'final_rmse': final_rmse,
            'final_mape': final_mape,
            'model_summary': str(final_sales_model.summary())
        }
    
    def print_final_model_predictions(self, final_results: Dict) -> None:
        """Print detailed final model predictions"""
        print("\n" + "="*80)
        print("FINAL MODEL PREDICTIONS (Last Iteration)")
        print("="*80)
        
        print(f"\nFinal Model Performance:")
        print(f"RMSE: {final_results['final_rmse']:,.2f}")
        print(f"MAPE: {final_results['final_mape']:.2f}%")
        
        print(f"\nJanuary 1988 Prediction: {final_results['jan88_prediction']:,.2f}")
        
        print(f"\nTest Period Predictions vs Actuals:")
        print(f"{'Period':<8} {'Actual':<12} {'Predicted':<12} {'Error':<12} {'Error %':<10}")
        print("-" * 60)
        
        for i, (period, actual, pred) in enumerate(zip(final_results['test_periods'], 
                                                      final_results['test_actuals'], 
                                                      final_results['test_predictions'])):
            error = pred - actual
            error_pct = (error / actual * 100) if actual != 0 else 0
            print(f"{period:<8} {actual:<12,.0f} {pred:<12,.0f} {error:<12,.0f} {error_pct:<10.2f}%")
        
        print(f"\nFinal Model Summary:")
        print(final_results['model_summary'])

    def calculate_residual_analysis(self, final_results: Dict) -> Dict:
        """Calculate comprehensive residual analysis"""
        model = final_results['final_model']
        test_predictions = np.array(final_results['test_predictions'])
        test_actuals = np.array(final_results['test_actuals'])
        
        # Calculate residuals
        residuals = test_actuals - test_predictions
        
        # Get fitted values from the model (on training data)
        fitted_values = model.fittedvalues
        training_residuals = model.resid
        
        # Durbin-Watson test
        dw_statistic = durbin_watson(training_residuals)
        
        # Get training data for plotting residuals vs variables
        n = len(self.df) - 1
        t = n - 1  # Last iteration
        train = self.df.iloc[12:t].copy()
        
        # Normality test on residuals
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        jarque_bera_stat, jarque_bera_p = stats.jarque_bera(residuals)
        
        # Fit normal distribution to residuals
        mu, sigma = stats.norm.fit(residuals)
        
        return {
            'residuals': residuals,
            'training_residuals': training_residuals,
            'fitted_values': fitted_values,
            'dw_statistic': dw_statistic,
            'test_periods': final_results['test_periods'],
            'train_data': train,
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p,
            'jarque_bera_stat': jarque_bera_stat,
            'jarque_bera_p': jarque_bera_p,
            'normal_mu': mu,
            'normal_sigma': sigma
        }
    
    def print_residual_statistics(self, residual_results: Dict) -> None:
        """Print residual analysis statistics"""
        print("\n" + "="*80)
        print("RESIDUAL ANALYSIS")
        print("="*80)
        
        residuals = residual_results['residuals']
        
        print(f"\nResidual Statistics:")
        print(f"Mean: {np.mean(residuals):.6f}")
        print(f"Standard Deviation: {np.std(residuals):.4f}")
        print(f"Skewness: {stats.skew(residuals):.4f}")
        print(f"Kurtosis: {stats.kurtosis(residuals):.4f}")
        print(f"Min: {np.min(residuals):.2f}")
        print(f"Max: {np.max(residuals):.2f}")
        
        print(f"\nDurbin-Watson Statistic: {residual_results['dw_statistic']:.4f}")
        print("(Values around 2 indicate no autocorrelation)")
        
        print(f"\nNormality Tests:")
        print(f"Shapiro-Wilk Test:")
        print(f"  Statistic: {residual_results['shapiro_stat']:.4f}")
        print(f"  P-value: {residual_results['shapiro_p']:.6f}")
        print(f"  {'Residuals appear normal' if residual_results['shapiro_p'] > 0.05 else 'Residuals may not be normal'}")
        
        print(f"\nJarque-Bera Test:")
        print(f"  Statistic: {residual_results['jarque_bera_stat']:.4f}")
        print(f"  P-value: {residual_results['jarque_bera_p']:.6f}")
        print(f"  {'Residuals appear normal' if residual_results['jarque_bera_p'] > 0.05 else 'Residuals may not be normal'}")
        
        print(f"\nFitted Normal Distribution:")
        print(f"  Mean (μ): {residual_results['normal_mu']:.4f}")
        print(f"  Standard Deviation (σ): {residual_results['normal_sigma']:.4f}")

    def create_residual_plots(self, residual_results: Dict) -> None:
        """Create comprehensive residual analysis plots"""
        residuals = residual_results['residuals']
        test_periods = residual_results['test_periods']
        fitted_values = residual_results['fitted_values']
        train_data = residual_results['train_data']
        
        # Create figure with 4 subplots (2x2)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Residual Analysis - Final Model', fontsize=16, fontweight='bold')
        
        # Plot 1: Residuals over time
        ax1 = axes[0, 0]
        ax1.plot(test_periods, residuals, 'o-', color='blue', alpha=0.7)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax1.set_title('Residuals Over Time')
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Residuals')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Residuals vs Fitted Values
        ax2 = axes[0, 1]
        # Get actual fitted values by re-predicting
        final_model = None
        n = len(self.df) - 1
        t = n - 1
        train = self.df.iloc[12:t].copy()
        X1_train = sm.add_constant(train[self.config.sales_v1_cols], has_constant='add')
        y1_train = train[self.config.sales_target]
        final_sales_model = sm.OLS(y1_train, X1_train).fit()
        
        test_fitted_actual = []
        for i in range(len(test_periods)):
            period = test_periods[i]
            test_row = self.df.iloc[period:period+1].copy()
            X_test = sm.add_constant(test_row[self.config.sales_v1_cols], has_constant='add')
            fitted_val = final_sales_model.predict(X_test).iloc[0]
            test_fitted_actual.append(fitted_val)
        
        ax2.scatter(test_fitted_actual, residuals, alpha=0.7, color='green')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax2.set_title('Residuals vs Fitted Values')
        ax2.set_xlabel('Fitted Values')
        ax2.set_ylabel('Residuals')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Histogram of residuals with normal distribution
        ax3 = axes[1, 0]
        n_bins = max(8, len(residuals) // 3)
        n_hist, bins, patches = ax3.hist(residuals, bins=n_bins, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Fit and plot normal distribution
        mu = residual_results['normal_mu']
        sigma = residual_results['normal_sigma']
        x = np.linspace(residuals.min(), residuals.max(), 100)
        normal_curve = stats.norm.pdf(x, mu, sigma)
        ax3.plot(x, normal_curve, 'r-', linewidth=2, label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
        ax3.set_title('Residuals Histogram with Normal Fit')
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Autocorrelation of residuals
        ax4 = axes[1, 1]
        if len(residuals) > 1:
            # Calculate autocorrelation manually for lag 1
            residuals_lag1 = residuals[1:]
            residuals_current = residuals[:-1]
            ax4.scatter(residuals_current, residuals_lag1, alpha=0.7, color='purple')
            ax4.axhline(y=0, color='red', linestyle='--', alpha=0.8)
            ax4.axvline(x=0, color='red', linestyle='--', alpha=0.8)
            ax4.set_title('Residuals Autocorrelation (Lag 1)')
            ax4.set_xlabel('Residuals (t)')
            ax4.set_ylabel('Residuals (t+1)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nResidual analysis plots saved as 'outputs/residual_analysis.png'")

    def save_results_to_txt(self, final_model_results: Dict, residual_results: Dict) -> None:
        """Save comprehensive results to a text file"""
        import datetime
        
        with open('outputs/forecasting_results.txt', 'w') as f:
            # Header
            f.write("="*80 + "\n")
            f.write("HARMON FOODS FORECASTING ANALYSIS RESULTS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")
            
            # Final Model Performance
            f.write("FINAL MODEL PREDICTIONS (Last Iteration)\n")
            f.write("="*80 + "\n")
            f.write(f"Final Model Performance:\n")
            f.write(f"RMSE: {final_model_results['final_rmse']:,.2f}\n")
            f.write(f"MAPE: {final_model_results['final_mape']:.2f}%\n")
            f.write(f"\nJanuary 1988 Prediction: {final_model_results['jan88_prediction']:,.2f}\n")
            f.write("\n")
            
            # Test Period Predictions
            f.write("Test Period Predictions vs Actuals:\n")
            f.write(f"{'Period':<8} {'Actual':<12} {'Predicted':<12} {'Error':<12} {'Error %':<10}\n")
            f.write("-" * 60 + "\n")
            
            for i, (period, actual, pred) in enumerate(zip(final_model_results['test_periods'], 
                                                          final_model_results['test_actuals'], 
                                                          final_model_results['test_predictions'])):
                error = pred - actual
                error_pct = (error / actual * 100) if actual != 0 else 0
                f.write(f"{period:<8} {actual:<12,.0f} {pred:<12,.0f} {error:<12,.0f} {error_pct:<10.2f}%\n")
            
            f.write("\n\n")
            
            # Residual Analysis
            f.write("RESIDUAL ANALYSIS\n")
            f.write("="*80 + "\n")
            
            residuals = residual_results['residuals']
            
            f.write(f"Residual Statistics:\n")
            f.write(f"Mean: {np.mean(residuals):.6f}\n")
            f.write(f"Standard Deviation: {np.std(residuals):.4f}\n")
            f.write(f"Skewness: {stats.skew(residuals):.4f}\n")
            f.write(f"Kurtosis: {stats.kurtosis(residuals):.4f}\n")
            f.write(f"Min: {np.min(residuals):.2f}\n")
            f.write(f"Max: {np.max(residuals):.2f}\n")
            f.write("\n")
            
            f.write(f"Durbin-Watson Statistic: {residual_results['dw_statistic']:.4f}\n")
            f.write("(Values around 2 indicate no autocorrelation)\n")
            f.write("\n")
            
            f.write(f"Normality Tests:\n")
            f.write(f"Shapiro-Wilk Test:\n")
            f.write(f"  Statistic: {residual_results['shapiro_stat']:.4f}\n")
            f.write(f"  P-value: {residual_results['shapiro_p']:.6f}\n")
            f.write(f"  {'Residuals appear normal' if residual_results['shapiro_p'] > 0.05 else 'Residuals may not be normal'}\n")
            f.write("\n")
            
            f.write(f"Jarque-Bera Test:\n")
            f.write(f"  Statistic: {residual_results['jarque_bera_stat']:.4f}\n")
            f.write(f"  P-value: {residual_results['jarque_bera_p']:.6f}\n")
            f.write(f"  {'Residuals appear normal' if residual_results['jarque_bera_p'] > 0.05 else 'Residuals may not be normal'}\n")
            f.write("\n")
            
            f.write(f"Fitted Normal Distribution:\n")
            f.write(f"  Mean (μ): {residual_results['normal_mu']:.4f}\n")
            f.write(f"  Standard Deviation (σ): {residual_results['normal_sigma']:.4f}\n")
            f.write("\n\n")
            
            # Model Summary
            f.write("FINAL MODEL SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(final_model_results['model_summary'])
            f.write("\n")
        
        print(f"\nComplete results saved to 'outputs/forecasting_results.txt'")

    def calculate_ensemble_predictions(self) -> Dict[str, float]:
        """Calculate ensemble predictions for January 1988"""
        jan88_v1 = np.array(self.results.jan88_predictions['v1'])
        
        # Get the final prediction (last iteration)
        final_prediction = jan88_v1[-1] if len(jan88_v1) > 0 else 0.0
        
        return {
            'final_prediction': final_prediction,
        }
    
    def print_coefficient_stats(self, model_stats: Dict, model_name: str) -> None:
        """Print detailed coefficient statistics for a model"""
        print(f"\n{'='*60}")
        print(f"{model_name} - Detailed Statistics")
        print(f"{'='*60}")
        
        print(f"R²: {model_stats['r_squared']:.4f}")
        print(f"Adjusted R²: {model_stats['adj_r_squared']:.4f}")
        print(f"F-statistic: {model_stats['f_statistic']:.4f} (p-value: {model_stats['f_pvalue']:.6f})")
        print(f"AIC: {model_stats['aic']:.2f}")
        print(f"BIC: {model_stats['bic']:.2f}")
        print(f"Observations: {int(model_stats['n_observations'])}")
        
        print(f"\nCoefficient Statistics (80% Confidence Intervals):")
        print(f"{'Variable':<15} {'Coeff':<10} {'Std Err':<10} {'t-score':<8} {'P>|t|':<8} {'[0.1':<10} {'0.9]':<10}")
        print("-" * 80)
        
        for var_name, stats in model_stats['coefficients'].items():
            print(f"{var_name:<15} {stats['coefficient']:<10.4f} {stats['std_error']:<10.4f} "
                  f"{stats['t_score']:<8.3f} {stats['p_value']:<8.3f} "
                  f"{stats['conf_lower']:<10.4f} {stats['conf_upper']:<10.4f}")
    
    def print_results(self, ensemble_results: Dict) -> None:
        """Print comprehensive results"""
        pass  # Results printing removed as requested
    
    def create_visualizations(self, ensemble_results: Dict) -> None:
        """Create comprehensive visualizations"""
        # Create simplified figure with 2x2 layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Harmon Foods Forecasting Analysis - Simplified', fontsize=16, fontweight='bold')
        
        iterations = list(range(len(self.results.jan88_predictions['v1'])))
        
        # Plot 1: January 1988 Sales Predictions Evolution
        ax1.plot(iterations, self.results.jan88_predictions['v1'], 'purple', linewidth=2, label='Model V1', marker='o', markersize=3)
        ax1.set_title('January 1988 Sales Predictions Evolution')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Predicted Sales')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative RMSE Evolution
        ax2.plot(iterations, self.results.cumulative_rmse, 'purple', linewidth=2, label='Model V1', marker='o', markersize=3)
        ax2.set_title('Cumulative RMSE Evolution')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Cumulative RMSE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: R² Evolution for Sales Model
        ax3.plot(iterations, self.results.r2_values['sales_v1'], 'purple', linewidth=2, label='Sales V1 R²', marker='o', markersize=3)
        ax3.plot(iterations, self.results.adj_r2_values['sales_v1'], 'purple', linewidth=2, label='Sales V1 Adj R²', marker='d', markersize=3, alpha=0.7, linestyle='--')
        ax3.set_title('R² and Adjusted R² Evolution - Sales Model')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('R² / Adjusted R²')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Sales Predictions vs Actuals
        ax4.scatter(self.results.sales_true, self.results.sales_pred_v1, alpha=0.6, color='purple')
        min_val = min(min(self.results.sales_true), min(self.results.sales_pred_v1))
        max_val = max(max(self.results.sales_true), max(self.results.sales_pred_v1))
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
        ax4.set_title('Sales Predictions vs Actuals')
        ax4.set_xlabel('Actual Sales')
        ax4.set_ylabel('Predicted Sales')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/harmon_foods_analysis_simplified.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nSimplified analysis visualization saved as 'outputs/harmon_foods_analysis_simplified.png'")

def main():
    """Main execution function"""
    # Initialize forecaster
    forecaster = HarmonFoodsForecaster("Harmon Foods Data.xlsx", initial_train=24)
    
    # Load and prepare data with row exclusions
    df = forecaster.load_and_prepare_data(rows_to_exclude=[12, 24, 38, 40, 41, 44])
    print(f"Data loaded: {len(df)} rows (including Jan 1988, after exclusions)")
    
    # Setup configuration
    config = forecaster.setup_model_config()
    
    # Initialize results
    results = forecaster.initialize_results()
    
    # Run rolling forecast
    results = forecaster.run_rolling_forecast()
    
    # Calculate ensemble predictions
    ensemble_results = forecaster.calculate_ensemble_predictions()
    
    # Calculate and print final model predictions
    final_model_results = forecaster.calculate_final_model_predictions()
    forecaster.print_final_model_predictions(final_model_results)
    
    # Perform residual analysis
    residual_results = forecaster.calculate_residual_analysis(final_model_results)
    forecaster.print_residual_statistics(residual_results)
    forecaster.create_residual_plots(residual_results)
    
    # Save results to text file
    forecaster.save_results_to_txt(final_model_results, residual_results)
    
    # Print results
    forecaster.print_results(ensemble_results)
    
    # Create visualizations
    forecaster.create_visualizations(ensemble_results)
    
    return forecaster, ensemble_results

if __name__ == "__main__":
    forecaster, ensemble_results = main()