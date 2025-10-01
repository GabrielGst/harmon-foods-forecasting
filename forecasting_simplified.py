import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
import matplotlib.pyplot as plt
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
    sales_v2_cols: List[str]  # Full model with current CP/DA
    cp_cols: List[str]
    da_cols: List[str]
    si_cols: List[str]
    
    # Target columns
    sales_target: str = 'Sales'
    cp_target: str = 'CP'
    da_target: str = 'DA'
    si_target: str = 'SeasIndx'

@dataclass 
class ForecastResults:
    """Container for storing forecast results"""
    # Predictions and actuals
    sales_pred_v1: List[float]
    sales_pred_v2: List[float]
    sales_pred_v2_official: List[float]
    sales_true: List[float]
    
    cp_pred: List[float]
    da_pred: List[float]
    si_pred: List[float]
    cp_true: List[float]
    da_true: List[float]
    si_true: List[float]
    
    # R² values
    r2_values: Dict[str, List[float]]
    # Adjusted R² values
    adj_r2_values: Dict[str, List[float]]
    
    # January 1988 predictions from each iteration
    jan88_predictions: Dict[str, List[float]]
    
    # Cumulative performance metrics
    cumulative_rmse: Dict[str, List[float]]
    cumulative_mape: Dict[str, List[float]]
    
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
        
        # Add artificial 'Exceptional Event' column
        df['Exceptional Event'] = 0
        exceptional_events = [12, 38, 40, 41, 44]
        for idx in exceptional_events:
            if idx < len(df):  # Make sure the index exists after exclusions
                df.loc[idx, 'Exceptional Event'] = 1
        
        # Convert to float
        cols_to_float = ['Sales', 'CP', 'CP(t-1)', 'CP(t-2)', 'DA', 'DA(t-1)', 'DA(t-2)', 'SeasIndx', 'Exceptional Event']
        df[cols_to_float] = df[cols_to_float].astype(float)
        
        # Apply log transformations
        log_cols = ['DA', 'DA(t-1)', 'DA(t-2)', 'CP', 'CP(t-1)', 'CP(t-2)']
        for col in log_cols:
            df[col] = np.log1p(df[col])
        
        # Create lag features
        for lag in [12]:
            df[f"lag_{lag}"] = df["SeasIndx"].shift(lag)
        
        # Add January 1988 row for prediction
        jan_88_row = {
            'CP': 100000,
            'CP(t-1)': df['CP'].iloc[-1],
            'CP(t-2)': df['CP(t-1)'].iloc[-1],
            'DA': 500000,
            'DA(t-1)': df['DA'].iloc[-1],
            'DA(t-2)': df['DA(t-1)'].iloc[-1],
            'SeasIndx': np.nan,  # Will be predicted
            'lag_12': df['SeasIndx'].iloc[-12],
            'Exceptional Event': 0,
            'Sales': np.nan
        }
        
        # Append January 1988 row
        df = pd.concat([df, pd.DataFrame([jan_88_row])], ignore_index=True)
        
        self.df = df
        return df
    
    def setup_model_config(self) -> ModelConfig:
        """Setup model configuration"""
        config = ModelConfig(
            sales_v1_cols=['CP(t-1)', 'DA(t-1)', 'SeasIndx'],
            sales_v2_cols=['CP', 'DA', 'SeasIndx'],
            cp_cols=['CP(t-1)'],
            da_cols=['DA(t-1)'],
            si_cols=['lag_12']
        )
        self.config = config
        return config
    
    def initialize_results(self) -> ForecastResults:
        """Initialize results container"""
        results = ForecastResults(
            sales_pred_v1=[], sales_pred_v2=[], sales_pred_v2_official=[], sales_true=[],
            cp_pred=[], da_pred=[], si_pred=[],
            cp_true=[], da_true=[], si_true=[],
            r2_values={'cp': [], 'da': [], 'si': [], 'sales_v1': [], 'sales_v2': []},
            adj_r2_values={'cp': [], 'da': [], 'si': [], 'sales_v1': [], 'sales_v2': []},
            jan88_predictions={'v1': [], 'v2': [], 'v2_official': []},
            cumulative_rmse={'v1': [], 'v2': []},
            cumulative_mape={'v1': [], 'v2': []},
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
        
        # 1. Predict SeasIndx
        # X_si_train = sm.add_constant(train[self.config.si_cols], has_constant='add')
        X_si_train = train[self.config.si_cols].copy().astype(float)
        si_model = sm.OLS(train[self.config.si_target], X_si_train).fit()
        models['si'] = si_model
        
        # X_si_test = sm.add_constant(test[self.config.si_cols], has_constant='add')
        X_si_test = test[self.config.si_cols].copy().astype(float)
        si_pred = si_model.predict(X_si_test).iloc[0]
        predictions['si'] = si_pred
        
        # 2. Predict CP (using predicted SI)
        X_cp_train = sm.add_constant(train[self.config.cp_cols], has_constant='add')
        cp_model = sm.OLS(train[self.config.cp_target], X_cp_train).fit()
        models['cp'] = cp_model
        
        X_cp_test = sm.add_constant(test[self.config.cp_cols], has_constant='add')
        # X_cp_test.loc[:, 'SeasIndx'] = si_pred
        cp_pred = cp_model.predict(X_cp_test).iloc[0]
        predictions['cp'] = cp_pred
        
        # 3. Predict DA
        X_da_train = sm.add_constant(train[self.config.da_cols], has_constant='add')
        da_model = sm.OLS(train[self.config.da_target], X_da_train).fit()
        models['da'] = da_model
        
        X_da_test = sm.add_constant(test[self.config.da_cols], has_constant='add')
        # X_da_test.loc[:, 'Exceptional Event'] = test['Exceptional Event'].values[0]
        da_pred = da_model.predict(X_da_test).iloc[0]
        predictions['da'] = da_pred
        
        # 4. Sales Model V1 (lag-only)
        X1_train = sm.add_constant(train[self.config.sales_v1_cols], has_constant='add')
        y1_train = train[self.config.sales_target]
        sales_model_v1 = sm.OLS(y1_train, X1_train).fit()
        models['sales_v1'] = sales_model_v1
        
        X1_test = sm.add_constant(test[self.config.sales_v1_cols], has_constant='add')
        X1_test.loc[:, 'SeasIndx'] = si_pred
        y1_pred = sales_model_v1.predict(X1_test).iloc[0]
        predictions['sales_v1'] = y1_pred
        
        # 5. Sales Model V2 (full)
        X2_train = sm.add_constant(train[self.config.sales_v2_cols], has_constant='add')
        y2_train = train[self.config.sales_target]
        sales_model_v2 = sm.OLS(y2_train, X2_train).fit()
        models['sales_v2'] = sales_model_v2
        
        X2_test = test[self.config.sales_v2_cols].copy().astype(float)
        X2_test.loc[:, 'CP'] = cp_pred
        X2_test.loc[:, 'DA'] = da_pred
        X2_test.loc[:, 'SeasIndx'] = si_pred
        X2_test = sm.add_constant(X2_test, has_constant='add')
        y2_pred = sales_model_v2.predict(X2_test).iloc[0]
        predictions['sales_v2'] = y2_pred
        
        # 6. January 1988 predictions with this iteration's models
        jan88_predictions = self.predict_jan88(models, jan88_row)
        predictions.update(jan88_predictions)
        
        return models, predictions
    
    def predict_jan88(self, models: Dict, jan88_row: pd.DataFrame) -> Dict:
        """Predict January 1988 values using current models"""
        jan88_row = jan88_row.copy()
        
        # Predict SeasIndx for Jan 1988
        # X_si_jan88 = sm.add_constant(jan88_row[self.config.si_cols], has_constant='add')
        X_si_jan88 = jan88_row[self.config.si_cols].copy().astype(float)
        si_pred_jan88 = models['si'].predict(X_si_jan88).iloc[0]
        jan88_row.loc[:, 'SeasIndx'] = si_pred_jan88
        
        # Predict CP for Jan 1988
        X_cp_jan88 = sm.add_constant(jan88_row[self.config.cp_cols], has_constant='add')
        cp_pred_jan88 = models['cp'].predict(X_cp_jan88).iloc[0]
        jan88_row.loc[:, 'CP'] = cp_pred_jan88
        
        # Predict DA for Jan 1988
        X_da_jan88 = sm.add_constant(jan88_row[self.config.da_cols], has_constant='add')
        da_pred_jan88 = models['da'].predict(X_da_jan88).iloc[0]
        jan88_row.loc[:, 'DA'] = da_pred_jan88
        
        # Predict Sales V1 for Jan 1988
        X_sales_v1_jan88 = sm.add_constant(jan88_row[self.config.sales_v1_cols], has_constant='add')
        sales_pred_v1_jan88 = models['sales_v1'].predict(X_sales_v1_jan88).iloc[0]
        
        # Predict Sales V2 for Jan 1988 (using predicted CP, DA, SI)
        X_sales_v2_jan88 = jan88_row[self.config.sales_v2_cols].copy()
        X_sales_v2_jan88['CP'] = cp_pred_jan88
        X_sales_v2_jan88['DA'] = da_pred_jan88
        X_sales_v2_jan88['SeasIndx'] = si_pred_jan88
        X_sales_v2_jan88 = sm.add_constant(X_sales_v2_jan88, has_constant='add')
        sales_pred_v2_jan88 = models['sales_v2'].predict(X_sales_v2_jan88).iloc[0]

        # Predict Sales V2 for Jan 1988 using official CP, DA (from last available data), and predicted SI
        X_sales_v2_jan88_official = jan88_row[self.config.sales_v2_cols].copy()
        X_sales_v2_jan88_official['SeasIndx'] = si_pred_jan88
        X_sales_v2_jan88_official = sm.add_constant(X_sales_v2_jan88_official, has_constant='add')
        sales_pred_v2_jan88_official = models['sales_v2'].predict(X_sales_v2_jan88_official).iloc[0]
        
        return {
            'jan88_si': si_pred_jan88,
            'jan88_cp': cp_pred_jan88,
            'jan88_da': da_pred_jan88,
            'jan88_sales_v1': sales_pred_v1_jan88,
            'jan88_sales_v2': sales_pred_v2_jan88,
            'jan88_sales_v2_official': sales_pred_v2_jan88_official
        }
    
    def calculate_cumulative_metrics(self) -> None:
        """Calculate cumulative RMSE and MAPE"""
        if len(self.results.sales_true) > 0:
            sales_true_arr = np.array(self.results.sales_true)
            sales_pred_v1_arr = np.array(self.results.sales_pred_v1)
            sales_pred_v2_arr = np.array(self.results.sales_pred_v2)
            
            # RMSE
            rmse_v1 = np.sqrt(np.mean((sales_true_arr - sales_pred_v1_arr)**2))
            rmse_v2 = np.sqrt(np.mean((sales_true_arr - sales_pred_v2_arr)**2))
            
            # MAPE with zero protection
            mask = sales_true_arr != 0
            if np.any(mask):
                mape_v1 = np.mean(np.abs((sales_true_arr[mask] - sales_pred_v1_arr[mask]) / sales_true_arr[mask])) * 100
                mape_v2 = np.mean(np.abs((sales_true_arr[mask] - sales_pred_v2_arr[mask]) / sales_true_arr[mask])) * 100
            else:
                mape_v1 = mape_v2 = 0.0
            
            self.results.cumulative_rmse['v1'].append(rmse_v1)
            self.results.cumulative_rmse['v2'].append(rmse_v2)
            self.results.cumulative_mape['v1'].append(mape_v1)
            self.results.cumulative_mape['v2'].append(mape_v2)
    
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
            self.results.sales_pred_v2.append(predictions['sales_v2'])
            self.results.sales_pred_v2_official.append(predictions['jan88_sales_v2_official'])
            self.results.sales_true.append(test_row[self.config.sales_target])
            
            self.results.cp_pred.append(predictions['cp'])
            self.results.da_pred.append(predictions['da'])
            self.results.si_pred.append(predictions['si'])
            self.results.cp_true.append(test_row[self.config.cp_target])
            self.results.da_true.append(test_row[self.config.da_target])
            self.results.si_true.append(test_row[self.config.si_target])
            
            # Store R² and Adjusted R² values
            self.results.r2_values['cp'].append(models['cp'].rsquared)
            self.results.r2_values['da'].append(models['da'].rsquared)
            self.results.r2_values['si'].append(models['si'].rsquared)
            self.results.r2_values['sales_v1'].append(models['sales_v1'].rsquared)
            self.results.r2_values['sales_v2'].append(models['sales_v2'].rsquared)
            
            self.results.adj_r2_values['cp'].append(models['cp'].rsquared_adj)
            self.results.adj_r2_values['da'].append(models['da'].rsquared_adj)
            self.results.adj_r2_values['si'].append(models['si'].rsquared_adj)
            self.results.adj_r2_values['sales_v1'].append(models['sales_v1'].rsquared_adj)
            self.results.adj_r2_values['sales_v2'].append(models['sales_v2'].rsquared_adj)
            
            # Store January 1988 predictions
            self.results.jan88_predictions['v1'].append(predictions['jan88_sales_v1'])
            self.results.jan88_predictions['v2'].append(predictions['jan88_sales_v2'])
            self.results.jan88_predictions['v2_official'].append(predictions['jan88_sales_v2_official'])
            
            # Update cumulative metrics
            self.calculate_cumulative_metrics()
            
            # Store final models for detailed statistics (last iteration)
            final_models = models
        
        # Extract detailed statistics from final models
        if final_models:
            self.results.final_model_stats = {
                'cp': self.extract_model_statistics(final_models['cp'], 'CP Model'),
                'da': self.extract_model_statistics(final_models['da'], 'DA Model'),
                'si': self.extract_model_statistics(final_models['si'], 'SeasIndx Model'),
                'sales_v1': self.extract_model_statistics(final_models['sales_v1'], 'Sales V1 Model'),
                'sales_v2': self.extract_model_statistics(final_models['sales_v2'], 'Sales V2 Model')
            }
        
        return self.results
    
    def calculate_ensemble_predictions(self) -> Dict[str, float]:
        """Calculate ensemble predictions for January 1988"""
        jan88_v1 = np.array(self.results.jan88_predictions['v1'])
        jan88_v2 = np.array(self.results.jan88_predictions['v2'])
        jan88_v2_official = np.array(self.results.jan88_predictions['v2_official'])
        rmse_v1 = np.array(self.results.cumulative_rmse['v1'])
        rmse_v2 = np.array(self.results.cumulative_rmse['v2'])
        
        # Simple mean ensemble
        simple_mean_v1 = np.mean(jan88_v1)
        simple_mean_v2 = np.mean(jan88_v2)
        simple_mean_v2_official = np.mean(jan88_v2_official)
        
        # Weighted mean ensemble (inverse RMSE)
        # Weighted ensemble for each model separately (inverse RMSE)
        
        # For V1
        weights_v1 = 1 / rmse_v1
        weighted_ensemble_v1 = np.sum(weights_v1 * jan88_v1) / np.sum(weights_v1)
        
        # For V2
        weights_v2 = 1 / rmse_v2
        weighted_ensemble_v2 = np.sum(weights_v2 * jan88_v2) / np.sum(weights_v2)
        
        return {
            'simple_mean_v1': simple_mean_v1,
            'simple_mean_v2': simple_mean_v2,
            'simple_mean_v2_official': simple_mean_v2_official,
            'weighted_ensemble_v1': weighted_ensemble_v1,
            'weighted_ensemble_v2': weighted_ensemble_v2,
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
        print("\n" + "="*50)
        print("HARMON FOODS FORECASTING RESULTS")
        print("="*50)
        
        # Convert to arrays for calculations
        sales_true = np.array(self.results.sales_true)
        sales_pred_v1 = np.array(self.results.sales_pred_v1)
        sales_pred_v2 = np.array(self.results.sales_pred_v2)
        sales_pred_v2_official = np.array(self.results.sales_pred_v2_official)
        
        # Calculate final metrics
        rmse_v1 = np.sqrt(np.mean((sales_true - sales_pred_v1)**2))
        rmse_v2 = np.sqrt(np.mean((sales_true - sales_pred_v2)**2))
        rmse_v2_official = np.sqrt(np.mean((sales_true - sales_pred_v2_official)**2))

        def mape(y_true, y_pred):
            mask = y_true != 0
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else 0.0
        
        mape_v1 = mape(sales_true, sales_pred_v1)
        mape_v2 = mape(sales_true, sales_pred_v2)
        mape_v2_official = mape(sales_true, sales_pred_v2_official)

        print(f"\nSales Model Performance:")
        print(f"Model V1 (Lag-only)  - RMSE: {rmse_v1:,.2f}, MAPE: {mape_v1:.2f}%")
        print(f"Model V2 (Full)      - RMSE: {rmse_v2:,.2f}, MAPE: {mape_v2:.2f}%")
        print(f"Model V2 (Official)  - RMSE: {rmse_v2_official:,.2f}, MAPE: {mape_v2_official:.2f}%")

        # Print R² and Adjusted R² statistics
        print(f"\n{'='*60}")
        print("R² AND ADJUSTED R² STATISTICS ACROSS ITERATIONS")
        print(f"{'='*60}")
        
        for model_name in ['cp', 'da', 'si', 'sales_v1', 'sales_v2']:
            r2_values = np.array(self.results.r2_values[model_name])
            adj_r2_values = np.array(self.results.adj_r2_values[model_name])
            
            print(f"\n{model_name.upper()} Model:")
            print(f"  R² - Mean: {np.mean(r2_values):.4f}, Std: {np.std(r2_values):.4f}, "
                  f"Min: {np.min(r2_values):.4f}, Max: {np.max(r2_values):.4f}")
            print(f"  Adj R² - Mean: {np.mean(adj_r2_values):.4f}, Std: {np.std(adj_r2_values):.4f}, "
                  f"Min: {np.min(adj_r2_values):.4f}, Max: {np.max(adj_r2_values):.4f}")
        
        print(f"\nJanuary 1988 Ensemble Predictions:")
        print(f"Simple Mean V1:      {ensemble_results['simple_mean_v1']:,.2f}")
        print(f"Simple Mean V2:      {ensemble_results['simple_mean_v2']:,.2f}")
        print(f"Simple Mean V2 Official: {ensemble_results['simple_mean_v2_official']:,.2f}")
        
        print(f"\nEnsemble Weights (Inverse RMSE):")
        print(f"V1 Weighted Ensemble: {ensemble_results['weighted_ensemble_v1']:,.2f}")
        print(f"V2 Weighted Ensemble: {ensemble_results['weighted_ensemble_v2']:,.2f}")
        
        # Print detailed coefficient statistics for final models
        if self.results.final_model_stats:
            print(f"\n{'='*80}")
            print("FINAL MODEL DETAILED STATISTICS (Last Iteration)")
            print(f"{'='*80}")
            
            for model_key, model_stats in self.results.final_model_stats.items():
                self.print_coefficient_stats(model_stats, model_stats['model_name'])
    
    def create_visualizations(self, ensemble_results: Dict) -> None:
        """Create comprehensive visualizations"""
        # Create two separate figure sets
        
        # Figure 1: Main Analysis
        fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig1.suptitle('Harmon Foods Forecasting Analysis', fontsize=16, fontweight='bold')
        
        iterations = list(range(len(self.results.jan88_predictions['v1'])))
        
        # Plot 1: January 1988 Predictions Evolution
        ax1.plot(iterations, self.results.jan88_predictions['v1'], 'purple', linewidth=2, label='Model V1', marker='o', markersize=3)
        ax1.plot(iterations, self.results.jan88_predictions['v2'], 'orange', linewidth=2, label='Model V2', marker='s', markersize=3)
        ax1.plot(iterations, self.results.jan88_predictions['v2_official'], 'orange', linewidth=2, label='Model V2 official', marker='x', markersize=3)
        ax1.set_title('January 1988 Sales Predictions Evolution')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Predicted Sales')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative RMSE Evolution
        ax2.plot(iterations, self.results.cumulative_rmse['v1'], 'purple', linewidth=2, label='Model V1', marker='o', markersize=3)
        ax2.plot(iterations, self.results.cumulative_rmse['v2'], 'orange', linewidth=2, label='Model V2', marker='s', markersize=3)
        ax2.set_title('Cumulative RMSE Evolution')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Cumulative RMSE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: R² Evolution for Sales Models
        ax3.plot(iterations, self.results.r2_values['sales_v1'], 'purple', linewidth=2, label='Sales V1 R²', marker='o', markersize=3)
        ax3.plot(iterations, self.results.r2_values['sales_v2'], 'orange', linewidth=2, label='Sales V2 R²', marker='s', markersize=3)
        ax3.plot(iterations, self.results.adj_r2_values['sales_v1'], 'purple', linewidth=2, label='Sales V1 Adj R²', marker='d', markersize=3, alpha=0.7, linestyle='--')
        ax3.plot(iterations, self.results.adj_r2_values['sales_v2'], 'orange', linewidth=2, label='Sales V2 Adj R²', marker='^', markersize=3, alpha=0.7, linestyle='--')
        ax3.set_title('R² and Adjusted R² Evolution - Sales Models')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('R² / Adjusted R²')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Ensemble Weights Evolution
        weights_v1 = []
        weights_v2 = []
        
        for i in range(len(self.results.cumulative_rmse['v1'])):
            rmse_v1 = self.results.cumulative_rmse['v1'][i] if self.results.cumulative_rmse['v1'][i] > 0 else 1e-8
            rmse_v2 = self.results.cumulative_rmse['v2'][i] if self.results.cumulative_rmse['v2'][i] > 0 else 1e-8
            
            w1 = (1/rmse_v1) / ((1/rmse_v1) + (1/rmse_v2))
            w2 = (1/rmse_v2) / ((1/rmse_v1) + (1/rmse_v2))
            
            weights_v1.append(w1)
            weights_v2.append(w2)
        
        ax4.plot(iterations, weights_v1, 'purple', linewidth=2, label='V1 Weight', marker='o', markersize=3)
        ax4.plot(iterations, weights_v2, 'orange', linewidth=2, label='V2 Weight', marker='s', markersize=3)
        ax4.set_title('Ensemble Weights Evolution')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Weight')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('outputs/harmon_foods_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\nMain analysis visualization saved as 'outputs/harmon_foods_analysis.png'")
        plt.show()
        
        # Figure 2: Model Performance Analysis
        fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig2.suptitle('Model Performance Analysis - R² Statistics', fontsize=16, fontweight='bold')
        
        # Plot 1: R² Evolution for All Models
        ax1.plot(iterations, self.results.r2_values['cp'], 'blue', linewidth=2, label='CP Model', marker='o', markersize=2)
        ax1.plot(iterations, self.results.r2_values['da'], 'red', linewidth=2, label='DA Model', marker='s', markersize=2)
        ax1.plot(iterations, self.results.r2_values['si'], 'green', linewidth=2, label='SI Model', marker='^', markersize=2)
        ax1.set_title('R² Evolution - Predictor Models')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('R²')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Adjusted R² Evolution for All Models
        ax2.plot(iterations, self.results.adj_r2_values['cp'], 'blue', linewidth=2, label='CP Model', marker='o', markersize=2)
        ax2.plot(iterations, self.results.adj_r2_values['da'], 'red', linewidth=2, label='DA Model', marker='s', markersize=2)
        ax2.plot(iterations, self.results.adj_r2_values['si'], 'green', linewidth=2, label='SI Model', marker='^', markersize=2)
        ax2.set_title('Adjusted R² Evolution - Predictor Models')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Adjusted R²')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: R² vs Adjusted R² Comparison (Final Values)
        models = ['CP', 'DA', 'SI', 'Sales V1', 'Sales V2']
        final_r2 = [
            self.results.r2_values['cp'][-1],
            self.results.r2_values['da'][-1],
            self.results.r2_values['si'][-1],
            self.results.r2_values['sales_v1'][-1],
            self.results.r2_values['sales_v2'][-1]
        ]
        final_adj_r2 = [
            self.results.adj_r2_values['cp'][-1],
            self.results.adj_r2_values['da'][-1],
            self.results.adj_r2_values['si'][-1],
            self.results.adj_r2_values['sales_v1'][-1],
            self.results.adj_r2_values['sales_v2'][-1]
        ]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax3.bar(x - width/2, final_r2, width, label='R²', alpha=0.7, color='skyblue')
        ax3.bar(x + width/2, final_adj_r2, width, label='Adjusted R²', alpha=0.7, color='lightcoral')
        ax3.set_title('Final R² vs Adjusted R² Comparison')
        ax3.set_ylabel('R² Value')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Model Fit Quality Distribution
        all_r2_values = []
        all_adj_r2_values = []
        model_labels = []
        
        for model_name in ['cp', 'da', 'si', 'sales_v1', 'sales_v2']:
            all_r2_values.extend(self.results.r2_values[model_name])
            all_adj_r2_values.extend(self.results.adj_r2_values[model_name])
            model_labels.extend([model_name.upper()] * len(self.results.r2_values[model_name]))
        
        ax4.scatter(all_r2_values, all_adj_r2_values, c=range(len(all_r2_values)), cmap='viridis', alpha=0.6)
        ax4.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='R² = Adj R²')
        ax4.set_xlabel('R²')
        ax4.set_ylabel('Adjusted R²')
        ax4.set_title('R² vs Adjusted R² Scatter Plot')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/model_performance_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Model performance visualization saved as 'outputs/model_performance_analysis.png'")
        plt.show()

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
    
    # Print results
    forecaster.print_results(ensemble_results)
    
    # Create visualizations
    forecaster.create_visualizations(ensemble_results)
    
    return forecaster, ensemble_results

if __name__ == "__main__":
    forecaster, ensemble_results = main()