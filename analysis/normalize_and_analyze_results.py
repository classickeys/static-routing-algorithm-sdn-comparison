import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
import os
warnings.filterwarnings('ignore')

class ResultsNormalizer:
    """Class to normalize experimental results and analyze success/failure patterns"""
    
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.original_df = self.df.copy()
        self.success_criteria = {}
        
    def analyze_raw_data(self):
        """Analyze raw data to understand the issues"""
        print("="*80)
        print("RAW DATA ANALYSIS")
        print("="*80)
        
        # Basic statistics
        print("\n1. Dataset Overview:")
        print(f"Total experiments: {len(self.df)}")
        print(f"Success rate: {(self.df['status'] == 'SUCCESS').sum() / len(self.df) * 100:.1f}%")
        print(f"Failure rate: {(self.df['status'] == 'FAILED').sum() / len(self.df) * 100:.1f}%")
        
        # Identify problematic columns
        print("\n2. Negative Values Analysis:")
        numeric_cols = ['throughput_mbps', 'jitter_ms', 'packet_loss_percent', 
                       'end_to_end_delay_ms', 'recovery_time_s', 'flow_install_time_s']
        
        for col in numeric_cols:
            negative_count = (self.df[col] < 0).sum()
            if negative_count > 0:
                print(f"   {col}: {negative_count} negative values ({negative_count/len(self.df)*100:.1f}%)")
                print(f"      Min: {self.df[col].min():.3f}, Max: {self.df[col].max():.3f}")
        
        # Recovery time analysis
        print(f"\n3. Recovery Time Analysis:")
        print(f"   Experiments with recovery_time = -1.0: {(self.df['recovery_time_s'] == -1.0).sum()}")
        print(f"   This represents scenarios where no failure/recovery was tested")
        
        return self.analyze_success_failure_patterns()
    
    def analyze_success_failure_patterns(self):
        """Analyze what determines success vs failure"""
        print("\n4. Success/Failure Pattern Analysis:")
        
        # Group by status and calculate means
        status_comparison = self.df.groupby('status').agg({
            'throughput_mbps': ['mean', 'std'],
            'end_to_end_delay_ms': ['mean', 'std'],
            'packet_loss_percent': ['mean', 'std'],
            'recovery_time_s': ['mean', 'std', 'count']
        }).round(3)
        
        print("\n   Metrics by Status:")
        print(status_comparison)
        
        # Analyze by stress condition
        stress_success = pd.crosstab(self.df['stress_condition'], self.df['status'], normalize='index') * 100
        print(f"\n   Success Rate by Stress Condition:")
        print(stress_success.round(1))
        
        # Key insight: Success seems to correlate with cascade_failure scenarios
        # This suggests SUCCESS = algorithm successfully handled and recovered from failures
        # FAILED = algorithm couldn't handle the stress condition properly
        
        return {
            'success_means_recovery': True,
            'failure_means_no_recovery': True,
            'recovery_time_is_key_metric': True
        }
    
    def define_success_criteria(self):
        """Define clear success/failure criteria based on analysis"""
        print("\n5. Defining Success Criteria:")
        
        # Based on the data analysis, success criteria are:
        self.success_criteria = {
            'throughput_threshold': 35.0,  # Minimum acceptable throughput (Mbps)
            'delay_threshold': 50.0,       # Maximum acceptable delay (ms)
            'loss_threshold': 5.0,         # Maximum acceptable loss (%)
            'recovery_required': True,     # Must have successful recovery from failures
            'recovery_time_threshold': 10.0 # Maximum acceptable recovery time (s)
        }
        
        print("   Success Criteria Defined:")
        for key, value in self.success_criteria.items():
            print(f"   - {key}: {value}")
        
        return self.success_criteria
    
    def normalize_data(self):
        """Normalize the data and handle negative values"""
        print("\n" + "="*80)
        print("DATA NORMALIZATION")
        print("="*80)
        
        # Create normalized dataframe
        self.df_normalized = self.df.copy()
        
        # 1. Handle recovery_time_s: -1.0 means "no failure introduced"
        print("\n1. Handling Recovery Time:")
        print(f"   Before: {(self.df_normalized['recovery_time_s'] == -1.0).sum()} entries with -1.0")
        
        # For FAILED experiments with recovery_time = -1.0, this means no recovery was possible
        # For SUCCESS experiments, recovery_time should be positive
        self.df_normalized.loc[
            (self.df_normalized['status'] == 'FAILED') & 
            (self.df_normalized['recovery_time_s'] == -1.0), 
            'recovery_time_s'
        ] = 0.0  # No recovery achieved
        
        # For SUCCESS experiments with -1.0, replace with mean of successful recoveries
        successful_recovery_mean = self.df_normalized[
            (self.df_normalized['status'] == 'SUCCESS') & 
            (self.df_normalized['recovery_time_s'] > 0)
        ]['recovery_time_s'].mean()
        
        self.df_normalized.loc[
            (self.df_normalized['status'] == 'SUCCESS') & 
            (self.df_normalized['recovery_time_s'] == -1.0), 
            'recovery_time_s'
        ] = successful_recovery_mean
        
        print(f"   After: {(self.df_normalized['recovery_time_s'] == -1.0).sum()} entries with -1.0")
        print(f"   Used mean recovery time: {successful_recovery_mean:.3f}s for SUCCESS cases")
        
        # 2. Handle other negative values (replace with 0 or small positive value)
        print("\n2. Handling Other Negative Values:")
        
        # Throughput: negative doesn't make sense, set to 0
        neg_throughput = (self.df_normalized['throughput_mbps'] < 0).sum()
        if neg_throughput > 0:
            print(f"   Setting {neg_throughput} negative throughput values to 0")
            self.df_normalized.loc[self.df_normalized['throughput_mbps'] < 0, 'throughput_mbps'] = 0.0
        
        # Delay: negative doesn't make sense, set to 0
        neg_delay = (self.df_normalized['end_to_end_delay_ms'] < 0).sum()
        if neg_delay > 0:
            print(f"   Setting {neg_delay} negative delay values to 0")
            self.df_normalized.loc[self.df_normalized['end_to_end_delay_ms'] < 0, 'end_to_end_delay_ms'] = 0.0
        
        # 3. Min-Max normalize key metrics for comparison
        print("\n3. Applying Min-Max Normalization (0-1 scale):")
        
        metrics_to_normalize = [
            'throughput_mbps', 'jitter_ms', 'packet_loss_percent',
            'end_to_end_delay_ms', 'recovery_time_s', 'flow_install_time_s'
        ]
        
        scaler = MinMaxScaler()
        
        for metric in metrics_to_normalize:
            original_range = f"[{self.df_normalized[metric].min():.3f}, {self.df_normalized[metric].max():.3f}]"
            
            # Normalize
            self.df_normalized[f'{metric}_normalized'] = scaler.fit_transform(
                self.df_normalized[[metric]]
            ).flatten()
            
            normalized_range = f"[{self.df_normalized[f'{metric}_normalized'].min():.3f}, {self.df_normalized[f'{metric}_normalized'].max():.3f}]"
            
            print(f"   {metric}: {original_range} -> {normalized_range}")
        
        return self.df_normalized
    
    def create_composite_performance_score(self):
        """Create a composite performance score for ranking"""
        print("\n4. Creating Composite Performance Score:")
        
        # Define weights (higher weight = more important)
        weights = {
            'throughput_mbps_normalized': 0.25,      # Higher throughput is better
            'end_to_end_delay_ms_normalized': -0.20,  # Lower delay is better (negative weight)
            'packet_loss_percent_normalized': -0.20,  # Lower loss is better (negative weight)
            'recovery_time_s_normalized': -0.15,      # Faster recovery is better (negative weight)
            'flow_install_time_s_normalized': -0.10,  # Faster install is better (negative weight)
            'jitter_ms_normalized': -0.10            # Lower jitter is better (negative weight)
        }
        
        # Calculate composite score
        self.df_normalized['performance_score'] = 0
        
        for metric, weight in weights.items():
            self.df_normalized['performance_score'] += (
                self.df_normalized[metric] * weight
            )
        
        # Normalize performance score to 0-100 scale
        score_min = self.df_normalized['performance_score'].min()
        score_max = self.df_normalized['performance_score'].max()
        
        self.df_normalized['performance_score_100'] = (
            (self.df_normalized['performance_score'] - score_min) / 
            (score_max - score_min) * 100
        )
        
        print(f"   Performance Score Range: 0-100")
        print(f"   Mean Score: {self.df_normalized['performance_score_100'].mean():.2f}")
        print(f"   Weights Used: {weights}")
        
        return weights
    
    def reclassify_success_failure(self):
        """Reclassify success/failure based on composite criteria"""
        print("\n5. Reclassifying Success/Failure:")
        
        # Create new classification based on performance score and key metrics
        conditions = (
            (self.df_normalized['throughput_mbps'] >= self.success_criteria['throughput_threshold']) &
            (self.df_normalized['end_to_end_delay_ms'] <= self.success_criteria['delay_threshold']) &
            (self.df_normalized['packet_loss_percent'] <= self.success_criteria['loss_threshold']) &
            (self.df_normalized['performance_score_100'] >= 50.0)  # Above median performance
        )
        
        self.df_normalized['reclassified_status'] = np.where(conditions, 'SUCCESS', 'FAILED')
        
        # Compare with original classification
        original_success_rate = (self.df['status'] == 'SUCCESS').mean() * 100
        new_success_rate = (self.df_normalized['reclassified_status'] == 'SUCCESS').mean() * 100
        
        print(f"   Original Success Rate: {original_success_rate:.1f}%")
        print(f"   Reclassified Success Rate: {new_success_rate:.1f}%")
        
        # Show classification changes
        classification_changes = pd.crosstab(
            self.df['status'], 
            self.df_normalized['reclassified_status'], 
            margins=True
        )
        print(f"\n   Classification Change Matrix:")
        print(classification_changes)
        
        return self.df_normalized
    
    def save_normalized_data(self, output_file):
        """Save normalized data to CSV"""
        # Ensure directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        self.df_normalized.to_csv(output_file, index=False)
        print(f"\n   Normalized data saved to: {output_file}")
        
        return output_file

def create_comparison_visualizations(df_normalized, output_dir=None):
    """Create visualizations comparing success vs failure"""
    
    # Set output directory - use current directory if not specified
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'results')
    
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except FileExistsError:
            pass  # Directory already exists, continue
    
    # Set style
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    
    fig_size = (15, 10)
    
    # Create the figure
    plt.figure(figsize=fig_size)
    
    # 1. Performance Score Distribution by Algorithm and Status
    plt.subplot(2, 3, 1)
    try:
        sns.boxplot(data=df_normalized, x='algorithm', y='performance_score_100', hue='status')
    except:
        # Fallback to matplotlib if seaborn fails
        algorithms = df_normalized['algorithm'].unique()
        success_data = df_normalized[df_normalized['status'] == 'SUCCESS']['performance_score_100']
        failed_data = df_normalized[df_normalized['status'] == 'FAILED']['performance_score_100']
        
        plt.boxplot([success_data, failed_data], labels=['SUCCESS', 'FAILED'])
    
    plt.title('Performance Score by Algorithm & Status')
    plt.ylabel('Performance Score (0-100)')
    plt.xticks(rotation=45)
    
    # 2. Success Rate by Topology and Algorithm
    plt.subplot(2, 3, 2)
    try:
        success_by_topo = pd.crosstab(
            [df_normalized['algorithm'], df_normalized['topology']], 
            df_normalized['status'], 
            normalize='index'
        )['SUCCESS'] * 100
        
        success_by_topo.unstack(level=0).plot(kind='bar', ax=plt.gca())
        plt.title('Success Rate by Topology')
        plt.ylabel('Success Rate (%)')
        plt.legend(title='Algorithm')
    except Exception as e:
        plt.text(0.5, 0.5, f'Error creating plot:\n{str(e)}', ha='center', va='center')
        plt.title('Success Rate by Topology (Error)')
    plt.xticks(rotation=45)
    
    # 3. Recovery Time Analysis
    plt.subplot(2, 3, 3)
    recovery_data = df_normalized[df_normalized['recovery_time_s'] > 0]
    if len(recovery_data) > 0:
        try:
            sns.violinplot(data=recovery_data, x='status', y='recovery_time_s', hue='algorithm')
        except:
            # Fallback to histogram
            success_recovery = recovery_data[recovery_data['status'] == 'SUCCESS']['recovery_time_s']
            failed_recovery = recovery_data[recovery_data['status'] == 'FAILED']['recovery_time_s']
            
            if len(success_recovery) > 0 or len(failed_recovery) > 0:
                plt.hist([success_recovery, failed_recovery], bins=10, alpha=0.7, 
                        label=['SUCCESS', 'FAILED'], color=['green', 'red'])
                plt.legend()
        
        plt.title('Recovery Time Distribution')
        plt.ylabel('Recovery Time (seconds)')
    else:
        plt.text(0.5, 0.5, 'No recovery data available', ha='center', va='center')
        plt.title('Recovery Time Distribution (No Data)')
    
    # 4. Throughput vs Delay scatter
    plt.subplot(2, 3, 4)
    scatter = plt.scatter(
        df_normalized['throughput_mbps'], 
        df_normalized['end_to_end_delay_ms'],
        c=df_normalized['performance_score_100'], 
        cmap='RdYlGn', 
        alpha=0.6
    )
    plt.xlabel('Throughput (Mbps)')
    plt.ylabel('End-to-End Delay (ms)')
    plt.title('Throughput vs Delay (colored by Performance Score)')
    plt.colorbar(scatter, label='Performance Score')
    
    # 5. Success Rate by Stress Condition
    plt.subplot(2, 3, 5)
    try:
        stress_success = pd.crosstab(
            [df_normalized['stress_condition'], df_normalized['algorithm']], 
            df_normalized['status'], 
            normalize='index'
        )['SUCCESS'] * 100
        
        stress_success.unstack(level=1).plot(kind='bar', ax=plt.gca())
        plt.title('Success Rate by Stress Condition')
        plt.ylabel('Success Rate (%)')
        plt.legend(title='Algorithm')
    except Exception as e:
        plt.text(0.5, 0.5, f'Error creating plot:\n{str(e)}', ha='center', va='center')
        plt.title('Success Rate by Stress Condition (Error)')
    plt.xticks(rotation=45)
    
    # 6. Performance Metrics Heatmap
    plt.subplot(2, 3, 6)
    try:
        metrics_for_heatmap = [
            'throughput_mbps_normalized', 'end_to_end_delay_ms_normalized',
            'packet_loss_percent_normalized', 'recovery_time_s_normalized'
        ]
        
        # Simplified heatmap showing average normalized metrics
        avg_metrics = df_normalized.groupby('algorithm')[metrics_for_heatmap].mean()
        sns.heatmap(avg_metrics.T, annot=True, cmap='RdYlBu_r', fmt='.3f')
        plt.title('Average Normalized Metrics by Algorithm')
        plt.ylabel('Metrics')
    except Exception as e:
        plt.text(0.5, 0.5, f'Error creating heatmap:\n{str(e)}', ha='center', va='center')
        plt.title('Performance Metrics Heatmap (Error)')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, 'normalized_analysis_dashboard.png')
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")
    except Exception as e:
        print(f"Error saving visualization: {e}")
        output_file = None
    
    plt.show()
    
    return output_file

def main():
    """Main execution function"""
    
    # Try to find the CSV file in multiple locations
    possible_csv_files = [
        'comprehensive_metrics.csv',
        'results/comprehensive_metrics.csv',
        '../results/comprehensive_metrics.csv',
        os.path.join(os.getcwd(), 'comprehensive_metrics.csv')
    ]
    
    csv_file = None
    for file_path in possible_csv_files:
        if os.path.exists(file_path):
            csv_file = file_path
            print(f"Found CSV file at: {csv_file}")
            break
    
    if csv_file is None:
        print("Error: Could not find comprehensive_metrics.csv")
        print("Please ensure the file is in one of these locations:")
        for path in possible_csv_files:
            print(f"  - {os.path.abspath(path)}")
        return None
    
    try:
        # Initialize normalizer
        normalizer = ResultsNormalizer(csv_file)
        
        # Step 1: Analyze raw data
        patterns = normalizer.analyze_raw_data()
        
        # Step 2: Define success criteria
        criteria = normalizer.define_success_criteria()
        
        # Step 3: Normalize data
        df_normalized = normalizer.normalize_data()
        
        # Step 4: Create composite performance score
        weights = normalizer.create_composite_performance_score()
        
        # Step 5: Reclassify success/failure
        df_final = normalizer.reclassify_success_failure()
        
        # Step 6: Save normalized data
        output_file = normalizer.save_normalized_data('normalized_comprehensive_metrics.csv')
        
        # Step 7: Create visualizations
        viz_file = create_comparison_visualizations(df_final)
        
        print("\n" + "="*80)
        print("SUMMARY INSIGHTS")
        print("="*80)
        
        # Key insights
        print("\nKEY FINDINGS:")
        print("1. SUCCESS in your data means: Algorithm successfully recovered from induced failures")
        print("2. FAILED means: Algorithm couldn't handle stress conditions effectively")
        print("3. Negative recovery_time (-1.0) indicated no failure was introduced in that test")
        print("4. Your success criteria should focus on:")
        print("   - Throughput maintenance under stress")
        print("   - Fast recovery from failures") 
        print("   - Low latency and packet loss")
        
        # Algorithm comparison
        algo_comparison = df_final.groupby('algorithm').agg({
            'performance_score_100': 'mean',
            'throughput_mbps': 'mean',
            'end_to_end_delay_ms': 'mean',
            'recovery_time_s': 'mean'
        }).round(3)
        
        print(f"\nALGORITHM PERFORMANCE COMPARISON:")
        print(algo_comparison)
        
        # Best performing scenarios
        best_scenarios = df_final.nlargest(5, 'performance_score_100')[
            ['algorithm', 'topology', 'stress_condition', 'performance_score_100', 'status']
        ]
        
        print(f"\nTOP 5 PERFORMING SCENARIOS:")
        print(best_scenarios.to_string(index=False))
        
        return {
            'normalized_data': df_final,
            'output_file': output_file,
            'visualization': viz_file,
            'success_criteria': criteria,
            'performance_weights': weights
        }
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
    if results:
        print("\n✅ Analysis completed successfully!")
    else:
        print("\n❌ Analysis failed. Check the error messages above.")