import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

def plot_comprehensive_results():
    """Generate comprehensive comparison plots"""
    
    # prefer project-related path: experiment/analysis/results/
    candidates = [
        os.path.join('analysis', 'results', 'optimized_enhanced_metrics.csv'),
    ]
    metrics_file = None
    for c in candidates:
        if os.path.exists(c):
            metrics_file = c
            print(f"Using metrics file: {c}")
            break

    if not metrics_file:
        print(f"Metrics file not found in any known location: {candidates}")
        return

    results_dir = os.path.dirname(metrics_file)
    
    # Load data
    df = pd.read_csv(metrics_file)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SDN Routing Algorithm Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. Throughput vs Offered Load
    ax1 = axes[0, 0]
    throughput_data = df.groupby(['algorithm', 'offered_load_mbps'])['throughput_mbps'].agg(['mean', 'std']).reset_index()
    for algo in df['algorithm'].unique():
        algo_data = throughput_data[throughput_data['algorithm'] == algo]
        ax1.errorbar(algo_data['offered_load_mbps'], algo_data['mean'], 
                    yerr=algo_data['std'], label=algo.capitalize(), marker='o', linewidth=2)
    ax1.set_xlabel('Offered Load (Mbps)')
    ax1.set_ylabel('Throughput (Mbps)')
    ax1.set_title('Throughput Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Recovery Time Comparison
    ax2 = axes[0, 1]
    recovery_data = df.groupby(['algorithm', 'offered_load_mbps'])['recovery_time_s'].agg(['mean', 'std']).reset_index()
    for algo in df['algorithm'].unique():
        algo_data = recovery_data[recovery_data['algorithm'] == algo]
        ax2.errorbar(algo_data['offered_load_mbps'], algo_data['mean'], 
                    yerr=algo_data['std'], label=algo.capitalize(), marker='s', linewidth=2)
    ax2.set_xlabel('Offered Load (Mbps)')
    ax2.set_ylabel('Recovery Time (s)')
    ax2.set_title('Link Failure Recovery Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Jitter Comparison
    ax3 = axes[0, 2]
    jitter_data = df.groupby(['algorithm', 'offered_load_mbps'])['jitter_ms'].agg(['mean', 'std']).reset_index()
    for algo in df['algorithm'].unique():
        algo_data = jitter_data[jitter_data['algorithm'] == algo]
        ax3.errorbar(algo_data['offered_load_mbps'], algo_data['mean'], 
                    yerr=algo_data['std'], label=algo.capitalize(), marker='^', linewidth=2)
    ax3.set_xlabel('Offered Load (Mbps)')
    ax3.set_ylabel('Jitter (ms)')
    ax3.set_title('Network Jitter')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Packet Loss
    ax4 = axes[1, 0]
    loss_data = df.groupby(['algorithm', 'offered_load_mbps'])['packet_loss_percent'].agg(['mean', 'std']).reset_index()
    for algo in df['algorithm'].unique():
        algo_data = loss_data[loss_data['algorithm'] == algo]
        ax4.errorbar(algo_data['offered_load_mbps'], algo_data['mean'], 
                    yerr=algo_data['std'], label=algo.capitalize(), marker='d', linewidth=2)
    ax4.set_xlabel('Offered Load (Mbps)')
    ax4.set_ylabel('Packet Loss (%)')
    ax4.set_title('Packet Loss Rate')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. End-to-End Delay
    ax5 = axes[1, 1]
    delay_data = df.groupby(['algorithm', 'offered_load_mbps'])['end_to_end_delay_ms'].agg(['mean', 'std']).reset_index()
    for algo in df['algorithm'].unique():
        algo_data = delay_data[delay_data['algorithm'] == algo]
        ax5.errorbar(algo_data['offered_load_mbps'], algo_data['mean'], 
                    yerr=algo_data['std'], label=algo.capitalize(), marker='v', linewidth=2)
    ax5.set_xlabel('Offered Load (Mbps)')
    ax5.set_ylabel('End-to-End Delay (ms)')
    ax5.set_title('End-to-End Delay')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Flow Installation Efficiency
    ax6 = axes[1, 2]
    flow_data = df.groupby(['algorithm', 'offered_load_mbps'])['total_flows'].agg(['mean', 'std']).reset_index()
    for algo in df['algorithm'].unique():
        algo_data = flow_data[flow_data['algorithm'] == algo]
        ax6.errorbar(algo_data['offered_load_mbps'], algo_data['mean'], 
                    yerr=algo_data['std'], label=algo.capitalize(), marker='*', linewidth=2)
    ax6.set_xlabel('Offered Load (Mbps)')
    ax6.set_ylabel('Total Flow Rules')
    ax6.set_title('Flow Installation Efficiency')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'comprehensive_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(results_dir, 'comprehensive_comparison.pdf'), bbox_inches='tight')
    
    # Generate summary statistics table
    generate_summary_table(df, results_dir)
    
    print("Comprehensive analysis complete!")
    print(f"Plots saved to: {results_dir}/comprehensive_comparison.png")

def generate_summary_table(df, results_dir):
    """Generate summary statistics table"""
    summary = df.groupby('algorithm').agg({
        'throughput_mbps': ['mean', 'std'],
        'recovery_time_s': ['mean', 'std'],
        'jitter_ms': ['mean', 'std'],
        'packet_loss_percent': ['mean', 'std'],
        'end_to_end_delay_ms': ['mean', 'std'],
        'total_flows': ['mean', 'std']
    }).round(3)
    
    summary_file = os.path.join(results_dir, 'summary_statistics.csv')
    summary.to_csv(summary_file)
    print(f"Summary statistics saved to: {summary_file}")

if __name__ == '__main__':
    plot_comprehensive_results()