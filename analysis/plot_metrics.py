import matplotlib.pyplot as plt
import pandas as pd
import os
import re

RESULTS_DIR = './results'
METRICS_CSV = os.path.join(RESULTS_DIR, 'metrics.csv')

def parse_ryu_log_for_flow_installs():
    """Parses the Ryu log to extract flow installation timestamps."""
    log_path = os.path.join(RESULTS_DIR, 'ryu.log')
    flow_install_times = []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if "Flow installed" in line:
                    match = re.search(r'time=([\d.]+)', line)
                    if match:
                        flow_install_times.append(float(match.group(1)))
    except FileNotFoundError:
        print(f"Log file not found: {log_path}")
    return flow_install_times

def parse_iperf_logs():
    """Parses iperf server logs to extract throughput, jitter, and loss."""
    results = {}
    for filename in os.listdir(RESULTS_DIR):
        if filename.startswith("iperf_server"):
            rate_match = re.search(r'rate(\w+)\.txt', filename)
            rate = rate_match.group(1) if rate_match else "unknown"
            
            if rate not in results:
                results[rate] = {'throughput': [], 'jitter': [], 'loss': []}

            path = os.path.join(RESULTS_DIR, filename)
            with open(path, 'r') as f:
                for line in f:
                    # Example iperf output: [  3]  0.0-1.0 sec  11.2 MBytes  94.1 Mbits/sec   0.015 ms    0/ 1372 (0%)
                    match = re.search(r'(\d+\.?\d*)\s+Mbits/sec\s+(\d+\.?\d*)\s+ms\s+\d+/ \d+\s+\((\d+\.?\d*)%\)', line)
                    if match:
                        results[rate]['throughput'].append(float(match.group(1)))
                        results[rate]['jitter'].append(float(match.group(2)))
                        results[rate]['loss'].append(float(match.group(3)))
    return results

def plot_recovery_times():
    """Plots link down to recovery times."""
    try:
        df = pd.read_csv(
            os.path.join(RESULTS_DIR, 'recovery_times.txt'), 
            delim_whitespace=True, 
            header=None,
            names=['RunLabel', 'Run', 'RateLabel', 'Rate', 'RecoveryLabel', 'RecoveryTime']
        )
        df_agg = df.groupby('Rate')['RecoveryTime'].mean().reset_index()
        
        plt.figure()
        plt.bar(df_agg['Rate'], df_agg['RecoveryTime'])
        plt.title('Average Link Failure Recovery Time vs. Traffic Rate')
        plt.xlabel('Traffic Rate (Mbps)')
        plt.ylabel('Average Recovery Time (s)')
        plt.savefig(os.path.join(RESULTS_DIR, 'recovery_time.png'))
        plt.close()
        print("Generated recovery_time.png")
    except FileNotFoundError:
        print("recovery_times.txt not found.")


def plot_jitter(iperf_data):
    """Plots average jitter for different traffic rates."""
    if not iperf_data:
        print("No iperf data to plot.")
        return
    
    avg_jitters = {rate: sum(vals['jitter'])/len(vals['jitter']) for rate, vals in iperf_data.items() if vals['jitter']}
    
    plt.figure()
    plt.bar(avg_jitters.keys(), avg_jitters.values(), color='orange')
    plt.title('Average Jitter vs. Offered Load')
    plt.xlabel('Offered Load (Mbps)')
    plt.ylabel('Average Jitter (ms)')
    plt.savefig(os.path.join(RESULTS_DIR, 'jitter.png'))
    plt.close()
    print("Generated jitter.png")

def plot_loss(iperf_data):
    """Plots average packet loss for different traffic rates."""
    if not iperf_data:
        print("No iperf data to plot.")
        return
    
    avg_loss = {rate: sum(vals['loss'])/len(vals['loss']) for rate, vals in iperf_data.items() if vals['loss']}
    
    plt.figure()
    plt.bar(avg_loss.keys(), avg_loss.values(), color='red')
    plt.title('Average Packet Loss vs. Offered Load')
    plt.xlabel('Offered Load (Mbps)')
    plt.ylabel('Average Packet Loss (%)')
    plt.savefig(os.path.join(RESULTS_DIR, 'loss.png'))
    plt.close()
    print("Generated loss.png")

def plot_throughput(iperf_data):
    """Plots throughput for different traffic rates."""
    if not iperf_data:
        print("No iperf data to plot.")
        return
        
    avg_throughputs = {rate: sum(vals['throughput'])/len(vals['throughput']) for rate, vals in iperf_data.items() if vals['throughput']}
    
    plt.figure()
    plt.bar(avg_throughputs.keys(), avg_throughputs.values())
    plt.title('Average Throughput vs. Offered Load')
    plt.xlabel('Offered Load (Mbps)')
    plt.ylabel('Average Throughput (Mbps)')
    plt.savefig(os.path.join(RESULTS_DIR, 'throughput.png'))
    plt.close()
    print("Generated throughput.png")


def plot_routing_metrics(metrics_file):
    # Load metrics from the CSV file
    if not os.path.exists(metrics_file):
        print(f"Metrics file {metrics_file} not found.")
        return

    metrics_data = pd.read_csv(metrics_file)

    # Plotting the metrics
    plt.figure(figsize=(10, 6))

    # Example: Plotting latency
    plt.subplot(2, 1, 1)
    plt.plot(metrics_data['time'], metrics_data['latency'], label='Latency', color='blue')
    plt.title('Routing Algorithm Latency')
    plt.xlabel('Time (s)')
    plt.ylabel('Latency (ms)')
    plt.legend()
    plt.grid()

    # Example: Plotting throughput
    plt.subplot(2, 1, 2)
    plt.plot(metrics_data['time'], metrics_data['throughput'], label='Throughput', color='green')
    plt.title('Routing Algorithm Throughput')
    plt.xlabel('Time (s)')
    plt.ylabel('Throughput (Mbps)')
    plt.legend()
    plt.grid()

    # Save the plots
    plt.tight_layout()
    plt.savefig('routing_metrics.png')
    plt.show()

def plot_comparison_metrics(metrics_csv=METRICS_CSV):
    if not os.path.exists(metrics_csv):
        print(f"Metrics CSV not found: {metrics_csv}")
        return
    df = pd.read_csv(metrics_csv)
    # aggregate mean and 95% CI
    agg = df.groupby(['algorithm','rate_mbps']).agg(
        recovery_mean=('recovery_time_s','mean'),
        recovery_std=('recovery_time_s','std'),
        throughput_mean=('throughput_mbps','mean'),
        throughput_std=('throughput_mbps','std'),
        flow_install_mean=('flow_install_time_s','mean')
    ).reset_index()

    # throughput plot
    plt.figure()
    for algo in agg['algorithm'].unique():
        d = agg[agg['algorithm']==algo]
        plt.errorbar(d['rate_mbps'], d['throughput_mean'], yerr=d['throughput_std'], label=algo, marker='o')
    plt.xlabel('Offered load (Mbps)')
    plt.ylabel('Throughput (Mbps)')
    plt.title('Throughput vs Offered Load')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(RESULTS_DIR,'throughput_algo_cmp.png'))
    plt.close()

    # recovery time plot
    plt.figure()
    for algo in agg['algorithm'].unique():
        d = agg[agg['algorithm']==algo]
        plt.errorbar(d['rate_mbps'], d['recovery_mean'], yerr=d['recovery_std'], label=algo, marker='o')
    plt.xlabel('Offered load (Mbps)')
    plt.ylabel('Recovery time (s)')
    plt.title('Link Failure Recovery Time vs Offered Load')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(RESULTS_DIR,'recovery_algo_cmp.png'))
    plt.close()

    print("Generated comparison plots in", RESULTS_DIR)

def main():
    metrics_file = 'path/to/your/metrics.csv'  # Update this path to your metrics file
    plot_routing_metrics(metrics_file)

if __name__ == "__main__":
    print("--- Parsing and Plotting Results ---")
    plot_recovery_times()
    
    iperf_data = parse_iperf_logs()
    plot_throughput(iperf_data)
    plot_jitter(iperf_data)
    plot_loss(iperf_data)
    
    # You can add more plotting functions here, e.g., for flow installation times
    flow_times = parse_ryu_log_for_flow_installs()
    print(f"Found {len(flow_times)} flow installation events.")
    print("--- Analysis Complete ---")