# SDN Routing Algorithm Comparison: Dijkstra vs Bellman-Ford

A comprehensive experimental study comparing the performance of Dijkstra's and Bellman-Ford's shortest path algorithms in Software-Defined Networks (SDN) under various network topologies, traffic conditions, and failure scenarios.

## Overview

This research addresses the gap in SDN literature regarding standardized comparisons of routing algorithms across different network topologies and failure recovery scenarios. The study evaluates whether algorithmic choice or network architecture is the dominant factor in SDN performance.

## Research Objectives

- Provide a standardized benchmark for SDN routing algorithm comparison
- Evaluate performance across common network topologies (Linear, Star, Mesh)
- Assess failure recovery capabilities under various stress conditions
- Establish baseline metrics for future SDN resilience research

## Key Performance Metrics

- **Throughput (Mbps)**: Actual data transfer rate achieved
- **End-to-End Delay (ms)**: Packet travel time from source to destination
- **Jitter (ms)**: Variation in packet delay
- **Packet Loss (%)**: Percentage of lost packets
- **Recovery Time (s)**: Time to restore connectivity after link failure

## Experimental Design

### Algorithms Tested
- Dijkstra's Shortest Path
- Bellman-Ford Shortest Path

### Network Topologies
1. **Linear**: Chain of 4 switches (tests sequential routing)
2. **Star**: Central switch with 3 edge switches (tests bottleneck scenarios)
3. **Mesh**: Partially connected 4-switch network (tests path diversity)

### Traffic Scenarios
- **Moderate Load**: 40M and 80M traffic rates
- **High Load**: 100M and 150M traffic rates

### Stress Conditions
- **Normal**: Baseline operation
- **Background Traffic**: Network congestion simulation
- **Cascade Failure**: Sequential link failures

### Total Experiments
**144 test runs**: 2 algorithms × 3 topologies × 2 traffic scenarios × 3 stress conditions × 2 runs × 2 rates

## Installation & Requirements

### Prerequisites
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3 python3-pip mininet openvswitch-switch

# Install Ryu SDN Controller
pip3 install ryu

# Install iperf for traffic generation
sudo apt-get install -y iperf
```

### Python Dependencies
```bash
pip3 install mininet-python
```

## Project Structure

```
Paper 1/
├── run_full_experiment.py          # Main experiment orchestrator
├── simple_working_spr.py           # Ryu SDN controller application
├── analysis/                       # Analysis scripts and results
│   ├── advanced_statistical_analysis.py
│   ├── normalize_and_analyze_results.py
│   ├── plot_comprehensive_metrics.py
│   └── plot_metrics.py

```

## Usage

### Running the Full Experiment

```bash
# Ensure you have root/sudo privileges (required for Mininet)
sudo python3 run_full_experiment.py
```

### Expected Runtime
- **Total Duration**: ~60-90 minutes for all 144 experiments
- **Per Test**: ~30-45 seconds average

### Output Files

All results are saved to `analysis/results/`:
- `optimized_enhanced_metrics.csv` - Raw experimental data
- `optimized_experiment_status.log` - Detailed execution log
- `iperf_*.txt` - Individual iperf measurement logs
- `controller_*.log` - SDN controller logs per configuration

## Analysis

### Generate Visualizations

```bash
cd analysis
python3 plot_comprehensive_metrics.py
python3 advanced_statistical_analysis.py
```

### Statistical Analysis

The analysis scripts provide:
- Comparative performance charts across topologies
- Recovery time distributions
- Statistical significance tests (t-tests, ANOVA)
- Correlation analysis between metrics

## Key Findings

1. **Algorithmic Impact**: Dijkstra and Bellman-Ford show nearly identical throughput and recovery times
2. **Delay Performance**: Dijkstra consistently achieves lower end-to-end delay
3. **Dominant Factors**: Network topology and control plane response time are more critical than algorithm choice
4. **Recovery Bottleneck**: Link failure detection and rule installation dominate recovery time, not path computation



## System Requirements

- **OS**: Linux (Ubuntu 18.04+ recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **CPU**: Multi-core processor recommended
- **Network**: No external connectivity required (uses virtual network)

## Troubleshooting

### Common Issues

**Mininet won't start**:
```bash
sudo mn -c  # Clean up previous Mininet instances
sudo service openvswitch-switch restart
```

**Controller connection fails**:
```bash
# Check if port 6653 is available
sudo netstat -tulpn | grep 6653
# Kill any process using the port
sudo kill -9 <PID>
```

**Permission errors**:
```bash
# Run with sudo
sudo python3 run_full_experiment.py
```
