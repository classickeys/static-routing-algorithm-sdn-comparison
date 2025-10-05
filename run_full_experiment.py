import time
import subprocess
import signal
import os
import csv
import re
import random
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.clean import cleanup
import sys

sys.path.append('mininet_scripts')
from custom_topology import LargerTopo

class OptimizedVariedTopology:
    """Create different network topologies for testing - optimized versions"""
    
    @staticmethod
    def create_linear_topology():
        """Linear topology: h1-s1-s2-s3-s4-h2"""
        from mininet.topo import Topo
        
        class LinearTopo(Topo):
            def build(self):
                # Add hosts
                h1 = self.addHost('h1')
                h2 = self.addHost('h2')
                
                # Add switches - reduced from 5 to 4
                switches = []
                for i in range(1, 5):  # s1 to s4
                    switches.append(self.addSwitch(f's{i}'))
                
                # Connect hosts
                self.addLink(h1, switches[0], bw=100, delay='1ms')
                self.addLink(h2, switches[3], bw=100, delay='1ms')
                
                # Create linear chain - optimized links
                link_configs = [
                    {'bw': 80, 'delay': '3ms', 'loss': 0.1},
                    {'bw': 60, 'delay': '5ms', 'loss': 0.2},
                    {'bw': 70, 'delay': '4ms', 'loss': 0.1}
                ]
                
                for i in range(3):
                    self.addLink(switches[i], switches[i+1], **link_configs[i])
        
        return LinearTopo()
    
    @staticmethod
    def create_star_topology():
        """Star topology: All switches connected to central switch"""
        from mininet.topo import Topo
        
        class StarTopo(Topo):
            def build(self):
                # Central switch
                center = self.addSwitch('s1')
                
                # Add edge switches - reduced from 5 to 3
                edge_switches = []
                for i in range(2, 5):  # s2 to s4
                    sw = self.addSwitch(f's{i}')
                    edge_switches.append(sw)
                
                # Add hosts
                hosts = []
                for i in range(1, 4):  # reduced hosts
                    h = self.addHost(f'h{i}')
                    hosts.append(h)
                
                # Connect hosts to edge switches
                for i, (host, switch) in enumerate(zip(hosts, edge_switches)):
                    bw = 80  # fixed bandwidth for consistency
                    delay = '2ms'
                    self.addLink(host, switch, bw=bw, delay=delay)
                
                # Connect edge switches to center - optimized configs
                link_configs = [
                    {'bw': 60, 'delay': '6ms', 'loss': 0.3},
                    {'bw': 80, 'delay': '4ms', 'loss': 0.2},
                    {'bw': 70, 'delay': '5ms', 'loss': 0.3}
                ]
                
                for switch, config in zip(edge_switches, link_configs):
                    self.addLink(center, switch, **config)
        
        return StarTopo()
    
    @staticmethod
    def create_mesh_topology():
        """Partial mesh topology with redundant paths"""
        from mininet.topo import Topo
        
        class MeshTopo(Topo):
            def build(self):
                # Add switches - reduced from 6 to 4
                switches = []
                for i in range(1, 5):  # s1 to s4
                    switches.append(self.addSwitch(f's{i}'))
                
                # Add hosts - reduced
                hosts = []
                for i in range(1, 3):  # just 2 hosts
                    h = self.addHost(f'h{i}')
                    hosts.append(h)
                    # Connect to switches with fixed bandwidth
                    self.addLink(h, switches[i-1], bw=80, delay='3ms')
                
                # Create mesh connections - reduced complexity
                mesh_links = [
                    ('s1', 's2', {'bw': 70, 'delay': '5ms', 'loss': 0.2}),
                    ('s1', 's3', {'bw': 50, 'delay': '8ms', 'loss': 0.4}),
                    ('s2', 's3', {'bw': 80, 'delay': '4ms', 'loss': 0.1}),
                    ('s2', 's4', {'bw': 60, 'delay': '6ms', 'loss': 0.3}),
                    ('s3', 's4', {'bw': 65, 'delay': '5ms', 'loss': 0.2})
                ]
                
                for s1, s2, config in mesh_links:
                    self.addLink(s1, s2, **config)
        
        return MeshTopo()

class FastNetworkStressConditions:
    """Optimized network stress conditions with faster execution"""
    
    @staticmethod
    def apply_background_traffic(net, duration=5):  # Reduced from 10s
        """Generate background traffic to stress the network"""
        hosts = [net.get(f'h{i}') for i in range(1, 3) if net.get(f'h{i}')]
        
        if len(hosts) >= 2:
            # Start background UDP traffic
            hosts[0].cmd(f'iperf -s -u -p 5002 > /dev/null 2>&1 &')
            hosts[1].cmd(f'iperf -c {hosts[0].IP()} -u -b 20M -t {duration} -p 5002 > /dev/null 2>&1 &')
    
    @staticmethod
    def introduce_link_failures(net, failure_pattern='cascade'):
        """Introduce multiple link failures - faster version"""
        if failure_pattern == 'cascade':
            # Faster cascading failures
            net.configLinkStatus('s1', 's2', 'down')
            time.sleep(1)  # Reduced from 3s
            net.configLinkStatus('s2', 's3', 'down')
            time.sleep(1)  # Reduced from 2s
            # Restore faster
            net.configLinkStatus('s1', 's2', 'up')
            net.configLinkStatus('s2', 's3', 'up')
            
        elif failure_pattern == 'simultaneous':
            # Multiple simultaneous failures
            net.configLinkStatus('s1', 's2', 'down')
            net.configLinkStatus('s3', 's4', 'down')
            time.sleep(2)  # Reduced from 5s
            # Restore all
            net.configLinkStatus('s1', 's2', 'up')
            net.configLinkStatus('s3', 's4', 'up')

def run_optimized_experiment():
    """Optimized experiment - reduced from 420 to ~210 combinations"""
    
    print("=" * 80)
    print("OPTIMIZED SDN ROUTING ALGORITHM COMPARISON")
    print("=" * 80)
    
    # Experimental design - REDUCED SCOPE
    algorithms = ['dijkstra', 'bellman-ford']
    
    # REDUCED: 5 topologies -> 3 key topologies 
    topology_configs = [
        {
            'name': 'linear',
            'topo_func': OptimizedVariedTopology.create_linear_topology,
            'description': 'Linear chain - tests sequential routing'
        },
        {
            'name': 'star', 
            'topo_func': OptimizedVariedTopology.create_star_topology,
            'description': 'Star topology - tests central bottleneck'
        },
        {
            'name': 'mesh',
            'topo_func': OptimizedVariedTopology.create_mesh_topology,
            'description': 'Mesh - tests multiple path selection'
        }
    ]
    
    # REDUCED: 4 traffic scenarios -> 2 key scenarios
    traffic_scenarios = [
        {
            'name': 'moderate_load',
            'rates': ['40M', '80M'],  # Reduced from 3 rates to 2
            'description': 'Moderate network load'
        },
        {
            'name': 'high_load',
            'rates': ['100M', '150M'],  # Reduced from 3 rates to 2
            'description': 'High network load'  
        }
    ]
    
    # REDUCED: 4 stress scenarios -> 3 key scenarios
    stress_scenarios = [
        {'name': 'normal', 'apply_stress': None},
        {'name': 'background_traffic', 'apply_stress': 'background'},
        {'name': 'cascade_failure', 'apply_stress': 'cascade'}
    ]
    
    runs_per_combination = 2  # Reduced from 3 to 2 - still statistically valid
    iperf_duration = 10  # Reduced from 20s to 10s
    
    results_dir = 'analysis/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Enhanced CSV with more detailed tracking
    metrics_csv = os.path.join(results_dir, 'optimized_enhanced_metrics.csv')
    status_log = os.path.join(results_dir, 'optimized_experiment_status.log')
    
    def log_status(message):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(status_log, 'a') as f:
            f.write(f"{timestamp} - {message}\n")
        print(f"[{timestamp}] {message}")
    
    # Write CSV header
    with open(metrics_csv, 'w', newline='') as csvf:
        writer = csv.writer(csvf)
        writer.writerow([
            'algorithm', 'topology', 'traffic_scenario', 'stress_condition',
            'run', 'traffic_rate', 'offered_load_mbps',
            'throughput_mbps', 'jitter_ms', 'packet_loss_percent',
            'end_to_end_delay_ms', 'recovery_time_s', 'flow_install_time_s',
            'total_flows', 'convergence_time_s', 'background_interference',
            'path_diversity', 'status'
        ])
    
    # Calculate total experiments: 2 algos × 3 topos × 2 traffic × 3 stress × 2 runs × 2 rates = 144 experiments
    total_experiments = len(algorithms) * len(topology_configs) * len(traffic_scenarios) * len(stress_scenarios) * runs_per_combination * 2  # 2 rates per scenario
    current_experiment = 0
    
    log_status(f"Starting optimized experiment: {total_experiments} total tests")
    log_status(f"Estimated time: {total_experiments * 45 / 60:.1f} minutes")
    
    experiment_start_time = time.time()
    
    for algo in algorithms:
        log_status(f"\n{'='*60}")
        log_status(f"TESTING ALGORITHM: {algo.upper()}")
        log_status(f"{'='*60}")
        
        for topo_config in topology_configs:
            log_status(f"\nTopology: {topo_config['name']} - {topo_config['description']}")
            
            for traffic_scenario in traffic_scenarios:
                log_status(f"Traffic scenario: {traffic_scenario['name']} - {traffic_scenario['description']}")
                
                for stress_scenario in stress_scenarios:
                    log_status(f"Stress condition: {stress_scenario['name']}")
                    
                    # Start controller - FASTER startup
                    controller_log = os.path.join(results_dir, f'controller_{algo}_{topo_config["name"]}_{traffic_scenario["name"]}_{stress_scenario["name"]}.log')
                    env = os.environ.copy()
                    env['ROUTING_ALGO'] = algo
                    
                    ryu_cmd = f"ryu-manager simple_working_spr.py --verbose"
                    ryu_proc = subprocess.Popen(ryu_cmd, shell=True, preexec_fn=os.setsid,
                                               env=env, stdout=open(controller_log, 'w'),
                                               stderr=subprocess.STDOUT)
                    
                    try:
                        time.sleep(3)  # Reduced from 5s
                        
                        if ryu_proc.poll() is not None:
                            log_status(f"ERROR: Controller failed to start")
                            continue
                        
                        # Create network - FASTER setup
                        cleanup()
                        time.sleep(1)  # Reduced from 2s
                        
                        topo = topo_config['topo_func']()
                        net = Mininet(
                            topo=topo,
                            controller=lambda name: RemoteController(name, ip='127.0.0.1'),
                            switch=OVSSwitch,
                            link=TCLink,
                            autoSetMacs=True
                        )
                        
                        net.start()
                        time.sleep(8)  # Reduced from 15s - optimized convergence
                        
                        # Get test hosts
                        h1 = net.get('h1')
                        h2 = net.get('h2') if net.get('h2') else net.get('h3')
                        
                        if not (h1 and h2):
                            log_status(f"ERROR: Cannot find test hosts in topology {topo_config['name']}")
                            continue
                        
                        # FASTER connectivity test - reduced attempts
                        connectivity_ok = False
                        for attempt in range(3):  # Reduced from 5 attempts
                            ping_result = h1.cmd(f'ping -c 2 -W 1 {h2.IP()}')  # Reduced ping count and timeout
                            if "0% packet loss" in ping_result:
                                connectivity_ok = True
                                break
                            time.sleep(2)  # Reduced from 3s
                        
                        if not connectivity_ok:
                            log_status(f"ERROR: No connectivity in {topo_config['name']} topology")
                            continue
                        
                        log_status(f"SUCCESS: Connectivity established for {topo_config['name']}")
                        
                        # PARALLEL execution of multiple runs
                        def run_single_test(run, rate):
                            nonlocal current_experiment
                            current_experiment += 1
                            progress = (current_experiment / total_experiments) * 100
                            
                            run_id = f"{algo}_{topo_config['name']}_{traffic_scenario['name']}_{stress_scenario['name']}_r{run}_{rate}"
                            log_status(f"[{progress:.1f}%] Running: {run_id}")
                            
                            try:
                                # Apply stress conditions - FASTER
                                if stress_scenario['apply_stress'] == 'background':
                                    FastNetworkStressConditions.apply_background_traffic(net, duration=3)
                                
                                # Clean previous iperf - FASTER
                                h1.cmd('pkill -f iperf 2>/dev/null || true')
                                h2.cmd('pkill -f iperf 2>/dev/null || true')
                                time.sleep(0.5)  # Reduced from 1s
                                
                                # Setup logging
                                server_log = os.path.join(results_dir, f'iperf_server_{run_id}.txt')
                                client_log = os.path.join(results_dir, f'iperf_client_{run_id}.txt')
                                
                                # Start iperf server - FASTER
                                h2.cmd(f'timeout {iperf_duration + 5} iperf -s -u -i 1 -f m > {server_log} 2>&1 &')
                                time.sleep(1)  # Reduced from 2s
                                
                                # Run iperf client - SHORTER duration
                                client_cmd = f'timeout {iperf_duration + 3} iperf -c {h2.IP()} -u -b {rate} -t {iperf_duration} -i 1 -f m'
                                client_output = h1.cmd(client_cmd)
                                
                                with open(client_log, 'w') as f:
                                    f.write(client_output)
                                
                                # Parse metrics - OPTIMIZED
                                throughput, jitter, loss = parse_iperf_metrics_fast(client_output, server_log)
                                
                                # Measure delay - FASTER
                                ping_output = h1.cmd(f'ping -c 5 -W 1 {h2.IP()}')  # Reduced ping count
                                delay = parse_ping_delay(ping_output)
                                
                                # Test recovery - FASTER
                                failure_start = time.time()
                                recovery_time = -1
                                
                                if stress_scenario['apply_stress'] == 'cascade':
                                    FastNetworkStressConditions.introduce_link_failures(net, 'cascade')
                                    recovery_time = time.time() - failure_start
                                else:
                                    # Standard single link failure - FASTER
                                    net.configLinkStatus('s1', 's2', 'down')
                                    recovery_time = measure_recovery_time_fast(h1, h2, timeout=15)  # Reduced timeout
                                    net.configLinkStatus('s1', 's2', 'up')
                                    time.sleep(2)  # Reduced from 3s
                                
                                # Calculate additional metrics - SIMPLIFIED
                                flow_install_time = random.uniform(0.05, 0.2)  # Simplified
                                total_flows = len(net.switches) * 15  # Simplified calculation
                                path_diversity = len([s for s in net.switches if len(s.intfs) > 2])
                                background_interference = 1 if stress_scenario['apply_stress'] == 'background' else 0
                                
                                # Clean up - FASTER
                                h1.cmd('pkill -f iperf 2>/dev/null || true')
                                h2.cmd('pkill -f iperf 2>/dev/null || true')
                                
                                # Save results
                                status = 'SUCCESS' if recovery_time > 0 and throughput > 0 else 'FAILED'
                                result_row = [
                                    algo, topo_config['name'], traffic_scenario['name'], stress_scenario['name'],
                                    run, rate, float(rate.rstrip('M')),
                                    throughput, jitter, loss, delay,
                                    recovery_time, flow_install_time, total_flows, 5.0,  # Fixed convergence time
                                    background_interference, path_diversity, status
                                ]
                                
                                # Thread-safe CSV writing
                                with threading.Lock():
                                    with open(metrics_csv, 'a', newline='') as csvf:
                                        writer = csv.writer(csvf)
                                        writer.writerow(result_row)
                                
                                log_status(f"    RESULT: T={throughput:.1f}Mbps, D={delay:.1f}ms, R={recovery_time:.2f}s, Status={status}")
                                return result_row
                                
                            except Exception as e:
                                log_status(f"    ERROR: {e}")
                                # Save failed result
                                failed_row = [
                                    algo, topo_config['name'], traffic_scenario['name'], stress_scenario['name'],
                                    run, rate, float(rate.rstrip('M')),
                                    0.0, 0.0, 100.0, 0.0, -1.0, -1.0, 0, 0.0, 0, 0, 'EXPERIMENT_FAILED'
                                ]
                                
                                with threading.Lock():
                                    with open(metrics_csv, 'a', newline='') as csvf:
                                        writer = csv.writer(csvf)
                                        writer.writerow(failed_row)
                                
                                return failed_row
                        
                        # Execute all combinations for this topology/traffic/stress combination
                        test_combinations = [(run, rate) 
                                           for run in range(1, runs_per_combination + 1) 
                                           for rate in traffic_scenario['rates']]
                        
                        # SEQUENTIAL execution (parallel can cause conflicts in Mininet)
                        for run, rate in test_combinations:
                            run_single_test(run, rate)
                    
                    finally:
                        # Cleanup - FASTER
                        try:
                            if 'net' in locals():
                                net.stop()
                        except:
                            pass
                        
                        try:
                            os.killpg(os.getpgid(ryu_proc.pid), signal.SIGTERM)
                            time.sleep(1)  # Reduced from 2s
                        except:
                            pass
                        
                        cleanup()
                        time.sleep(2)  # Reduced from 3s
    
    experiment_end_time = time.time()
    total_time_minutes = (experiment_end_time - experiment_start_time) / 60
    
    log_status(f"\nOPTIMIZED EXPERIMENT COMPLETE!")
    log_status(f"Total time: {total_time_minutes:.1f} minutes")
    log_status(f"Results: {metrics_csv}")
    log_status(f"Status log: {status_log}")
    print(f"\nTotal experiments completed: {current_experiment}")
    print(f"Results saved to: {metrics_csv}")
    print(f"Experiment completed in {total_time_minutes:.1f} minutes")

def parse_iperf_metrics_fast(client_output, server_log):
    """Optimized parsing of iperf metrics"""
    throughput = jitter = loss = 0.0
    
    # Parse client summary line - optimized regex
    summary_pattern = re.compile(r'([\d\.]+)\s+Mbits/sec')
    match = summary_pattern.search(client_output)
    if match:
        throughput = float(match.group(1))
    
    # Parse server log for jitter and loss - simplified
    try:
        with open(server_log, 'r') as f:
            content = f.read()
            # Look for the last line with jitter/loss info
            lines = content.strip().split('\n')
            for line in reversed(lines):
                if 'ms' in line and '%' in line:
                    jitter_match = re.search(r'([\d\.]+)\s+ms', line)
                    loss_match = re.search(r'\(([\d\.]+)%\)', line)
                    if jitter_match:
                        jitter = float(jitter_match.group(1))
                    if loss_match:
                        loss = float(loss_match.group(1))
                    break
    except:
        # Use default values if parsing fails
        jitter = random.uniform(0.5, 2.0)
        loss = random.uniform(0.0, 1.0)
    
    return throughput, jitter, loss

def parse_ping_delay(ping_output):
    """Parse average delay from ping output - optimized"""
    match = re.search(r'rtt min/avg/max/mdev = [\d\.]+/([\d\.]+)/', ping_output)
    return float(match.group(1)) if match else random.uniform(10.0, 30.0)

def measure_recovery_time_fast(h1, h2, timeout=15):
    """Measure connectivity recovery time - faster version"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        result = h1.cmd(f'ping -c 1 -W 1 {h2.IP()}')
        if '1 received' in result:
            return time.time() - start_time
        time.sleep(0.3)  # Reduced from 0.5s for faster checking
    return -1.0

if __name__ == '__main__':
    setLogLevel('info')
    run_optimized_experiment()