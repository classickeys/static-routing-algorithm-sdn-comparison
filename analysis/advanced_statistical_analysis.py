import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, f_oneway, kruskal
import re
import os
import warnings
warnings.filterwarnings('ignore')

class AdvancedSDNAnalyzer:
    """Advanced statistical analyzer for multi-topology SDN routing experiments"""
    
    def __init__(self, results_dir='analysis/results'):
        self.results_dir = results_dir
        self.df = None
        self.control_plane_stats = {}
        self.time_series_data = {}
        self.topology_characteristics = {
            'linear': {'complexity': 'low', 'redundancy': 'none', 'bottleneck_potential': 'high'},
            'star': {'complexity': 'medium', 'redundancy': 'low', 'bottleneck_potential': 'very_high'},
            'mesh': {'complexity': 'high', 'redundancy': 'high', 'bottleneck_potential': 'low'},
            'original': {'complexity': 'high', 'redundancy': 'medium', 'bottleneck_potential': 'medium'},
            'bottleneck': {'complexity': 'medium', 'redundancy': 'low', 'bottleneck_potential': 'extreme'}
        }
        
    def load_and_enhance_data(self):
        """Load CSV data and enhance with topology-aware metrics"""
        # Try multiple file names
        metrics_files = [
            'optimized_enhanced_metrics.csv',
            'comprehensive_enhanced_metrics.csv', 
            'comprehensive_metrics.csv'
        ]
        
        metrics_file = None
        for filename in metrics_files:
            filepath = os.path.join(self.results_dir, filename)
            if os.path.exists(filepath):
                metrics_file = filepath
                break
        
        if not metrics_file:
            print(f"ERROR: No metrics file found in {self.results_dir}")
            print(f"Searched for: {metrics_files}")
            return False
            
        self.df = pd.read_csv(metrics_file)
        print(f"Loaded {len(self.df)} experimental records")
        print(f"Topologies: {self.df['topology'].unique() if 'topology' in self.df.columns else 'No topology column'}")
        print(f"Algorithms: {self.df['algorithm'].unique()}")
        
        # Add topology characteristics
        self._enhance_topology_data()
        
        # Extract control plane metrics (topology-aware)
        self._extract_control_plane_metrics()
        
        # Extract time series data (topology-aware)
        self._extract_flow_time_series()
        
        # Calculate topology-aware derived metrics
        self._calculate_derived_metrics()
        
        return True
    
    def _enhance_topology_data(self):
        """Add topology characteristic data"""
        print("Enhancing data with topology characteristics...")
        
        if 'topology' not in self.df.columns:
            print("WARNING: No topology column found")
            return
        
        # Add topology complexity scores
        complexity_map = {'low': 1, 'medium': 2, 'high': 3}
        redundancy_map = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
        bottleneck_map = {'low': 1, 'medium': 2, 'high': 3, 'very_high': 4, 'extreme': 5}
        
        self.df['topology_complexity'] = self.df['topology'].map(
            lambda x: complexity_map.get(self.topology_characteristics.get(x, {}).get('complexity', 'medium'), 2)
        )
        
        self.df['topology_redundancy'] = self.df['topology'].map(
            lambda x: redundancy_map.get(self.topology_characteristics.get(x, {}).get('redundancy', 'medium'), 2)
        )
        
        self.df['bottleneck_potential'] = self.df['topology'].map(
            lambda x: bottleneck_map.get(self.topology_characteristics.get(x, {}).get('bottleneck_potential', 'medium'), 2)
        )
        
        # Calculate topology-algorithm interaction effects
        self.df['algo_topo_interaction'] = self.df['algorithm'] + '_' + self.df['topology']
        
    def _extract_control_plane_metrics(self):
        """Extract topology-aware control plane message counts"""
        print("Extracting topology-aware control plane metrics...")
        
        control_metrics = []
        
        # Look for topology-specific controller logs
        log_patterns = [
            'controller_{algorithm}_{topology}_*.log',
            'controller_{algorithm}.log'
        ]
        
        for algo in self.df['algorithm'].unique():
            for topo in self.df['topology'].unique() if 'topology' in self.df.columns else ['']:
                
                # Try topology-specific logs first
                log_files = []
                if topo:
                    import glob
                    pattern = os.path.join(self.results_dir, f'controller_{algo}_{topo}_*.log')
                    log_files = glob.glob(pattern)
                
                # Fall back to general controller log
                if not log_files:
                    general_log = os.path.join(self.results_dir, f'controller_{algo}.log')
                    if os.path.exists(general_log):
                        log_files = [general_log]
                
                for log_file in log_files:
                    if os.path.exists(log_file):
                        metrics = self._parse_controller_log(log_file, algo, topo)
                        control_metrics.extend(metrics)
        
        if control_metrics:
            control_df = pd.DataFrame(control_metrics)
            
            # Merge based on available columns
            merge_cols = ['algorithm']
            if 'topology' in control_df.columns and 'topology' in self.df.columns:
                merge_cols.append('topology')
            
            self.df = self.df.merge(control_df, on=merge_cols, how='left')
        
        # Fill missing values with topology-algorithm group averages
        for col in ['packet_in_count', 'packet_out_count', 'flow_mod_count', 'flow_removed_count']:
            if col in self.df.columns:
                if 'topology' in self.df.columns:
                    self.df[col] = self.df.groupby(['algorithm', 'topology'])[col].transform(
                        lambda x: x.fillna(x.mean())
                    )
                else:
                    self.df[col] = self.df.groupby('algorithm')[col].transform(
                        lambda x: x.fillna(x.mean())
                    )
    
    def _parse_controller_log(self, log_file, algorithm, topology=''):
        """Parse controller log for OpenFlow message counts"""
        packet_in = packet_out = flow_mod = flow_removed = 0
        flow_install_times = []
        topology_events = 0
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                
                # Count different message types
                packet_in = len(re.findall(r'PACKET IN:', content))
                packet_out = len(re.findall(r'PACKET OUT:', content))
                flow_mod = len(re.findall(r'FLOW INSTALLED:', content))
                flow_removed = len(re.findall(r'FLOW REMOVED:', content))
                
                # Topology-specific events
                topology_events = len(re.findall(r'(LINK DOWN|LINK UP|TOPOLOGY CHANGE)', content))
                
                # Extract flow installation timestamps
                flow_lines = re.findall(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*FLOW INSTALLED', content)
                if flow_lines:
                    timestamps = pd.to_datetime(flow_lines)
                    if len(timestamps) > 1:
                        flow_install_times = (timestamps[1:] - timestamps[:-1]).total_seconds().tolist()
        
        except Exception as e:
            print(f"Warning: Could not parse {log_file}: {e}")
        
        result = {
            'algorithm': algorithm,
            'packet_in_count': packet_in,
            'packet_out_count': packet_out, 
            'flow_mod_count': flow_mod,
            'flow_removed_count': flow_removed,
            'topology_events': topology_events,
            'avg_flow_install_interval': np.mean(flow_install_times) if flow_install_times else 0,
            'total_control_messages': packet_in + packet_out + flow_mod + flow_removed
        }
        
        if topology:
            result['topology'] = topology
            
        return [result]
    
    def _extract_flow_time_series(self):
        """Extract topology-aware time series data"""
        print("Extracting topology-aware time series data...")
        
        for file in os.listdir(self.results_dir):
            if file.startswith('iperf_client') and file.endswith('.txt'):
                # Parse filename: iperf_client_dijkstra_linear_moderate_load_normal_r1_40M.txt
                parts = file.replace('iperf_client_', '').replace('.txt', '').split('_')
                
                if len(parts) >= 3:
                    algo = parts[0]
                    
                    # Try to identify topology in filename
                    topology = 'unknown'
                    for topo in ['linear', 'star', 'mesh', 'original', 'bottleneck']:
                        if topo in file:
                            topology = topo
                            break
                    
                    # Extract run and rate info
                    run = rate = 'unknown'
                    for part in parts:
                        if part.startswith('r') and part[1:].isdigit():
                            run = part
                        elif part.endswith('M') and part[:-1].isdigit():
                            rate = part
                    
                    series_data = self._parse_iperf_time_series(
                        os.path.join(self.results_dir, file), algo, topology, run, rate
                    )
                    
                    key = f"{algo}_{topology}_{run}_{rate}"
                    self.time_series_data[key] = series_data
    
    def _parse_iperf_time_series(self, file_path, algo, topology, run, rate):
        """Parse iperf output for topology-aware time series throughput data"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            intervals = []
            lines = content.split('\n')
            
            for line in lines:
                match = re.search(r'\[\s*\d+\]\s+([\d\.]+)-\s*([\d\.]+)\s+sec.*?([\d\.]+)\s+Mbits/sec', line)
                if match:
                    start_time = float(match.group(1))
                    end_time = float(match.group(2))
                    throughput = float(match.group(3))
                    
                    intervals.append({
                        'algorithm': algo,
                        'topology': topology,
                        'run': run,
                        'rate': rate,
                        'time_start': start_time,
                        'time_end': end_time,
                        'interval_throughput': throughput
                    })
            
            return intervals
            
        except Exception as e:
            print(f"Warning: Could not parse {file_path}: {e}")
            return []
    
    def _calculate_derived_metrics(self):
        """Calculate topology-aware sophisticated derived metrics"""
        print("Calculating topology-aware derived metrics...")
        
        # Control plane efficiency (topology-adjusted)
        if 'total_control_messages' in self.df.columns and 'throughput_mbps' in self.df.columns:
            self.df['control_efficiency'] = self.df['throughput_mbps'] / (self.df['total_control_messages'] + 1)
            
            # Topology-adjusted control efficiency
            if 'topology_complexity' in self.df.columns:
                self.df['topology_adjusted_control_efficiency'] = (
                    self.df['control_efficiency'] / (self.df['topology_complexity'] + 1)
                )
        
        # Network utilization (topology-aware)
        self.df['network_utilization'] = self.df['throughput_mbps'] / self.df['offered_load_mbps']
        
        # Performance consistency per topology
        if 'topology' in self.df.columns:
            consistency_stats = self.df.groupby(['algorithm', 'topology', 'offered_load_mbps'])['throughput_mbps'].agg(['mean', 'std'])
            consistency_stats['cv'] = consistency_stats['std'] / consistency_stats['mean']
            consistency_stats = consistency_stats.reset_index()
            
            self.df = self.df.merge(
                consistency_stats[['algorithm', 'topology', 'offered_load_mbps', 'cv']], 
                on=['algorithm', 'topology', 'offered_load_mbps'], 
                how='left',
                suffixes=('', '_cv')
            )
        
        # Topology-weighted QoS score
        topology_weight = 1.0
        if 'bottleneck_potential' in self.df.columns:
            # Higher weight penalty for high bottleneck potential
            topology_weight = 1.0 / (1.0 + self.df['bottleneck_potential'] * 0.1)
        
        self.df['qos_score'] = (
            (self.df['throughput_mbps'] / self.df['offered_load_mbps']) * 0.4 +
            (1 / (self.df['end_to_end_delay_ms'] + 1)) * 0.3 +
            (1 / (self.df['jitter_ms'] + 1)) * 0.2 +
            (1 / (self.df['packet_loss_percent'] + 1)) * 0.1
        ) * topology_weight
        
        # Topology-adjusted resilience metric
        base_resilience = 1 / (self.df['recovery_time_s'] + 0.1)
        
        if 'topology_redundancy' in self.df.columns:
            # Resilience should be harder to achieve in low-redundancy topologies
            redundancy_factor = (self.df['topology_redundancy'] + 1) / 4.0  # Normalize to 0.25-1.0
            self.df['resilience_score'] = base_resilience * redundancy_factor
        else:
            self.df['resilience_score'] = base_resilience
        
        # Algorithm suitability per topology
        if 'topology' in self.df.columns:
            self.df['algorithm_topology_suitability'] = self._calculate_algo_topo_suitability()
    
    def _calculate_algo_topo_suitability(self):
        """Calculate algorithm suitability for each topology"""
        suitability_scores = []
        
        for _, row in self.df.iterrows():
            algo = row['algorithm']
            topo = row['topology'] if 'topology' in row else 'unknown'
            
            # Heuristic: Dijkstra better for simple topologies, Bellman-Ford better for complex
            if algo == 'dijkstra':
                if topo in ['linear', 'star']:
                    score = 1.2  # 20% bonus for simple topologies
                elif topo in ['mesh', 'original']:
                    score = 0.9  # 10% penalty for complex topologies
                else:
                    score = 1.0
            elif algo == 'bellman-ford':
                if topo in ['mesh', 'original', 'bottleneck']:
                    score = 1.1  # 10% bonus for complex/problematic topologies
                elif topo in ['linear', 'star']:
                    score = 0.95  # 5% penalty for simple topologies
                else:
                    score = 1.0
            else:
                score = 1.0
            
            suitability_scores.append(score)
        
        return suitability_scores
    
    def run_statistical_tests(self):
        """Run comprehensive topology-aware statistical analysis"""
        print("\n" + "="*80)
        print("COMPREHENSIVE TOPOLOGY-AWARE STATISTICAL ANALYSIS")
        print("="*80)
        
        results = {}
        
        # 1. Normality tests (per topology)
        results['normality'] = self._test_normality_by_topology()
        
        # 2. Algorithm performance comparison (topology-stratified)
        results['performance_tests'] = self._topology_stratified_performance_tests()
        
        # 3. Topology effect analysis
        results['topology_effects'] = self._analyze_topology_effects()
        
        # 4. Algorithm-topology interaction effects
        results['interaction_effects'] = self._analyze_interaction_effects()
        
        # 5. Control plane analysis (topology-aware)
        results['control_plane'] = self._topology_aware_control_plane_analysis()
        
        # 6. Time series analysis (topology-specific)
        results['time_series'] = self._topology_specific_time_series_analysis()
        
        # 7. Effect size calculations (topology-stratified)
        results['effect_sizes'] = self._calculate_topology_stratified_effect_sizes()
        
        return results
    
    def _test_normality_by_topology(self):
        """Test normality of key metrics by topology"""
        print("\n1. NORMALITY TESTS BY TOPOLOGY")
        print("-" * 50)
        
        normality_results = {}
        metrics = ['throughput_mbps', 'recovery_time_s', 'jitter_ms', 'end_to_end_delay_ms']
        
        if 'topology' not in self.df.columns:
            print("No topology column found - running basic normality tests")
            return self._test_normality_basic()
        
        for topo in self.df['topology'].unique():
            print(f"\nTOPOLOGY: {topo.upper()}")
            topo_data = self.df[self.df['topology'] == topo]
            
            for metric in metrics:
                if metric in topo_data.columns:
                    for algo in topo_data['algorithm'].unique():
                        data = topo_data[topo_data['algorithm'] == algo][metric].dropna()
                        
                        if len(data) >= 3:
                            stat, p_value = stats.shapiro(data)
                            key = f"{topo}_{algo}_{metric}"
                            normality_results[key] = {
                                'statistic': stat,
                                'p_value': p_value,
                                'is_normal': p_value > 0.05
                            }
                            
                            print(f"  {algo} {metric}: W={stat:.4f}, p={p_value:.4f} "
                                  f"({'Normal' if p_value > 0.05 else 'Non-normal'})")
        
        return normality_results
    
    def _test_normality_basic(self):
        """Basic normality test when no topology info"""
        normality_results = {}
        metrics = ['throughput_mbps', 'recovery_time_s', 'jitter_ms', 'end_to_end_delay_ms']
        
        for metric in metrics:
            if metric in self.df.columns:
                for algo in self.df['algorithm'].unique():
                    data = self.df[self.df['algorithm'] == algo][metric].dropna()
                    
                    if len(data) >= 3:
                        stat, p_value = stats.shapiro(data)
                        normality_results[f"{algo}_{metric}"] = {
                            'statistic': stat,
                            'p_value': p_value,
                            'is_normal': p_value > 0.05
                        }
                        
                        print(f"{algo} {metric}: W={stat:.4f}, p={p_value:.4f} "
                              f"({'Normal' if p_value > 0.05 else 'Non-normal'})")
        
        return normality_results
    
    def _topology_stratified_performance_tests(self):
        """Statistical tests comparing algorithms within each topology"""
        print("\n2. TOPOLOGY-STRATIFIED PERFORMANCE COMPARISON")
        print("-" * 50)
        
        test_results = {}
        metrics = ['throughput_mbps', 'recovery_time_s', 'jitter_ms', 'end_to_end_delay_ms', 'qos_score']
        
        if 'topology' not in self.df.columns:
            print("No topology column - running basic comparison")
            return self._basic_performance_tests()
        
        for topo in self.df['topology'].unique():
            print(f"\nTOPOLOGY: {topo.upper()}")
            topo_data = self.df[self.df['topology'] == topo]
            
            algorithms = topo_data['algorithm'].unique()
            if len(algorithms) < 2:
                print(f"  Not enough algorithms for comparison in {topo}")
                continue
            
            for metric in metrics:
                if metric not in topo_data.columns:
                    continue
                    
                print(f"\n  {metric.upper()}:")
                
                # Get data for each algorithm in this topology
                groups = []
                for algo in algorithms:
                    data = topo_data[topo_data['algorithm'] == algo][metric].dropna()
                    groups.append(data)
                    print(f"    {algo}: n={len(data)}, mean={data.mean():.3f}±{data.std():.3f}")
                
                if len(groups) >= 2 and all(len(g) >= 3 for g in groups):
                    # Independent t-test
                    if len(groups) == 2:
                        t_stat, t_p = ttest_ind(groups[0], groups[1])
                        u_stat, u_p = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                        
                        test_results[f"{topo}_{metric}_ttest"] = {
                            'statistic': t_stat, 'p_value': t_p,
                            'significant': t_p < 0.05
                        }
                        
                        test_results[f"{topo}_{metric}_mannwhitney"] = {
                            'statistic': u_stat, 'p_value': u_p,
                            'significant': u_p < 0.05
                        }
                        
                        print(f"    t-test: t={t_stat:.3f}, p={t_p:.4f} {'***' if t_p < 0.001 else '**' if t_p < 0.01 else '*' if t_p < 0.05 else 'ns'}")
                        print(f"    Mann-Whitney: U={u_stat:.3f}, p={u_p:.4f} {'***' if u_p < 0.001 else '**' if u_p < 0.01 else '*' if u_p < 0.05 else 'ns'}")
        
        return test_results
    
    def _basic_performance_tests(self):
        """Basic performance tests when no topology data"""
        test_results = {}
        metrics = ['throughput_mbps', 'recovery_time_s', 'jitter_ms', 'end_to_end_delay_ms', 'qos_score']
        
        algorithms = self.df['algorithm'].unique()
        if len(algorithms) < 2:
            return test_results
        
        for metric in metrics:
            if metric not in self.df.columns:
                continue
                
            groups = []
            for algo in algorithms:
                data = self.df[self.df['algorithm'] == algo][metric].dropna()
                groups.append(data)
            
            if len(groups) >= 2 and all(len(g) >= 3 for g in groups):
                if len(groups) == 2:
                    t_stat, t_p = ttest_ind(groups[0], groups[1])
                    u_stat, u_p = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                    
                    test_results[f"{metric}_ttest"] = {
                        'statistic': t_stat, 'p_value': t_p,
                        'significant': t_p < 0.05
                    }
                    
                    test_results[f"{metric}_mannwhitney"] = {
                        'statistic': u_stat, 'p_value': u_p,
                        'significant': u_p < 0.05
                    }
        
        return test_results
    
    def _analyze_topology_effects(self):
        """Analyze the effect of topology on performance metrics"""
        print("\n3. TOPOLOGY EFFECTS ANALYSIS")
        print("-" * 50)
        
        topology_results = {}
        
        if 'topology' not in self.df.columns:
            print("No topology column found - skipping topology effects analysis")
            return topology_results
        
        metrics = ['throughput_mbps', 'recovery_time_s', 'jitter_ms', 'qos_score']
        
        for metric in metrics:
            if metric not in self.df.columns:
                continue
                
            print(f"\n{metric.upper()} BY TOPOLOGY:")
            
            # Group by topology and calculate statistics
            topo_stats = self.df.groupby('topology')[metric].agg(['mean', 'std', 'count'])
            
            for topo in topo_stats.index:
                stats_row = topo_stats.loc[topo]
                print(f"  {topo}: mean={stats_row['mean']:.3f}±{stats_row['std']:.3f} (n={stats_row['count']})")
            
            # ANOVA to test for topology effects
            topology_groups = [self.df[self.df['topology'] == topo][metric].dropna() 
                             for topo in self.df['topology'].unique()]
            
            try:
                f_stat, f_p = f_oneway(*topology_groups)
                k_stat, k_p = kruskal(*topology_groups)
            except ValueError as e:
                if "identical" in str(e):
                    print(f"  Warning: All values identical for {metric} - skipping statistical tests")
                    f_stat = f_p = k_stat = k_p = np.nan
                else:
                    raise e
            
            topology_results[f"{metric}_topology_anova"] = {
                'f_statistic': f_stat, 'p_value': f_p,
                'significant': f_p < 0.05
            }
            
            topology_results[f"{metric}_topology_kruskal"] = {
                'h_statistic': k_stat, 'p_value': k_p,
                'significant': k_p < 0.05
            }
            
            print(f"  ANOVA: F={f_stat:.3f}, p={f_p:.4f} {'***' if f_p < 0.001 else '**' if f_p < 0.01 else '*' if f_p < 0.05 else 'ns'}")
            print(f"  Kruskal-Wallis: H={k_stat:.3f}, p={k_p:.4f} {'***' if k_p < 0.001 else '**' if k_p < 0.01 else '*' if k_p < 0.05 else 'ns'}")
        
        return topology_results
    
    def _analyze_interaction_effects(self):
        """Analyze algorithm-topology interaction effects"""
        print("\n4. ALGORITHM-TOPOLOGY INTERACTION ANALYSIS")
        print("-" * 50)
        
        interaction_results = {}
        
        if 'topology' not in self.df.columns:
            print("No topology column found - skipping interaction analysis")
            return interaction_results
        
        metrics = ['throughput_mbps', 'recovery_time_s', 'qos_score']
        
        for metric in metrics:
            if metric not in self.df.columns:
                continue
                
            print(f"\n{metric.upper()} ALGORITHM-TOPOLOGY INTERACTIONS:")
            
            # Create interaction summary
            interaction_summary = self.df.groupby(['algorithm', 'topology'])[metric].agg(['mean', 'std', 'count'])
            
            for (algo, topo), stats_row in interaction_summary.iterrows():
                print(f"  {algo} × {topo}: {stats_row['mean']:.3f}±{stats_row['std']:.3f} (n={stats_row['count']})")
            
            # Two-way ANOVA using scipy (simplified)
            # Calculate main effects and interaction
            algo_means = self.df.groupby('algorithm')[metric].mean()
            topo_means = self.df.groupby('topology')[metric].mean()
            
            print(f"  Algorithm main effects:")
            for algo, mean_val in algo_means.items():
                print(f"    {algo}: {mean_val:.3f}")
            
            print(f"  Topology main effects:")
            for topo, mean_val in topo_means.items():
                print(f"    {topo}: {mean_val:.3f}")
            
            # Store interaction effects
            interaction_results[f"{metric}_algorithm_effects"] = algo_means.to_dict()
            interaction_results[f"{metric}_topology_effects"] = topo_means.to_dict()
            interaction_results[f"{metric}_interaction_matrix"] = interaction_summary['mean'].unstack().to_dict()
        
        return interaction_results
    
    def _topology_aware_control_plane_analysis(self):
        """Analyze control plane efficiency by topology"""
        print("\n5. TOPOLOGY-AWARE CONTROL PLANE ANALYSIS")
        print("-" * 50)
        
        cp_results = {}
        
        if 'total_control_messages' not in self.df.columns:
            print("No control plane data available")
            return cp_results
        
        if 'topology' in self.df.columns:
            # Control plane efficiency by topology
            cp_stats = self.df.groupby(['algorithm', 'topology']).agg({
                'total_control_messages': ['mean', 'std'],
                'control_efficiency': ['mean', 'std'] if 'control_efficiency' in self.df.columns else ['count']
            })
            
            cp_results['control_plane_by_topology'] = cp_stats
            
            print("Control Messages by Algorithm and Topology:")
            for (algo, topo), stats_row in cp_stats.iterrows():
                total_msgs = stats_row[('total_control_messages', 'mean')]
                print(f"  {algo} × {topo}: {total_msgs:.1f} messages")
                
        return cp_results
    
    def _topology_specific_time_series_analysis(self):
        """Analyze time series patterns by topology"""
        print("\n6. TOPOLOGY-SPECIFIC TIME SERIES ANALYSIS")  
        print("-" * 50)
        
        ts_results = {}
        
        if not self.time_series_data:
            print("No time series data available")
            return ts_results
        
        # Group time series by topology
        topology_series = {}
        
        for key, series in self.time_series_data.items():
            if series:
                topo = series[0].get('topology', 'unknown')
                if topo not in topology_series:
                    topology_series[topo] = []
                topology_series[topo].append(series)
        
        # Analyze stability by topology
        for topo, topo_series_list in topology_series.items():
            print(f"\nTOPOLOGY: {topo.upper()}")
            
            stability_metrics = []
            
            for series in topo_series_list:
                if series:
                    df_series = pd.DataFrame(series)
                    throughputs = df_series['interval_throughput']
                    
                    if len(throughputs) > 1:
                        stability = {
                            'algorithm': series[0]['algorithm'],
                            'topology': topo,
                            'cv': throughputs.std() / throughputs.mean() if throughputs.mean() > 0 else 0,
                            'range': throughputs.max() - throughputs.min()
                        }
                        stability_metrics.append(stability)
            
            if stability_metrics:
                stability_df = pd.DataFrame(stability_metrics)
                
                # Compare algorithms within this topology
                algo_stability = stability_df.groupby('algorithm')['cv'].agg(['mean', 'std'])
                
                for algo in algo_stability.index:
                    avg_cv = algo_stability.loc[algo, 'mean']
                    print(f"  {algo}: CV = {avg_cv:.4f} ({'Stable' if avg_cv < 0.1 else 'Moderate' if avg_cv < 0.3 else 'Unstable'})")
                
                ts_results[f"{topo}_stability"] = algo_stability.to_dict()
        
        return ts_results
    
    def _calculate_topology_stratified_effect_sizes(self):
        """Calculate effect sizes within each topology"""
        print("\n7. TOPOLOGY-STRATIFIED EFFECT SIZE ANALYSIS")
        print("-" * 50)
        
        effect_sizes = {}
        metrics = ['throughput_mbps', 'recovery_time_s', 'jitter_ms', 'qos_score']
        
        if 'topology' not in self.df.columns:
            print("No topology column - calculating overall effect sizes")
            return self._basic_effect_sizes()
        
        for topo in self.df['topology'].unique():
            print(f"\nTOPOLOGY: {topo.upper()}")
            topo_data = self.df[self.df['topology'] == topo]
            
            algorithms = list(topo_data['algorithm'].unique())
            
            if len(algorithms) < 2:
                continue
            
            for metric in metrics:
                if metric not in topo_data.columns:
                    continue
                    
                print(f"  {metric.upper()}:")
                
                for i, algo1 in enumerate(algorithms):
                    for algo2 in algorithms[i+1:]:
                        data1 = topo_data[topo_data['algorithm'] == algo1][metric].dropna()
                        data2 = topo_data[topo_data['algorithm'] == algo2][metric].dropna()
                        
                        if len(data1) >= 3 and len(data2) >= 3:
                            # Cohen's d
                            pooled_std = np.sqrt(((len(data1) - 1) * data1.var() + 
                                                 (len(data2) - 1) * data2.var()) / 
                                                 (len(data1) + len(data2) - 2))
                            
                            if pooled_std > 0:
                                cohens_d = (data1.mean() - data2.mean()) / pooled_std
                                
                                # Effect size interpretation
                                if abs(cohens_d) < 0.2:
                                    size_label = "negligible"
                                elif abs(cohens_d) < 0.5:
                                    size_label = "small"
                                elif abs(cohens_d) < 0.8:
                                    size_label = "medium"
                                else:
                                    size_label = "large"
                                
                                effect_sizes[f"{topo}_{metric}_{algo1}_{algo2}"] = {
                                    'cohens_d': cohens_d,
                                    'effect_size': size_label,
                                    'mean_diff': data1.mean() - data2.mean()
                                }
                                
                                print(f"    {algo1} vs {algo2}: d={cohens_d:.3f} ({size_label})")
        
        return effect_sizes
    
    def _basic_effect_sizes(self):
        """Basic effect sizes when no topology data"""
        effect_sizes = {}
        metrics = ['throughput_mbps', 'recovery_time_s', 'jitter_ms', 'qos_score']
        algorithms = list(self.df['algorithm'].unique())
        
        for metric in metrics:
            if metric not in self.df.columns:
                continue
            
            for i, algo1 in enumerate(algorithms):
                for algo2 in algorithms[i+1:]:
                    data1 = self.df[self.df['algorithm'] == algo1][metric].dropna()
                    data2 = self.df[self.df['algorithm'] == algo2][metric].dropna()
                    
                    if len(data1) >= 3 and len(data2) >= 3:
                        pooled_std = np.sqrt(((len(data1) - 1) * data1.var() + 
                                             (len(data2) - 1) * data2.var()) / 
                                             (len(data1) + len(data2) - 2))
                        
                        if pooled_std > 0:
                            cohens_d = (data1.mean() - data2.mean()) / pooled_std
                            effect_sizes[f"{metric}_{algo1}_{algo2}"] = {
                                'cohens_d': cohens_d,
                                'effect_size': 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'
                            }
        
        return effect_sizes
    
    def create_advanced_visualizations(self):
        """Create comprehensive topology-aware visualizations"""
        print("\nCreating advanced topology-aware visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Topology-stratified performance dashboard
        self._plot_topology_stratified_dashboard()
        
        # 2. Algorithm-topology interaction heatmaps
        self._plot_interaction_heatmaps()
        
        # 3. Topology-specific control plane analysis
        self._plot_topology_control_plane()
        
        # 4. Time series by topology
        self._plot_topology_time_series()
        
        # 5. Performance distributions by topology
        self._plot_topology_performance_distributions()
    
    def _plot_topology_stratified_dashboard(self):
        """Main topology-aware dashboard"""
        if 'topology' not in self.df.columns:
            print("No topology data for stratified plotting")
            return
            
        topologies = self.df['topology'].unique()
        n_topos = len(topologies)
        
        fig, axes = plt.subplots(2, min(3, n_topos), figsize=(6*min(3, n_topos), 12))
        if n_topos == 1:
            axes = axes.reshape(-1, 1)
        fig.suptitle('Topology-Stratified Performance Analysis', fontsize=16, fontweight='bold')
        
        for i, topo in enumerate(topologies[:min(3, n_topos)]):
            topo_data = self.df[self.df['topology'] == topo]
            
            # Throughput comparison
            ax1 = axes[0, i] if n_topos > 1 else axes[0]
            if len(topo_data['algorithm'].unique()) > 1:
                throughput_stats = topo_data.groupby(['algorithm', 'offered_load_mbps'])['throughput_mbps'].agg(['mean', 'std']).reset_index()
                
                for algo in topo_data['algorithm'].unique():
                    algo_data = throughput_stats[throughput_stats['algorithm'] == algo]
                    ax1.errorbar(algo_data['offered_load_mbps'], algo_data['mean'], 
                                yerr=algo_data['std'], label=algo.capitalize(), 
                                marker='o', linewidth=2, capsize=3)
            
            ax1.set_xlabel('Offered Load (Mbps)')
            ax1.set_ylabel('Throughput (Mbps)')
            ax1.set_title(f'{topo.capitalize()} Topology')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # QoS Score comparison
            ax2 = axes[1, i] if n_topos > 1 else axes[1]
            if 'qos_score' in topo_data.columns:
                sns.boxplot(data=topo_data, x='algorithm', y='qos_score', ax=ax2)
                ax2.set_title(f'{topo.capitalize()} QoS Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'topology_stratified_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.results_dir, 'topology_stratified_analysis.pdf'), 
                   bbox_inches='tight')
    
    def _plot_interaction_heatmaps(self):
        """Algorithm-topology interaction heatmaps"""
        if 'topology' not in self.df.columns:
            return
            
        metrics = ['throughput_mbps', 'recovery_time_s', 'qos_score']
        n_metrics = len([m for m in metrics if m in self.df.columns])
        
        if n_metrics == 0:
            return
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        fig.suptitle('Algorithm-Topology Interaction Effects', fontsize=14, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            if metric not in self.df.columns:
                continue
                
            # Create pivot table for heatmap
            pivot_data = self.df.pivot_table(
                values=metric, 
                index='algorithm', 
                columns='topology', 
                aggfunc='mean'
            )
            
            # Create heatmap
            sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                       ax=axes[i], cbar_kws={'shrink': 0.8})
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_xlabel('Topology')
            axes[i].set_ylabel('Algorithm')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'algorithm_topology_interactions.png'), 
                   dpi=300, bbox_inches='tight')
    
    def _plot_topology_control_plane(self):
        """Topology-specific control plane analysis"""
        if 'total_control_messages' not in self.df.columns or 'topology' not in self.df.columns:
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Control Plane Efficiency by Topology', fontsize=14, fontweight='bold')
        
        # Total messages by topology and algorithm
        ax1 = axes[0]
        sns.barplot(data=self.df, x='topology', y='total_control_messages', 
                   hue='algorithm', ax=ax1, ci=95)
        ax1.set_title('Control Messages by Topology')
        ax1.set_ylabel('Total Messages')
        ax1.tick_params(axis='x', rotation=45)
        
        # Control efficiency by topology
        ax2 = axes[1]
        if 'control_efficiency' in self.df.columns:
            sns.boxplot(data=self.df, x='topology', y='control_efficiency', 
                       hue='algorithm', ax=ax2)
            ax2.set_title('Control Efficiency by Topology')
            ax2.set_ylabel('Efficiency Score')
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'topology_control_plane.png'), 
                   dpi=300, bbox_inches='tight')
    
    def _plot_topology_time_series(self):
        """Time series plots by topology"""
        if not self.time_series_data:
            return
            
        # Group by topology
        topology_series = {}
        for key, series in self.time_series_data.items():
            if series:
                topo = series[0].get('topology', 'unknown')
                if topo not in topology_series:
                    topology_series[topo] = []
                topology_series[topo].append((key, series))
        
        n_topos = len(topology_series)
        if n_topos == 0:
            return
        
        fig, axes = plt.subplots(n_topos, 1, figsize=(12, 4*n_topos))
        if n_topos == 1:
            axes = [axes]
        
        fig.suptitle('Throughput Time Series by Topology', fontsize=14, fontweight='bold')
        
        for i, (topo, series_list) in enumerate(topology_series.items()):
            ax = axes[i]
            
            for key, series in series_list[:3]:  # Show up to 3 series per topology
                if series:
                    df_series = pd.DataFrame(series)
                    algo = series[0]['algorithm']
                    ax.plot(df_series['time_start'], df_series['interval_throughput'], 
                           label=f"{algo} ({key.split('_')[-1]})", alpha=0.7)
            
            ax.set_title(f'{topo.capitalize()} Topology')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Throughput (Mbps)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'topology_time_series.png'), 
                   dpi=300, bbox_inches='tight')
    
    def _plot_topology_performance_distributions(self):
        """Performance distribution plots by topology"""
        if 'topology' not in self.df.columns:
            return
            
        metrics = ['throughput_mbps', 'recovery_time_s', 'jitter_ms', 'end_to_end_delay_ms']
        available_metrics = [m for m in metrics if m in self.df.columns]
        
        if not available_metrics:
            return
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        fig.suptitle('Performance Distributions by Topology', fontsize=14, fontweight='bold')
        
        for i, metric in enumerate(available_metrics[:4]):
            ax = axes[i]
            
            # Create subplot for each topology
            for topo in self.df['topology'].unique():
                topo_data = self.df[self.df['topology'] == topo]
                
                for algo in topo_data['algorithm'].unique():
                    data = topo_data[topo_data['algorithm'] == algo][metric].dropna()
                    if len(data) > 0:
                        ax.hist(data, alpha=0.4, label=f"{algo}-{topo}", bins=10)
            
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'topology_performance_distributions.png'), 
                   dpi=300, bbox_inches='tight')

def main():
    """Main analysis execution"""
    # Try different results directories
    possible_dirs = ['analysis/results', 'results', '.']
    
    analyzer = None
    for results_dir in possible_dirs:
        if os.path.exists(results_dir):
            analyzer = AdvancedSDNAnalyzer(results_dir)
            if analyzer.load_and_enhance_data():
                break
    
    if not analyzer or not analyzer.df is not None:
        print("ERROR: Could not load data from any results directory")
        return
    
    # Run statistical tests
    test_results = analyzer.run_statistical_tests()
    
    # Create visualizations
    analyzer.create_advanced_visualizations()
    
    # Save results
    results_file = os.path.join(analyzer.results_dir, 'topology_aware_statistical_analysis.txt')
    with open(results_file, 'w') as f:
        f.write("TOPOLOGY-AWARE STATISTICAL ANALYSIS RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for category, results in test_results.items():
            f.write(f"{category.upper()}:\n")
            f.write("-" * 40 + "\n")
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
    
    print(f"\nTopology-aware analysis complete! Results saved to:")
    print(f"- Statistical tests: {results_file}")
    print(f"- Main dashboard: {analyzer.results_dir}/topology_stratified_analysis.png")
    print(f"- Interaction analysis: {analyzer.results_dir}/algorithm_topology_interactions.png")
    print(f"- Control plane analysis: {analyzer.results_dir}/topology_control_plane.png")

if __name__ == '__main__':
    main()