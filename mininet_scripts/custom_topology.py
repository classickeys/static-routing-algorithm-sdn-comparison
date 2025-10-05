from mininet.topo import Topo
from mininet.net import Mininet
from mininet.cli import CLI
from mininet.log import setLogLevel
import time

class LargerTopo(Topo):
    """
    Larger topology with multiple redundant paths for SPR+STP testing.
    Creates a mesh network where STP will block some links to prevent loops,
    but SPR will find optimal paths through the remaining spanning tree.
    
    Topology:
    h1 -- s1 -- s2 -- s4 -- h2
          |  \  |   /  |
          |   \ |  /   |  
          s3 -- s5 -- s6 -- h3
          |           |
          h4         h5
    """
    def build(self, **_opts):
        # Add hosts - make sure h2 is properly added
        h1 = self.addHost('h1', ip='10.0.0.1/24', mac='00:00:00:00:00:01')
        h2 = self.addHost('h2', ip='10.0.0.2/24', mac='00:00:00:00:00:02')  # This line crucial!
        h3 = self.addHost('h3', ip='10.0.0.3/24', mac='00:00:00:00:00:03')
        h4 = self.addHost('h4', ip='10.0.0.4/24', mac='00:00:00:00:00:04')
        h5 = self.addHost('h5', ip='10.0.0.5/24', mac='00:00:00:00:00:05')
        
        # Add switches with proper OpenFlow version
        switches = []
        for i in range(1, 7):
            switch = self.addSwitch(f's{i}', protocols='OpenFlow13')
            switches.append(switch)

        # Define host link options (empty by default)
        host_link_opts = {}

        # Connect hosts to switches - verify these connections!
        self.addLink(h1, switches[0], port1=0, port2=1, **host_link_opts)  # h1 -> s1 port 1
        self.addLink(h2, switches[3], port1=0, port2=1, **host_link_opts)  # h2 -> s4 port 1  ← CRITICAL!
        self.addLink(h3, switches[5], port1=0, port2=1, **host_link_opts)  # h3 -> s6 port 1
        self.addLink(h4, switches[2], port1=0, port2=1, **host_link_opts)  # h4 -> s3 port 1
        self.addLink(h5, switches[5], port1=0, port2=2, **host_link_opts)  # h5 -> s6 port 2

        # Create mesh topology with redundant paths
        # Primary backbone
        self.addLink(switches[0], switches[1], bw=80, delay='5ms')    # s1-s2 Main path
        self.addLink(switches[1], switches[3], bw=80, delay='5ms')    # s2-s4 To h2
        
        # Secondary paths (some will be blocked by STP)
        self.addLink(switches[0], switches[2], bw=60, delay='8ms')    # s1-s3 Alternative route
        self.addLink(switches[2], switches[4], bw=60, delay='8ms')    # s3-s5 Through bottom
        self.addLink(switches[4], switches[5], bw=60, delay='8ms')    # s5-s6 To edge
        
        # Cross connections (create redundancy)
        self.addLink(switches[0], switches[4], bw=50, delay='12ms')   # s1-s5 Direct connection
        self.addLink(switches[1], switches[4], bw=55, delay='10ms')   # s2-s5 Cross link
        self.addLink(switches[3], switches[5], bw=70, delay='6ms')    # s4-s6 Edge connection
        
        # Additional redundant links (STP will block some)
        self.addLink(switches[2], switches[5], bw=40, delay='15ms')   # s3-s6 Long path
        self.addLink(switches[1], switches[5], bw=45, delay='14ms')   # s2-s6 Another cross link

def run_experiment():
    """Run the SPR+STP experiment with the larger topology"""
    setLogLevel('info')
    
    print("🚀 Starting SPR+STP Experiment with Larger Topology 🚀")
    
    topo = LargerTopo()
    net = Mininet(topo=topo)
    
    net.start()
    
    print("Network started. Waiting for STP convergence...")
    time.sleep(15)  # Allow STP to converge
    
    # Get hosts for testing
    h1, h2, h3, h4, h5 = net.get('h1', 'h2', 'h3', 'h4', 'h5')
    
    print(f"\n=== Network Ready ===")
    print(f"h1: {h1.IP()}")
    print(f"h2: {h2.IP()}")
    print(f"h3: {h3.IP()}")
    print(f"h4: {h4.IP()}")
    print(f"h5: {h5.IP()}")
    
    # Basic connectivity tests
    print(f"\n=== Testing Connectivity ===")
    print("h1 -> h2:", "SUCCESS" if "0%" in h1.cmd(f'ping -c 3 {h2.IP()}') else "FAILED")
    print("h1 -> h3:", "SUCCESS" if "0%" in h1.cmd(f'ping -c 3 {h3.IP()}') else "FAILED")
    print("h4 -> h5:", "SUCCESS" if "0%" in h1.cmd(f'ping -c 3 {h5.IP()}') else "FAILED")
    
    print("\n=== Experiment Complete ===")
    print("Press Enter to stop network...")
    input()
    
    net.stop()

if __name__ == '__main__':
    run_experiment()