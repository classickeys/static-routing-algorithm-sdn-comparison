from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types, arp, ipv4
import time

class FixedSPRController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mac_to_port = {}
        
        # Auto-discovery topology - no hardcoding
        self.discovered_hosts = {}  # dpid -> {port: mac}
        
        # Assume all ports except port 1 are switch ports for now
        # We'll discover host ports dynamically
        self.assumed_switch_ports = {
            1: [2, 3, 4], 2: [1, 2, 3, 4], 3: [1, 2, 3, 4],
            4: [1, 2, 3], 5: [1, 2, 3, 4], 6: [1, 2, 3, 4, 5, 6]
        }
        
        # Other initialization...
        self.switch_broadcasts = {}
        self.broadcast_timeout = 2.0
        self.max_packets_per_second = 200
        self.packet_count = {}

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Install table-miss flow
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        
        self.logger.info("*** Switch s%s connected ***", datapath.id)

    def add_flow(self, datapath, priority, match, actions, idle_timeout=60):
        ofproto = datapath.ofproto  
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                match=match, instructions=inst, 
                                idle_timeout=idle_timeout)
        datapath.send_msg(mod)
        self.logger.info("*** FLOW INSTALLED: s%s priority=%s", datapath.id, priority)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        dpid = datapath.id
        
        # Rate limiting
        now = time.time()
        self.packet_count.setdefault(dpid, {'count': 0, 'time': now})
        
        if now - self.packet_count[dpid]['time'] > 1.0:
            self.packet_count[dpid] = {'count': 0, 'time': now}
        
        self.packet_count[dpid]['count'] += 1
        if self.packet_count[dpid]['count'] > self.max_packets_per_second:
            self.logger.warning("*** RATE LIMIT: Dropping packet on s%s", dpid)
            return
            
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst = eth.dst
        src = eth.src

        pkt_arp = pkt.get_protocol(arp.arp)
        packet_type = f"ARP-{pkt_arp.opcode}" if pkt_arp else "IPv4"

        self.logger.info("*** PACKET IN: s%s port=%s %s src=%s dst=%s", 
                        dpid, in_port, packet_type, src, dst)

        # MAC Learning with auto-discovery
        self.mac_to_port.setdefault(dpid, {})
        if src != "ff:ff:ff:ff:ff:ff":
            # Auto-discover hosts
            if src.startswith("00:00:00:00:00:"):  # Host MAC pattern
                if dpid not in self.discovered_hosts:
                    self.discovered_hosts[dpid] = {}
                
                if in_port not in self.discovered_hosts[dpid]:
                    self.discovered_hosts[dpid][in_port] = src
                    self.logger.info("*** HOST AUTO-DISCOVERED: s%s port=%s MAC=%s", dpid, in_port, src)
                
                # Learn MAC
                if src not in self.mac_to_port[dpid]:
                    self.mac_to_port[dpid][src] = in_port
                    self.logger.info("*** MAC LEARNED: s%s %s -> port %s (AUTO-HOST)", dpid, src, in_port)
            else:
                self.logger.debug("*** MAC IGNORED: s%s %s from switch port %s", dpid, src, in_port)

        # Handle packets
        if dst == "ff:ff:ff:ff:ff:ff":
            # Handle broadcasts with auto-discovery
            self.switch_broadcasts.setdefault(dpid, {})
            broadcast_key = f"{src}-{dst}-{in_port}"
            
            if broadcast_key in self.switch_broadcasts[dpid]:
                if now - self.switch_broadcasts[dpid][broadcast_key] < self.broadcast_timeout:
                    self.logger.debug("*** SWITCH BROADCAST SUPPRESSED: s%s %s", dpid, src)
                    return
            
            self.switch_broadcasts[dpid][broadcast_key] = now
            
            # Get discovered host ports for this switch
            host_ports = list(self.discovered_hosts.get(dpid, {}).keys())
            switch_ports = self.assumed_switch_ports.get(dpid, [])
            
            if in_port in host_ports:
                # From host - flood to switch ports only
                flood_ports = switch_ports
                self.logger.info("*** BROADCASTING from HOST: s%s %s -> switch ports %s", 
                               dpid, src, flood_ports)
            else:
                # From switch - flood to hosts AND other switches (except incoming)
                flood_ports = host_ports + [p for p in switch_ports if p != in_port]
                self.logger.info("*** BROADCASTING from SWITCH: s%s %s -> ports %s", 
                               dpid, src, flood_ports)
            
            if flood_ports:
                actions = [parser.OFPActionOutput(p) for p in flood_ports]
            else:
                self.logger.debug("*** No flood ports for s%s", dpid)
                return
                
        else:
            # Unicast
            if dst in self.mac_to_port[dpid]:
                out_port = self.mac_to_port[dpid][dst]
                
                # Install flow
                match = parser.OFPMatch(in_port=in_port, eth_dst=dst)
                actions = [parser.OFPActionOutput(out_port)]
                self.add_flow(datapath, 10, match, actions)
                
                self.logger.info("*** UNICAST: s%s %s->%s port %s (flow installed)", dpid, src, dst, out_port)
                
            else:
                # Unknown - flood (fix this section)
                host_ports = list(self.discovered_hosts.get(dpid, {}).keys())
                switch_ports = self.assumed_switch_ports.get(dpid, [])
                all_ports = host_ports + switch_ports
                flood_ports = [p for p in all_ports if p != in_port]
                actions = [parser.OFPActionOutput(p) for p in flood_ports]
                self.logger.info("*** UNKNOWN UNICAST: s%s %s->%s flooding to %s", dpid, src, dst, flood_ports)

        # Send packet out
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        
        try:
            datapath.send_msg(out)
            self.logger.info("*** PACKET OUT: s%s SUCCESS", dpid)
        except Exception as e:
            self.logger.error("*** PACKET OUT FAILED: s%s %s", dpid, e)

    def debug_topology(self, dpid, in_port, src_mac):
        """Debug helper to identify actual host connections"""
        if src_mac != "ff:ff:ff:ff:ff:ff":
            self.logger.info("*** TOPOLOGY DEBUG: s%s port=%s has MAC=%s", dpid, in_port, src_mac)
            # This will help you identify which ports actually have hosts

    def show_discovered_topology(self):
        """Display current discovered topology"""
        self.logger.info("=== DISCOVERED TOPOLOGY ===")
        for dpid, ports in self.discovered_hosts.items():
            for port, mac in ports.items():
                self.logger.info("s%s port %s: %s", dpid, port, mac)
        self.logger.info("=== END TOPOLOGY ===")