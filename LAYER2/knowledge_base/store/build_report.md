# Knowledge base build report

Built 2026-11-13 | passages.jsonl sha256 `6f6b4d5ed81e3069580eb9bdfa196995c82206a4921e87404fe9db687a192d80` | dataset `train_full.csv (Protocol B training split)` sha256 `64f48652b0f6b1b7...`

Validation: **passed** (every quote verbatim in its locked source, every data key present, coverage complete)


## 1. Sources

| id | tier | version | retrieved | files (sha256) |
|---|---|---|---|---|
| unb-ids2018 | dataset | page retrieved on the date in sources.lock.json | 2026-11-13 | page.html `c1258503f336` |
| cicflowmeter-readme | tool | git 98a5ebad0df579cc8b43eedd3421b3ae87699901 (master, 2022-01-05) | 2026-11-13 | ReadMe.txt `c8919171add3`; LICENSE.txt `f60c54b998ca`; FlowFeature.java `fadddf4ff18d` |
| cicflowmeter-src-2018 | tool | git 50d668db9bdb0a5fae457916dae4538ba8b41876 (2018-03-07) | 2026-11-13 | FlowFeature.java `941611e67a7a`; BasicFlow.java `7519720889fe`; BasicPacketInfo.java `eff86c02d123`; PacketReader.java `776a9234941c`; FlowGenerator.java `c760789268ba`; CICFlowMeter.java `9b74a25ae292` |
| attack-20.1 | knowledge | 20.1 | 2026-11-13 | enterprise-attack-20.1.json `539a843a7b53` |
| capec-3.9 | knowledge | 3.9 | 2026-11-13 | capec_v3.9.xml `70279a2dff0c` |
| iana-ports | standard | registry retrieved on the date in sources.lock.json | 2026-11-13 | service-names-port-numbers.csv `ee7416ab54a1` |
| iana-protocols | standard | registry retrieved on the date in sources.lock.json | 2026-11-13 | protocol-numbers-1.csv `ba37fe0a4e01` |
| rfc9293 | standard | RFC 9293 (2022) | 2026-11-13 | rfc9293.txt `6d9ac8be4b02` |
| rfc768 | standard | RFC 768 (1980) | 2026-11-13 | rfc768.txt `7dc8880e1ece` |
| rfc3168 | standard | RFC 3168 (2001) | 2026-11-13 | rfc3168.txt `35b437f1826b` |
| java8-hashmap | platform | Java SE 8 | 2026-11-13 | HashMap.html `8f718a56aa37` |
| java8-jls | platform | Java SE 8 | 2026-11-13 | jls-4.html `f99deeffe02e` |
| layer1-code | system | repository working tree at build time | 2026-11-13 | autoencoder.py `0321d6c1fd5a`; train_protocol_b.py `4f7df3652681` |

## 2. Passages per part

| part | passages | verbatim | rendered | authored | sent to VLM |
|---|---|---|---|---|---|
| A' label card | 14 | 0 | 0 | 14 | yes |
| B' mechanism (ATT&CK + CAPEC) | 13 | 13 | 0 | 0 | yes |
| H1 feature definitions | 81 | 1 | 76 | 4 | yes |
| H2 protocol / port / TCP flag | 21 | 9 | 12 | 0 | yes |
| H3 conditional extractor artefacts | 5 | 0 | 0 | 5 | yes |
| G' mitigations (code-filled, not sent to the VLM) | 19 | 0 | 19 | 0 | no |

## 3. Label mapping (the only author decisions)

| label | technique | previous profile | changed | CAPEC (route) | CAPEC entries CAPEC links to the technique |
|---|---|---|---|---|---|
| SSH_Bruteforce | T1110.001 Password Guessing | T1110.001 | no | CAPEC-49 (taxonomy_mapping) | CAPEC-49 |
| FTP_BruteForce | T1110.001 Password Guessing | T1110.001 | no | CAPEC-49 (taxonomy_mapping) | CAPEC-49 |
| DoS_attacks_GoldenEye | T1499.002 Service Exhaustion Flood | T1499.002 | no | CAPEC-488 (taxonomy_mapping) | CAPEC-469, CAPEC-482, CAPEC-488, CAPEC-489, CAPEC-528 |
| DoS_attacks_Hulk | T1499.002 Service Exhaustion Flood | T1499.002 | no | CAPEC-488 (taxonomy_mapping) | CAPEC-469, CAPEC-482, CAPEC-488, CAPEC-489, CAPEC-528 |
| DoS_attacks_SlowHTTPTest | T1499.002 Service Exhaustion Flood | T1499.002 | no | CAPEC-469 (taxonomy_mapping) | CAPEC-469, CAPEC-482, CAPEC-488, CAPEC-489, CAPEC-528 |
| DoS_attacks_Slowloris | T1499.002 Service Exhaustion Flood | T1499.002 | no | CAPEC-469 (taxonomy_mapping) | CAPEC-469, CAPEC-482, CAPEC-488, CAPEC-489, CAPEC-528 |
| DDoS_attacks_LOIC_HTTP | T1499.002 Service Exhaustion Flood | T1499.002 | no | CAPEC-488 (taxonomy_mapping) | CAPEC-469, CAPEC-482, CAPEC-488, CAPEC-489, CAPEC-528 |
| DDOS_attack_LOIC_UDP | T1498.001 Direct Network Flood | T1499.002 | **yes** | CAPEC-125 (taxonomy_mapping) | CAPEC-125, CAPEC-482, CAPEC-528, CAPEC-666 |
| DDOS_attack_HOIC | T1499.002 Service Exhaustion Flood | T1499.002 | no | CAPEC-488 (taxonomy_mapping) | CAPEC-469, CAPEC-482, CAPEC-488, CAPEC-489, CAPEC-528 |
| Brute_Force_Web | T1110.001 Password Guessing | T1110.001 | no | CAPEC-49 (taxonomy_mapping) | CAPEC-49 |
| Brute_Force_XSS | T1190 Exploit Public-Facing Application | T1189 | **yes** | CAPEC-63 (description) | none |
| SQL_Injection | T1190 Exploit Public-Facing Application | T1190 | no | CAPEC-66 (description) | none |
| Infiltration | T1046 Network Service Discovery | T1204.002 | **yes** | CAPEC-300 (taxonomy_mapping) | CAPEC-300 |
| Bot | T1071.001 Web Protocols | T1583.005 | **yes** | - (none) | none |

Rationale for each technique and CAPEC choice:

- **SSH_Bruteforce** — The scenario guesses account passwords from a dictionary; Password Guessing is the ATT&CK sub-technique for guessing passwords from a list. Same as the previous mapping. CAPEC: 
- **FTP_BruteForce** — Same scenario as SSH_Bruteforce on the FTP service. Same as the previous mapping. CAPEC: 
- **DoS_attacks_GoldenEye** — Denial of service against a web service (UNB: DoS attack on Apache; data: all flows to port 80). Same as the previous mapping. CAPEC: Of the CAPEC entries linked to T1499.002 (469, 482, 488, 489, 528), HTTP Flood matches a DoS on a web server reached on port 80; the dataset page does not describe a slow-request attack for this tool.
- **DoS_attacks_Hulk** — Denial of service against a web service (UNB: DoS attack on Apache; data: all flows to port 80). Same as the previous mapping. CAPEC: Same choice as DoS_attacks_GoldenEye.
- **DoS_attacks_SlowHTTPTest** — Denial of service against a web service per the dataset page. Same as the previous mapping. The flows of this label all go to port 21 with no payload (data_facts), which does not match the scenario; this is reported as a dataset limitation, not in the KB. CAPEC: Of the CAPEC entries linked to T1499.002, HTTP DoS describes holding HTTP sessions open, which is what a slow-HTTP test tool does; the dataset page gives only the tool name.
- **DoS_attacks_Slowloris** — Exhausts the connection capacity of a web service. Same as the previous mapping. CAPEC: HTTP DoS (keeping HTTP sessions alive) is the CAPEC entry linked to T1499.002 that matches the scenario's description of Slowloris.
- **DDoS_attacks_LOIC_HTTP** — HTTP requests flooding a web service. Same as the previous mapping. CAPEC: The label names HTTP requests; HTTP Flood is the CAPEC entry linked to T1499.002 for floods of HTTP requests.
- **DDOS_attack_LOIC_UDP** — All flows of this label are UDP (data), i.e. a flood of network traffic rather than requests to a service's application logic. Direct Network Flood names UDP floods. CHANGED from the previous mapping T1499.002 (Service Exhaustion Flood), which describes attacks on the services themselves. CAPEC: Of the CAPEC entries linked to T1498.001 (125, 482, 528, 666), Flooding is the general one; 482 is TCP-specific, 528 is XML-specific and 666 is Bluetooth-specific.
- **DDOS_attack_HOIC** — The tool attacks URLs, i.e. a web service. Same as the previous mapping. CAPEC: Attacking URLs is an HTTP-level flood; HTTP Flood is the CAPEC entry linked to T1499.002 for it.
- **Brute_Force_Web** — Brute force against a web login guesses passwords. Same as the previous mapping. CAPEC: 
- **Brute_Force_XSS** — The captured flows are the attacker's automated requests to the vulnerable web application, i.e. an attempt to exploit a public-facing application. Drive-by Compromise (T1189) describes a user's browser visiting a compromised site, which is not the traffic captured in this scenario. CHANGED from the previous mapping T1189. SQL injection and XSS share T1190 and are told apart by their CAPEC entries. CAPEC: CAPEC v3.9 links no attack pattern to T1190; CAPEC-63 Cross-Site Scripting is chosen because it describes the attack named by the label.
- **SQL_Injection** — SQL injection exploits a weakness of a public-facing web application. Same as the previous mapping. CAPEC: CAPEC v3.9 links no attack pattern to T1190; CAPEC-66 SQL Injection is chosen because it describes the attack named by the label.
- **Infiltration** — Of the scenario's stages, the internal scan is the one carried by network flows to many hosts and ports (data: no destination port above 30% of the label's flows); the e-mail delivery and the exploitation run on the victim host. CHANGED from the previous mapping T1204.002 + T1046: only the observable stage is kept, because the report must describe the flow. CAPEC: Port Scanning is the CAPEC entry linked to T1046.
- **Bot** — The network-visible part of the scenario is the controller's commands and the infected hosts' replies; 97% of the label's flows use port 8080, registered by IANA as HTTP Alternate. CHANGED from the previous mapping T1583.005 (Botnet) + T1113 (Screen Capture): the first describes the attacker acquiring infrastructure and the second an action on the host, neither of which is observable in a flow. CAPEC: CAPEC v3.9 links no attack pattern to T1071.001 and has no entry for botnet command and control; no CAPEC passage.

## 4. Every author-written sentence next to its evidence

Code checks that each quote exists verbatim; a reviewer checks that the sentence says no more than its evidence.

### A-SSH_Bruteforce

> Class SSH_Bruteforce of CSE-CIC-IDS2018: Dictionary password guessing against the SSH service of the victim server. ATT&CK technique: T1110.001 Password Guessing (tactic: credential access).

- [unb-ids2018 | 1. Introduction, Brute force attacks] "by running a dictionary brute force attack against the main server."
- [data_facts | by_label.SSH_Bruteforce.top_ports.22] = 1.0
- [decision T1110.001] The scenario guesses account passwords from a dictionary; Password Guessing is the ATT&CK sub-technique for guessing passwords from a list. Same as the previous mapping.

### A-FTP_BruteForce

> Class FTP_BruteForce of CSE-CIC-IDS2018: Dictionary password guessing against the FTP service of the victim server. ATT&CK technique: T1110.001 Password Guessing (tactic: credential access).

- [unb-ids2018 | 2.2.1 Brute-force attack] "For a list of passwords, we use a large dictionary that contains 90 million words."
- [data_facts | by_label.FTP_BruteForce.top_ports.21] = 1.0
- [decision T1110.001] Same scenario as SSH_Bruteforce on the FTP service. Same as the previous mapping.

### A-DoS_attacks_GoldenEye

> Class DoS_attacks_GoldenEye of CSE-CIC-IDS2018: Denial-of-service attack on the victim's Apache web server. ATT&CK technique: T1499.002 Service Exhaustion Flood (tactic: impact).

- [unb-ids2018 | Table 1] "DoS attack Hulk, GoldenEye, Slowloris, Slowhttptest One day Kali linux Ubuntu 16.4 (Apache)"
- [data_facts | by_label.DoS_attacks_GoldenEye.top_ports.80] = 1.0
- [decision T1499.002] Denial of service against a web service (UNB: DoS attack on Apache; data: all flows to port 80). Same as the previous mapping.

### A-DoS_attacks_Hulk

> Class DoS_attacks_Hulk of CSE-CIC-IDS2018: Denial-of-service attack on the victim's Apache web server. ATT&CK technique: T1499.002 Service Exhaustion Flood (tactic: impact).

- [unb-ids2018 | Table 1] "DoS attack Hulk, GoldenEye, Slowloris, Slowhttptest One day Kali linux Ubuntu 16.4 (Apache)"
- [data_facts | by_label.DoS_attacks_Hulk.top_ports.80] = 1.0
- [decision T1499.002] Denial of service against a web service (UNB: DoS attack on Apache; data: all flows to port 80). Same as the previous mapping.

### A-DoS_attacks_SlowHTTPTest

> Class DoS_attacks_SlowHTTPTest of CSE-CIC-IDS2018: Denial-of-service attack on the victim's Apache web server. ATT&CK technique: T1499.002 Service Exhaustion Flood (tactic: impact).

- [unb-ids2018 | Table 1] "DoS attack Hulk, GoldenEye, Slowloris, Slowhttptest One day Kali linux Ubuntu 16.4 (Apache)"
- [decision T1499.002] Denial of service against a web service per the dataset page. Same as the previous mapping. The flows of this label all go to port 21 with no payload (data_facts), which does not match the scenario; this is reported as a dataset limitation, not in the KB.

### A-DoS_attacks_Slowloris

> Class DoS_attacks_Slowloris of CSE-CIC-IDS2018: Slow HTTP denial of service: connections to the victim web server are held open by sending valid but incomplete HTTP requests at regular intervals, until the server has no free connections. ATT&CK technique: T1499.002 Service Exhaustion Flood (tactic: impact).

- [unb-ids2018 | 1. Introduction, HTTP denial of service] "The tool holds the connection open by sending valid, incomplete HTTP requests to the server at regular intervals to keep the sockets from closing. Since any Web server has a finite ability to serve connections, it will only be a matter of time before all sockets are used up and no other connection can be made."
- [data_facts | by_label.DoS_attacks_Slowloris.top_ports.80] = 1.0
- [decision T1499.002] Exhausts the connection capacity of a web service. Same as the previous mapping.

### A-DDoS_attacks_LOIC_HTTP

> Class DDoS_attacks_LOIC_HTTP of CSE-CIC-IDS2018: Distributed denial of service with HTTP requests against the victim web server. ATT&CK technique: T1499.002 Service Exhaustion Flood (tactic: impact).

- [unb-ids2018 | Table 1] "Low Orbit Ion Canon (LOIC) for UDP, TCP, or HTTP requests"
- [data_facts | by_label.DDoS_attacks_LOIC_HTTP.top_ports.80] = 1.0
- [decision T1499.002] HTTP requests flooding a web service. Same as the previous mapping.

### A-DDOS_attack_LOIC_UDP

> Class DDOS_attack_LOIC_UDP of CSE-CIC-IDS2018: Distributed denial of service with UDP traffic sent to the victim server. ATT&CK technique: T1498.001 Direct Network Flood (tactic: impact).

- [unb-ids2018 | Table 1] "Low Orbit Ion Canon (LOIC) for UDP, TCP, or HTTP requests"
- [data_facts | by_label.DDOS_attack_LOIC_UDP.udp] = 1.0
- [decision T1498.001] All flows of this label are UDP (data), i.e. a flood of network traffic rather than requests to a service's application logic. Direct Network Flood names UDP floods. CHANGED from the previous mapping T1499.002 (Service Exhaustion Flood), which describes attacks on the services themselves.

### A-DDOS_attack_HOIC

> Class DDOS_attack_HOIC of CSE-CIC-IDS2018: Distributed denial of service with HTTP requests against the victim web server, sent from several attacking computers. ATT&CK technique: T1499.002 Service Exhaustion Flood (tactic: impact).

- [unb-ids2018 | 2.2.5 Distributed Denial-of-Service] "designed to attack as many as 256 URLs at the same time."
- [unb-ids2018 | 2.2.5 Distributed Denial-of-Service] "In this scenario, we use free HOIC tool to conduct DDoS attack by using 4 different computers."
- [data_facts | by_label.DDOS_attack_HOIC.top_ports.80] = 1.0
- [decision T1499.002] The tool attacks URLs, i.e. a web service. Same as the previous mapping.

### A-Brute_Force_Web

> Class Brute_Force_Web of CSE-CIC-IDS2018: Automated password guessing against a deliberately vulnerable web application (DVWA) on the victim web server. ATT&CK technique: T1110.001 Password Guessing (tactic: credential access).

- [unb-ids2018 | 2.2.6 Web Attacks] "DVWA is a PHP/MySQL web application that is vulnerable."
- [unb-ids2018 | 2.2.6 Web Attacks] "To automate the attacks in XSS and Brute-force section we developed an automation code with Selenium framework."
- [data_facts | by_label.Brute_Force_Web.top_ports.80] = 0.7455
- [decision T1110.001] Brute force against a web login guesses passwords. Same as the previous mapping.

### A-Brute_Force_XSS

> Class Brute_Force_XSS of CSE-CIC-IDS2018: Automated cross-site scripting (XSS) attempts against a deliberately vulnerable web application (DVWA) on the victim web server. ATT&CK technique: T1190 Exploit Public-Facing Application (tactic: initial access).

- [unb-ids2018 | 2.2.6 Web Attacks] "DVWA is a PHP/MySQL web application that is vulnerable."
- [unb-ids2018 | 2.2.6 Web Attacks] "To automate the attacks in XSS and Brute-force section we developed an automation code with Selenium framework."
- [data_facts | by_label.Brute_Force_XSS.top_ports.80] = 0.9835
- [decision T1190] The captured flows are the attacker's automated requests to the vulnerable web application, i.e. an attempt to exploit a public-facing application. Drive-by Compromise (T1189) describes a user's browser visiting a compromised site, which is not the traffic captured in this scenario. CHANGED from the previous mapping T1189. SQL injection and XSS share T1190 and are told apart by their CAPEC entries.

### A-SQL_Injection

> Class SQL_Injection of CSE-CIC-IDS2018: SQL injection attempts against a deliberately vulnerable web application (DVWA) on the victim web server. ATT&CK technique: T1190 Exploit Public-Facing Application (tactic: initial access).

- [unb-ids2018 | 1. Introduction, Collection of web application attacks] "we conduct different types of web attacks on the vulnerable website, including SQL injection, command injection, and unrestricted file upload."
- [unb-ids2018 | 2.2.6 Web Attacks] "DVWA is a PHP/MySQL web application that is vulnerable."
- [data_facts | by_label.SQL_Injection.top_ports.80] = 1.0
- [decision T1190] SQL injection exploits a weakness of a public-facing web application. Same as the previous mapping.

### A-Infiltration

> Class Infiltration of CSE-CIC-IDS2018: Reconnaissance of the internal network from a compromised victim host: IP sweep, port scan and service enumeration, after the host was compromised through a malicious e-mail attachment. ATT&CK technique: T1046 Network Service Discovery (tactic: discovery).

- [unb-ids2018 | 2.2.7 Infiltration of the network from inside] "IP sweep, full port scan and service enumerations using Nmap."
- [unb-ids2018 | 1. Introduction, Infiltration of the network from inside] "In this scenario, we send a malicious file via an email to the victim and exploit an application vulnerability."
- [data_facts | by_label.Infiltration.top_ports] = {"53": 0.2917, "443": 0.2029, "3389": 0.0963, "445": 0.0396, "80": 0.0395}
- [decision T1046] Of the scenario's stages, the internal scan is the one carried by network flows to many hosts and ports (data: no destination port above 30% of the label's flows); the e-mail delivery and the exploitation run on the victim host. CHANGED from the previous mapping T1204.002 + T1046: only the observable stage is kept, because the report must describe the flow.

### A-Bot

> Class Bot of CSE-CIC-IDS2018: Botnet command-and-control traffic between infected hosts and their controller: remote shell, file transfer and periodic screenshot requests. ATT&CK technique: T1071.001 Web Protocols (tactic: command and control).

- [unb-ids2018 | 2.2.3 Botnet] "In this scenario, we infect machines with two different botnets (Zeus and Ares), also every 400 seconds we request screenshots from the zombies."
- [unb-ids2018 | Table 1] "Ares (developed by Python): remote shell, file upload/download, capturing screenshots and key logging"
- [data_facts | by_label.Bot.top_ports.8080] = 0.9667
- [decision T1071.001] The network-visible part of the scenario is the controller's commands and the infected hosts' replies; 97% of the label's flows use port 8080, registered by IANA as HTTP Alternate. CHANGED from the previous mapping T1583.005 (Botnet) + T1113 (Screen Capture): the first describes the attacker acquiring infrastructure and the second an action on the host, neither of which is observable in a flow.

### H1-Dst_Port

> Destination port of the flow: the TCP or UDP port of the destination host, read from the transport header of the flow's packets.

- [unb-ids2018 | 3. Feature extraction] "six columns labeled for each flow, namely FlowID, SourceIP, DestinationIP, SourcePort, DestinationPort, and Protocol"
- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setDstPort(tcp.destination());"
- [cicflowmeter-src-2018 | PacketReader.java, UDP branch] "packetInfo.setDstPort(udp.destination());"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //5] "dst_pot("Dst Port","DPT""

### H1-Protocol

> Transport protocol of the flow as an IP protocol number: the extractor writes 6 for TCP, 17 for UDP, and 0 when neither header was found.

- [unb-ids2018 | 3. Feature extraction] "six columns labeled for each flow, namely FlowID, SourceIP, DestinationIP, SourcePort, DestinationPort, and Protocol"
- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setProtocol(6);"
- [cicflowmeter-src-2018 | PacketReader.java, UDP branch] "packetInfo.setProtocol(17);"
- [java8-jls | Section 4.12.5, Initial Values of Variables] "For type int, the default value is zero, that is, 0."
- [cicflowmeter-src-2018 | FlowFeature.java, feature //6] "prot("Protocol","PROT""

### H1-ae_recon_error

> Reconstruction error of an autoencoder trained on benign flows only: the mean squared difference between the flow's scaled features and the autoencoder's reconstruction of them. A higher value means the flow is less like the benign training traffic, and the gate's XGBoost model is constrained so that a higher value can only raise the attack score.

- [layer1-code | autoencoder.py, module docstring] "Train tren benign-only. Reconstruction error = anomaly score."
- [layer1-code | autoencoder.py, reconstruction_error()] "return ((x - x_hat) ** 2).mean(dim=1)"
- [layer1-code | train_protocol_b.py, scaling] "return np.clip(sc.transform(clean(X)), -CLIP_VAL, CLIP_VAL)"
- [layer1-code | train_protocol_b.py, AE features] "names = feats + [f"ae_latent_{i}" for i in range(AE_BOTTLENECK)] + ["ae_recon_error"]"
- [layer1-code | train_protocol_b.py, monotone constraints] "mono = build_monotone_constraints(n_base=len(feats), n_augmented_monotone=1, n_augmented_free=AE_BOTTLENECK)"

### H1-ae_latent

> One of the 16 coordinates of the autoencoder's bottleneck (latent) representation of the flow's scaled features. The coordinates are learned from benign flows; individually they have no network meaning and are model inputs, not traffic measurements.

- [layer1-code | autoencoder.py, module docstring] "Encoder: input_dim -> 64 -> 32 -> bottleneck_dim"
- [layer1-code | train_protocol_b.py, configuration] "AE_BOTTLENECK = 16"
- [layer1-code | train_protocol_b.py, AE features] "names = feats + [f"ae_latent_{i}" for i in range(AE_BOTTLENECK)] + ["ae_recon_error"]"

### H3-NOPAYLOAD

> This flow carries no transport-layer payload in either direction, so application content (for example an HTTP request or an SQL statement) cannot be observed in it. The extractor ends a TCP flow at the first FIN packet, and a packet that arrives when no flow is open starts a new flow, so the last packets of a connection can form a flow of their own.

- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setPayloadBytes(tcp.getPayloadLength());"
- [cicflowmeter-src-2018 | FlowGenerator.java, addPacket()] "// Flow finished due FIN flag (tcp only):"
- [cicflowmeter-src-2018 | FlowGenerator.java, addPacket()] "}else if(packet.hasFlagFIN()){"
- [cicflowmeter-src-2018 | FlowGenerator.java, addPacket(), packet without an open flow] "currentFlows.put(packet.getFlowId(), new BasicFlow(bidirectional,packet));"
- [data_facts | by_label.Benign.zero_payload] = 0.2286
- [data_facts | by_label.DoS_attacks_Hulk.zero_payload] = 0.8266

### H3-NOWIN

> A value of -1 in an initial-window column is not a window size: the TCP window field is a 16-bit count of octets. It is the extractor's default, kept when no TCP window was read for that direction, for example in a UDP flow or when no packet was seen in that direction.

- [rfc9293 | Section 3.1, Header Format] "Window: 16 bits The number of data octets beginning with the one indicated in the acknowledgment field that the sender of this segment is willing to accept."
- [cicflowmeter-src-2018 | BasicPacketInfo.java, field declaration] "private int TCPWindow=-1;"
- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setTCPWindow(tcp.window());"
- [data_facts | init_win_minus1.bwd_zero_bwd_pkts_or_udp_share] = 0.9984

### H3-TIMEOUT

> The flow lasted until the extractor's 120 s flow timeout. A longer connection is split into several consecutive flows, so this duration does not cover the whole session.

- [cicflowmeter-src-2018 | CICFlowMeter.java, main()] "flowGen = new FlowGenerator(true,120000000L, 5000000L);"
- [cicflowmeter-src-2018 | FlowGenerator.java, addPacket()] "// Flow finished due flowtimeout: // 1.- we move the flow to finished flow list // 2.- we eliminate the flow from the current flow list // 3.- we create a new flow with the packet-in-process"
- [data_facts | flow_duration.max] = 119999998
- [data_facts | by_label.DDOS_attack_LOIC_UDP.duration_ge_119s] = 0.94

### H3-UDP

> This is a UDP flow. The extractor reads flags and the window only from TCP headers, so for UDP the flag columns stay 0 and the initial-window columns stay -1; they carry no information about this flow.

- [cicflowmeter-readme | ReadMe.txt, list of extracted features] "Fwd PSH flags Number of times the PSH flag was set in packets travelling in the forward direction (0 for UDP)"
- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setFlagRST(tcp.flags_RST());"
- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setTCPWindow(tcp.window());"
- [data_facts | by_label.DDOS_attack_LOIC_UDP.init_fwd_win_minus1] = 1.0

### H3-NOTRANSPORT

> Protocol 0 here means the packets carried neither a TCP nor a UDP header: the extractor sets the protocol only to 6 (TCP) or 17 (UDP) and otherwise keeps the Java default value 0, and ports and payload are then also 0. It does not refer to IANA protocol number 0 (HOPOPT).

- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setProtocol(6);"
- [cicflowmeter-src-2018 | PacketReader.java, UDP branch] "packetInfo.setProtocol(17);"
- [cicflowmeter-src-2018 | BasicPacketInfo.java, field declarations] "private int protocol;"
- [java8-jls | Section 4.12.5, Initial Values of Variables] "For type int, the default value is zero, that is, 0."
- [iana-protocols | protocol-numbers-1.csv, row 0] "0,HOPOPT,IPv6 Hop-by-Hop Option"
- [data_facts | protocol_0.dst_port_0_share] = 1.0
- [data_facts | protocol_0.zero_payload_share] = 1.0

### Notes attached to H1 passages

**XN-TIME-MICROSECONDS** (on: Flow Duration, Flow IAT Mean, Flow IAT Std, Flow IAT Max, Flow IAT Min, Fwd IAT Tot, Fwd IAT Mean, Fwd IAT Std, Fwd IAT Max, Fwd IAT Min, Bwd IAT Tot, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Bwd IAT Min, Active Mean, Active Std, Active Max, Active Min, Idle Mean, Idle Std, Idle Max, Idle Min)

> Times are in microseconds: the extractor reads packet timestamps in microseconds.

- [cicflowmeter-src-2018 | PacketReader.java, IPv4 packet parsing] "packetInfo.setTimeStamp(packet.getCaptureHeader().timestampInMicros());"
- [cicflowmeter-readme | ReadMe.txt, list of extracted features] "Flow duration Duration of the flow in Microsecond"

**XN-FLOW-TIMEOUT** (on: Flow Duration)

> The command-line extractor closes a flow 120 s (120,000,000 microseconds) after its first packet and starts a new flow, so no duration in this dataset exceeds that limit.

- [cicflowmeter-src-2018 | CICFlowMeter.java, main()] "flowGen = new FlowGenerator(true,120000000L, 5000000L);"
- [cicflowmeter-src-2018 | FlowGenerator.java, addPacket()] "if((currentTimestamp -flow.getFlowStartTime())>flowTimeOut){"
- [data_facts | flow_duration.max] = 119999998

**DATA-IDENTICAL** (on: Tot Fwd Pkts)

> In this dataset this column is identical to 'Subflow Fwd Pkts' in every row.

- [data_facts | identical_with.Tot Fwd Pkts] = ["Subflow Fwd Pkts"]

**DATA-IDENTICAL** (on: Tot Bwd Pkts)

> In this dataset this column is identical to 'Subflow Bwd Pkts' in every row.

- [data_facts | identical_with.Tot Bwd Pkts] = ["Subflow Bwd Pkts"]

**XN-PAYLOAD-BYTES** (on: TotLen Fwd Pkts, TotLen Bwd Pkts, Fwd Pkt Len Max, Fwd Pkt Len Min, Fwd Pkt Len Mean, Fwd Pkt Len Std, Bwd Pkt Len Max, Bwd Pkt Len Min, Bwd Pkt Len Mean, Bwd Pkt Len Std, Pkt Len Min, Pkt Len Max, Pkt Len Mean, Pkt Len Std, Pkt Len Var, Pkt Size Avg, Fwd Seg Size Avg, Bwd Seg Size Avg, Subflow Fwd Byts, Subflow Bwd Byts)

> Lengths count transport-layer payload bytes (the TCP or UDP payload), not packet headers.

- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setPayloadBytes(tcp.getPayloadLength());"
- [cicflowmeter-src-2018 | PacketReader.java, UDP branch] "packetInfo.setPayloadBytes(udp.getPayloadLength());"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "this.fwdPktStats.addValue((double)packet.getPayloadBytes());"
- [cicflowmeter-src-2018 | BasicFlow.java, getAvgPacketSize()] "return (this.flowLengthStats.getSum()/this.packetCount());"
- [cicflowmeter-src-2018 | BasicFlow.java, fAvgSegmentSize()] "return (this.fwdPktStats.getSum() / (double)this.forward.size());"

**DATA-IDENTICAL** (on: TotLen Fwd Pkts)

> In this dataset this column is identical to 'Subflow Fwd Byts' in every row.

- [data_facts | identical_with.TotLen Fwd Pkts] = ["Subflow Fwd Byts"]

**DATA-NEAR-IDENTICAL** (on: TotLen Bwd Pkts)

> In this dataset this column equals 'Subflow Bwd Byts' in at least 99.9% of rows.

- [data_facts | near_identical_with.TotLen Bwd Pkts] = ["Subflow Bwd Byts"]

**XN-FLAGS-FIRST-PACKET** (on: Fwd PSH Flags, Fwd URG Flags, FIN Flag Cnt, SYN Flag Cnt, RST Flag Cnt, PSH Flag Cnt, ACK Flag Cnt, URG Flag Cnt, CWE Flag Count, ECE Flag Cnt)

> In the extractor version that produced this dataset, flags are counted only for the first packet of the flow (the count is disabled for later packets), so the value is 0 or 1 and describes that first packet only.

- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "public void firstPacket(BasicPacketInfo packet){ updateFlowBulk(packet); detectUpdateSubflows(packet); checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), forward branch] "if(packet.hasFlagPSH()){ this.fPSH_cnt++; } if(packet.hasFlagURG()){ this.fURG_cnt++; }"
- [data_facts | flag_values] = {"FIN Flag Cnt": [0.0, 1.0], "SYN Flag Cnt": [0.0, 1.0], "RST Flag Cnt": [0.0, 1.0], "PSH Flag Cnt": [0.0, 1.0], "ACK Flag Cnt": [0.0, 1.0], "URG Flag Cnt": [0.0, 1.0], "CWE Flag Count": [0.0, 1.0], "ECE Flag Cnt": [0.0, 1.0], "Fwd PSH Flags": [0.0, 1.0], "Bwd PSH Flags": [0.0], "Fwd URG Flags": [0.0, 1.0], "Bwd URG Flags": [0.0]}

**DATA-IDENTICAL** (on: Fwd PSH Flags)

> In this dataset this column is identical to 'SYN Flag Cnt' in every row.

- [data_facts | identical_with.Fwd PSH Flags] = ["SYN Flag Cnt"]

**XN-BWD-FLAGS-ZERO** (on: Bwd PSH Flags, Bwd URG Flags)

> Always 0 in this dataset: the backward PSH and URG counters are incremented only in the first-packet routine, and the first packet always defines the forward direction.

- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "if(this.src==null){ this.src = packet.getSrc();"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), backward branch] "if(packet.hasFlagPSH()){ this.bPSH_cnt++; } if(packet.hasFlagURG()){ this.bURG_cnt++; }"
- [data_facts | constant_columns.Bwd PSH Flags] = 0.0
- [data_facts | constant_columns.Bwd URG Flags] = 0.0

**DATA-IDENTICAL** (on: Fwd URG Flags)

> In this dataset this column is identical to 'CWE Flag Count' in every row.

- [data_facts | identical_with.Fwd URG Flags] = ["CWE Flag Count"]

**XN-FLAG-ORDER** (on: FIN Flag Cnt)

> In this dataset the column named 'FIN Flag Cnt' holds the RST flag of the first packet: the extractor writes the eight flag counts in the iteration order of a Java HashMap, which does not follow the order of the column names.

- [cicflowmeter-src-2018 | BasicFlow.java, field declaration] "private HashMap<String, MutableInt> flagCounts;"
- [cicflowmeter-src-2018 | BasicFlow.java, initFlags()] "flagCounts.put("FIN", new MutableInt()); flagCounts.put("SYN", new MutableInt()); flagCounts.put("RST", new MutableInt()); flagCounts.put("PSH", new MutableInt()); flagCounts.put("ACK", new MutableInt()); flagCounts.put("URG", new MutableInt()); flagCounts.put("CWR", new MutableInt()); flagCounts.put("ECE", new MutableInt());"
- [cicflowmeter-src-2018 | BasicFlow.java, CSV row writer] "for(String key: flagCounts.keySet()){ dump.append(flagCounts.get(key).value).append(separator); //50,51,52,53,54,55,56,57"
- [java8-hashmap | class description] "This class makes no guarantees as to the order of the map; in particular, it does not guarantee that the order will remain constant over time."
- [data_facts | flag_column_holds] = {"FIN Flag Cnt": "RST", "SYN Flag Cnt": "PSH", "RST Flag Cnt": "ECE", "PSH Flag Cnt": "SYN", "ACK Flag Cnt": "ACK", "URG Flag Cnt": "FIN", "CWE Flag Count": "URG", "ECE Flag Cnt": "CWR"}
- [data_facts | flag_consistency.SYN_Flag_Cnt_equals_Fwd_PSH_Flags] = 1.0
- [data_facts | flag_consistency.CWE_Flag_Count_equals_Fwd_URG_Flags] = 1.0
- [data_facts | flag_consistency.RST_Flag_Cnt_equals_ECE_Flag_Cnt] = 1.0

**XN-FLAG-ORDER** (on: SYN Flag Cnt)

> In this dataset the column named 'SYN Flag Cnt' holds the PSH flag of the first packet: the extractor writes the eight flag counts in the iteration order of a Java HashMap, which does not follow the order of the column names.

- [cicflowmeter-src-2018 | BasicFlow.java, field declaration] "private HashMap<String, MutableInt> flagCounts;"
- [cicflowmeter-src-2018 | BasicFlow.java, initFlags()] "flagCounts.put("FIN", new MutableInt()); flagCounts.put("SYN", new MutableInt()); flagCounts.put("RST", new MutableInt()); flagCounts.put("PSH", new MutableInt()); flagCounts.put("ACK", new MutableInt()); flagCounts.put("URG", new MutableInt()); flagCounts.put("CWR", new MutableInt()); flagCounts.put("ECE", new MutableInt());"
- [cicflowmeter-src-2018 | BasicFlow.java, CSV row writer] "for(String key: flagCounts.keySet()){ dump.append(flagCounts.get(key).value).append(separator); //50,51,52,53,54,55,56,57"
- [java8-hashmap | class description] "This class makes no guarantees as to the order of the map; in particular, it does not guarantee that the order will remain constant over time."
- [data_facts | flag_column_holds] = {"FIN Flag Cnt": "RST", "SYN Flag Cnt": "PSH", "RST Flag Cnt": "ECE", "PSH Flag Cnt": "SYN", "ACK Flag Cnt": "ACK", "URG Flag Cnt": "FIN", "CWE Flag Count": "URG", "ECE Flag Cnt": "CWR"}
- [data_facts | flag_consistency.SYN_Flag_Cnt_equals_Fwd_PSH_Flags] = 1.0
- [data_facts | flag_consistency.CWE_Flag_Count_equals_Fwd_URG_Flags] = 1.0
- [data_facts | flag_consistency.RST_Flag_Cnt_equals_ECE_Flag_Cnt] = 1.0

**DATA-IDENTICAL** (on: SYN Flag Cnt)

> In this dataset this column is identical to 'Fwd PSH Flags' in every row.

- [data_facts | identical_with.SYN Flag Cnt] = ["Fwd PSH Flags"]

**XN-FLAG-ORDER** (on: RST Flag Cnt)

> In this dataset the column named 'RST Flag Cnt' holds the ECE flag of the first packet: the extractor writes the eight flag counts in the iteration order of a Java HashMap, which does not follow the order of the column names.

- [cicflowmeter-src-2018 | BasicFlow.java, field declaration] "private HashMap<String, MutableInt> flagCounts;"
- [cicflowmeter-src-2018 | BasicFlow.java, initFlags()] "flagCounts.put("FIN", new MutableInt()); flagCounts.put("SYN", new MutableInt()); flagCounts.put("RST", new MutableInt()); flagCounts.put("PSH", new MutableInt()); flagCounts.put("ACK", new MutableInt()); flagCounts.put("URG", new MutableInt()); flagCounts.put("CWR", new MutableInt()); flagCounts.put("ECE", new MutableInt());"
- [cicflowmeter-src-2018 | BasicFlow.java, CSV row writer] "for(String key: flagCounts.keySet()){ dump.append(flagCounts.get(key).value).append(separator); //50,51,52,53,54,55,56,57"
- [java8-hashmap | class description] "This class makes no guarantees as to the order of the map; in particular, it does not guarantee that the order will remain constant over time."
- [data_facts | flag_column_holds] = {"FIN Flag Cnt": "RST", "SYN Flag Cnt": "PSH", "RST Flag Cnt": "ECE", "PSH Flag Cnt": "SYN", "ACK Flag Cnt": "ACK", "URG Flag Cnt": "FIN", "CWE Flag Count": "URG", "ECE Flag Cnt": "CWR"}
- [data_facts | flag_consistency.SYN_Flag_Cnt_equals_Fwd_PSH_Flags] = 1.0
- [data_facts | flag_consistency.CWE_Flag_Count_equals_Fwd_URG_Flags] = 1.0
- [data_facts | flag_consistency.RST_Flag_Cnt_equals_ECE_Flag_Cnt] = 1.0

**DATA-NEAR-IDENTICAL** (on: RST Flag Cnt)

> In this dataset this column equals 'ECE Flag Cnt' in at least 99.9% of rows.

- [data_facts | near_identical_with.RST Flag Cnt] = ["ECE Flag Cnt"]

**XN-FLAG-ORDER** (on: PSH Flag Cnt)

> In this dataset the column named 'PSH Flag Cnt' holds the SYN flag of the first packet: the extractor writes the eight flag counts in the iteration order of a Java HashMap, which does not follow the order of the column names.

- [cicflowmeter-src-2018 | BasicFlow.java, field declaration] "private HashMap<String, MutableInt> flagCounts;"
- [cicflowmeter-src-2018 | BasicFlow.java, initFlags()] "flagCounts.put("FIN", new MutableInt()); flagCounts.put("SYN", new MutableInt()); flagCounts.put("RST", new MutableInt()); flagCounts.put("PSH", new MutableInt()); flagCounts.put("ACK", new MutableInt()); flagCounts.put("URG", new MutableInt()); flagCounts.put("CWR", new MutableInt()); flagCounts.put("ECE", new MutableInt());"
- [cicflowmeter-src-2018 | BasicFlow.java, CSV row writer] "for(String key: flagCounts.keySet()){ dump.append(flagCounts.get(key).value).append(separator); //50,51,52,53,54,55,56,57"
- [java8-hashmap | class description] "This class makes no guarantees as to the order of the map; in particular, it does not guarantee that the order will remain constant over time."
- [data_facts | flag_column_holds] = {"FIN Flag Cnt": "RST", "SYN Flag Cnt": "PSH", "RST Flag Cnt": "ECE", "PSH Flag Cnt": "SYN", "ACK Flag Cnt": "ACK", "URG Flag Cnt": "FIN", "CWE Flag Count": "URG", "ECE Flag Cnt": "CWR"}
- [data_facts | flag_consistency.SYN_Flag_Cnt_equals_Fwd_PSH_Flags] = 1.0
- [data_facts | flag_consistency.CWE_Flag_Count_equals_Fwd_URG_Flags] = 1.0
- [data_facts | flag_consistency.RST_Flag_Cnt_equals_ECE_Flag_Cnt] = 1.0

**XN-FLAG-ORDER** (on: URG Flag Cnt)

> In this dataset the column named 'URG Flag Cnt' holds the FIN flag of the first packet: the extractor writes the eight flag counts in the iteration order of a Java HashMap, which does not follow the order of the column names.

- [cicflowmeter-src-2018 | BasicFlow.java, field declaration] "private HashMap<String, MutableInt> flagCounts;"
- [cicflowmeter-src-2018 | BasicFlow.java, initFlags()] "flagCounts.put("FIN", new MutableInt()); flagCounts.put("SYN", new MutableInt()); flagCounts.put("RST", new MutableInt()); flagCounts.put("PSH", new MutableInt()); flagCounts.put("ACK", new MutableInt()); flagCounts.put("URG", new MutableInt()); flagCounts.put("CWR", new MutableInt()); flagCounts.put("ECE", new MutableInt());"
- [cicflowmeter-src-2018 | BasicFlow.java, CSV row writer] "for(String key: flagCounts.keySet()){ dump.append(flagCounts.get(key).value).append(separator); //50,51,52,53,54,55,56,57"
- [java8-hashmap | class description] "This class makes no guarantees as to the order of the map; in particular, it does not guarantee that the order will remain constant over time."
- [data_facts | flag_column_holds] = {"FIN Flag Cnt": "RST", "SYN Flag Cnt": "PSH", "RST Flag Cnt": "ECE", "PSH Flag Cnt": "SYN", "ACK Flag Cnt": "ACK", "URG Flag Cnt": "FIN", "CWE Flag Count": "URG", "ECE Flag Cnt": "CWR"}
- [data_facts | flag_consistency.SYN_Flag_Cnt_equals_Fwd_PSH_Flags] = 1.0
- [data_facts | flag_consistency.CWE_Flag_Count_equals_Fwd_URG_Flags] = 1.0
- [data_facts | flag_consistency.RST_Flag_Cnt_equals_ECE_Flag_Cnt] = 1.0

**XN-FLAG-ORDER** (on: CWE Flag Count)

> In this dataset the column named 'CWE Flag Count' holds the URG flag of the first packet: the extractor writes the eight flag counts in the iteration order of a Java HashMap, which does not follow the order of the column names.

- [cicflowmeter-src-2018 | BasicFlow.java, field declaration] "private HashMap<String, MutableInt> flagCounts;"
- [cicflowmeter-src-2018 | BasicFlow.java, initFlags()] "flagCounts.put("FIN", new MutableInt()); flagCounts.put("SYN", new MutableInt()); flagCounts.put("RST", new MutableInt()); flagCounts.put("PSH", new MutableInt()); flagCounts.put("ACK", new MutableInt()); flagCounts.put("URG", new MutableInt()); flagCounts.put("CWR", new MutableInt()); flagCounts.put("ECE", new MutableInt());"
- [cicflowmeter-src-2018 | BasicFlow.java, CSV row writer] "for(String key: flagCounts.keySet()){ dump.append(flagCounts.get(key).value).append(separator); //50,51,52,53,54,55,56,57"
- [java8-hashmap | class description] "This class makes no guarantees as to the order of the map; in particular, it does not guarantee that the order will remain constant over time."
- [data_facts | flag_column_holds] = {"FIN Flag Cnt": "RST", "SYN Flag Cnt": "PSH", "RST Flag Cnt": "ECE", "PSH Flag Cnt": "SYN", "ACK Flag Cnt": "ACK", "URG Flag Cnt": "FIN", "CWE Flag Count": "URG", "ECE Flag Cnt": "CWR"}
- [data_facts | flag_consistency.SYN_Flag_Cnt_equals_Fwd_PSH_Flags] = 1.0
- [data_facts | flag_consistency.CWE_Flag_Count_equals_Fwd_URG_Flags] = 1.0
- [data_facts | flag_consistency.RST_Flag_Cnt_equals_ECE_Flag_Cnt] = 1.0

**XN-CWE-NAME** (on: CWE Flag Count)

> The name 'CWE Flag Count' is how the 2018 extractor labelled its CWR flag column; later versions call it 'CWR Flag Count'.

- [cicflowmeter-src-2018 | FlowFeature.java, feature //56] "cwe_cnt("CWE Flag Count","CWECT""
- [cicflowmeter-readme | FlowFeature.java (master), feature //56] "CWR_cnt("CWR Flag Count","CWRCT""

**DATA-IDENTICAL** (on: CWE Flag Count)

> In this dataset this column is identical to 'Fwd URG Flags' in every row.

- [data_facts | identical_with.CWE Flag Count] = ["Fwd URG Flags"]

**XN-FLAG-ORDER** (on: ECE Flag Cnt)

> In this dataset the column named 'ECE Flag Cnt' holds the CWR flag of the first packet: the extractor writes the eight flag counts in the iteration order of a Java HashMap, which does not follow the order of the column names.

- [cicflowmeter-src-2018 | BasicFlow.java, field declaration] "private HashMap<String, MutableInt> flagCounts;"
- [cicflowmeter-src-2018 | BasicFlow.java, initFlags()] "flagCounts.put("FIN", new MutableInt()); flagCounts.put("SYN", new MutableInt()); flagCounts.put("RST", new MutableInt()); flagCounts.put("PSH", new MutableInt()); flagCounts.put("ACK", new MutableInt()); flagCounts.put("URG", new MutableInt()); flagCounts.put("CWR", new MutableInt()); flagCounts.put("ECE", new MutableInt());"
- [cicflowmeter-src-2018 | BasicFlow.java, CSV row writer] "for(String key: flagCounts.keySet()){ dump.append(flagCounts.get(key).value).append(separator); //50,51,52,53,54,55,56,57"
- [java8-hashmap | class description] "This class makes no guarantees as to the order of the map; in particular, it does not guarantee that the order will remain constant over time."
- [data_facts | flag_column_holds] = {"FIN Flag Cnt": "RST", "SYN Flag Cnt": "PSH", "RST Flag Cnt": "ECE", "PSH Flag Cnt": "SYN", "ACK Flag Cnt": "ACK", "URG Flag Cnt": "FIN", "CWE Flag Count": "URG", "ECE Flag Cnt": "CWR"}
- [data_facts | flag_consistency.SYN_Flag_Cnt_equals_Fwd_PSH_Flags] = 1.0
- [data_facts | flag_consistency.CWE_Flag_Count_equals_Fwd_URG_Flags] = 1.0
- [data_facts | flag_consistency.RST_Flag_Cnt_equals_ECE_Flag_Cnt] = 1.0

**DATA-NEAR-IDENTICAL** (on: ECE Flag Cnt)

> In this dataset this column equals 'RST Flag Cnt' in at least 99.9% of rows.

- [data_facts | near_identical_with.ECE Flag Cnt] = ["RST Flag Cnt"]

**XN-BULK-ZERO** (on: Fwd Byts/b Avg, Fwd Pkts/b Avg, Fwd Blk Rate Avg, Bwd Byts/b Avg, Bwd Pkts/b Avg, Bwd Blk Rate Avg)

> Always 0 in this dataset: bulk statistics are updated only for the first packet of a flow (the update is disabled for later packets), so no bulk transfer is ever recorded.

- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [data_facts | constant_columns.Fwd Byts/b Avg] = 0.0
- [data_facts | constant_columns.Bwd Blk Rate Avg] = 0.0

**XN-SUBFLOW-ONE** (on: Subflow Fwd Pkts, Subflow Fwd Byts, Subflow Bwd Pkts, Subflow Bwd Byts)

> Subflow detection runs only for the first packet of a flow, so every flow counts as a single subflow and these columns repeat the flow totals.

- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, getSflow_fpackets()] "return this.forward.size()/sfCount;"
- [data_facts | identical_with.Subflow Fwd Pkts] = ["Tot Fwd Pkts"]
- [data_facts | near_identical_with.Subflow Bwd Byts] = ["TotLen Bwd Pkts"]

**DATA-IDENTICAL** (on: Subflow Fwd Pkts)

> In this dataset this column is identical to 'Tot Fwd Pkts' in every row.

- [data_facts | identical_with.Subflow Fwd Pkts] = ["Tot Fwd Pkts"]

**DATA-IDENTICAL** (on: Subflow Fwd Byts)

> In this dataset this column is identical to 'TotLen Fwd Pkts' in every row.

- [data_facts | identical_with.Subflow Fwd Byts] = ["TotLen Fwd Pkts"]

**DATA-IDENTICAL** (on: Subflow Bwd Pkts)

> In this dataset this column is identical to 'Tot Bwd Pkts' in every row.

- [data_facts | identical_with.Subflow Bwd Pkts] = ["Tot Bwd Pkts"]

**DATA-NEAR-IDENTICAL** (on: Subflow Bwd Byts)

> In this dataset this column equals 'TotLen Bwd Pkts' in at least 99.9% of rows.

- [data_facts | near_identical_with.Subflow Bwd Byts] = ["TotLen Bwd Pkts"]

**XN-INIT-WIN** (on: Init Fwd Win Byts, Init Bwd Win Byts)

> The value is the TCP window field read from a packet header: for the forward direction, the first packet of the flow; for the backward direction, the most recent backward packet (the value is overwritten on each one). -1 is the default kept when no TCP window was read, for example in a UDP flow or when there is no backward packet.

- [cicflowmeter-src-2018 | BasicPacketInfo.java, field declaration] "private int TCPWindow=-1;"
- [cicflowmeter-src-2018 | BasicFlow.java, field declarations] "private int Init_Win_bytes_forward=-1; private int Init_Win_bytes_backward=-1;"
- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setTCPWindow(tcp.window());"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), forward branch] "this.min_seg_size_forward = packet.getHeaderBytes(); Init_Win_bytes_forward = packet.getTCPWindow();"
- [cicflowmeter-src-2018 | BasicFlow.java, addPacket(), backward branch] "this.bwdPktStats.addValue((double)packet.getPayloadBytes()); Init_Win_bytes_backward = packet.getTCPWindow();"
- [data_facts | init_win_minus1.bwd_zero_bwd_pkts_or_udp_share] = 0.9984
- [data_facts | init_win_minus1.fwd_udp_share] = 0.9072

**XN-ACT-DATA-FIRST-PACKET** (on: Fwd Act Data Pkts)

> The first packet of the flow is never counted: the counter is incremented only for later forward packets that carry at least 1 byte of payload. A flow whose only forward payload is in its first packet therefore shows 0 here while its forward length is positive.

- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "if(packet.getPayloadBytes() >=1){ this.Act_data_pkt_forward++; }"
- [data_facts | fwd_payload_without_act_data.rows] = 87044
- [data_facts | fwd_payload_without_act_data.single_fwd_pkt_share] = 0.8746

**XN-SEG-SIZE-MIN-IS-HEADER** (on: Fwd Seg Size Min)

> Despite its name, this column is the smallest transport-header length in bytes among the forward packets of the flow, not a payload size.

- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "this.min_seg_size_forward = Math.min(packet.getHeaderBytes(),this.min_seg_size_forward);"
- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setHeaderBytes(tcp.getHeaderLength());"

## 5. H3 conditions: share of flows per label (train_full.csv)

| label | H3-NOPAYLOAD | H3-NOWIN | H3-TIMEOUT | H3-UDP |
|---|---|---|---|---|
| Benign | 22.9% | 37.1% | 0.6% | 20.3% |
| Bot | 1.2% | 1.1% | 0.0% | 0.0% |
| Brute_Force_Web | 30.6% | 51.1% | 0.0% | 21.2% |
| Brute_Force_XSS | 47.2% | 48.4% | 0.0% | 1.7% |
| DDOS_attack_HOIC | 17.8% | 17.8% | 0.0% | 0.0% |
| DDOS_attack_LOIC_UDP | 0.0% | 100.0% | 94.0% | 100.0% |
| DDoS_attacks_LOIC_HTTP | 50.3% | 50.4% | 0.1% | 0.1% |
| DoS_attacks_GoldenEye | 32.9% | 32.9% | 0.0% | 0.0% |
| DoS_attacks_Hulk | 82.7% | 82.7% | 0.0% | 0.0% |
| DoS_attacks_SlowHTTPTest | 100.0% | 0.0% | 0.0% | 0.0% |
| DoS_attacks_Slowloris | 23.1% | 23.1% | 0.0% | 0.0% |
| FTP_BruteForce | 100.0% | 0.0% | 0.0% | 0.0% |
| Infiltration | 33.3% | 45.2% | 0.7% | 29.8% |
| SQL_Injection | 40.3% | 26.9% | 0.0% | 0.0% |
| SSH_Bruteforce | 0.2% | 0.0% | 0.0% | 0.0% |

## 6. Findings from the 2018 extractor source code

Flag columns (each holds the flag of the FIRST packet only):

| CSV column | flag it actually holds |
|---|---|
| FIN Flag Cnt | RST |
| SYN Flag Cnt | PSH |
| RST Flag Cnt | ECE |
| PSH Flag Cnt | SYN |
| ACK Flag Cnt | ACK |
| URG Flag Cnt | FIN |
| CWE Flag Count | URG |
| ECE Flag Cnt | CWR |

Consistency in the data: {"SYN_Flag_Cnt_equals_Fwd_PSH_Flags": 1.0, "CWE_Flag_Count_equals_Fwd_URG_Flags": 1.0, "RST_Flag_Cnt_equals_ECE_Flag_Cnt": 1.0}

Constant columns: Bwd PSH Flags, Bwd URG Flags, Fwd Byts/b Avg, Fwd Pkts/b Avg, Fwd Blk Rate Avg, Bwd Byts/b Avg, Bwd Pkts/b Avg, Bwd Blk Rate Avg

Identical column pairs: [['Tot Fwd Pkts', 'Subflow Fwd Pkts'], ['Tot Bwd Pkts', 'Subflow Bwd Pkts'], ['TotLen Fwd Pkts', 'Subflow Fwd Byts'], ['Fwd PSH Flags', 'SYN Flag Cnt'], ['Fwd URG Flags', 'CWE Flag Count']]; near-identical: [['TotLen Bwd Pkts', 'Subflow Bwd Byts', 0.999993], ['RST Flag Cnt', 'ECE Flag Cnt', 0.999999]]

