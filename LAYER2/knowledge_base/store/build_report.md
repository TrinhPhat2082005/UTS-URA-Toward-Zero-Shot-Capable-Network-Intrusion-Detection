# Knowledge base build report

Built 2026-11-13 | passages.jsonl sha256 `7c5459ca3ce2849cfd4f1f4f1756ef5d85e01ef8a7140b4071992302ff2eb510` | dataset `train_full.csv (Protocol B training split)` sha256 `64f48652b0f6b1b7...`

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
| rfc7323 | standard | RFC 7323 (2014) | 2026-11-13 | rfc7323.txt `29e07732ea23` |
| rfc6335 | standard | RFC 6335 (2011) | 2026-11-13 | rfc6335.txt `a78d74017b69` |
| distrinet-2018 | audit | page retrieved on the date in sources.lock.json | 2026-11-13 | CSECICIDS2018.html `c9edddbcba99` |
| java8-hashmap | platform | Java SE 8 | 2026-11-13 | HashMap.html `8f718a56aa37` |
| java8-jls | platform | Java SE 8 | 2026-11-13 | jls-4.html `f99deeffe02e` |
| layer1-code | system | git: commit of each file recorded in sources.lock.json | 2026-11-13 | autoencoder.py `0321d6c1fd5a`; train_protocol_b.py `a873c3c549c2` |

## 2. Passages per part

| part | passages | verbatim | rendered | authored | sent to VLM |
|---|---|---|---|---|---|
| A' label card | 14 | 0 | 0 | 14 | yes |
| B' mechanism (ATT&CK + CAPEC) | 14 | 14 | 0 | 0 | yes |
| H1 feature definitions | 81 | 1 | 54 | 26 | yes |
| H2 protocol / port / TCP flag | 21 | 9 | 12 | 0 | yes |
| H3 conditional extractor artefacts | 6 | 0 | 0 | 6 | yes |
| G' mitigations (code-filled, not sent to the VLM) | 19 | 0 | 19 | 0 | no |

## 3. Label mapping (the only author decisions)

| label | technique | previous profile | changed | CAPEC (route) | CAPEC entries CAPEC links to the technique |
|---|---|---|---|---|---|
| SSH_Bruteforce | T1110.001 Password Guessing | T1110.001 | no | CAPEC-16 (child_of_linked) | CAPEC-49 |
| FTP_BruteForce | T1110.001 Password Guessing | T1110.001 | no | CAPEC-16 (child_of_linked) | CAPEC-49 |
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

- **SSH_Bruteforce** — The scenario guesses account passwords from a dictionary; Password Guessing is the ATT&CK sub-technique for guessing passwords from a list. Same as the previous mapping. CAPEC: The dataset page says the passwords come from a dictionary; CAPEC-16 Dictionary-based Password Attack describes that case and is a child of CAPEC-49, the entry CAPEC links to T1110.001. CHANGED from CAPEC-49, whose description (trying every possible value) does not match a dictionary attack.
- **FTP_BruteForce** — Same scenario as SSH_Bruteforce on the FTP service; the label keeps the technique named by the scenario, while the label card states that the data show a failed attempt. Same as the previous mapping. CAPEC: Same choice as SSH_Bruteforce: the scenario uses a dictionary. CHANGED from CAPEC-49.
- **DoS_attacks_GoldenEye** — Denial of service against a web service (UNB: DoS attack on Apache; every flow of the label goes to port 80). Same as the previous mapping. CAPEC: Of the CAPEC entries linked to T1499.002 (469, 482, 488, 489, 528), HTTP Flood matches a DoS on a web server reached on port 80; the dataset page does not describe a slow-request attack for this tool.
- **DoS_attacks_Hulk** — Denial of service against a web service (UNB: DoS attack on Apache; every flow of the label goes to port 80). Same as the previous mapping. CAPEC: Same choice as DoS_attacks_GoldenEye.
- **DoS_attacks_SlowHTTPTest** — Denial of service against a web service per the dataset page; the label keeps the technique named by the scenario, while the label card states that the data show a failed attempt on port 21. Same as the previous mapping. CAPEC: Of the CAPEC entries linked to T1499.002, HTTP DoS describes holding HTTP sessions open, which is what a slow-HTTP test tool does; the dataset page gives only the tool name.
- **DoS_attacks_Slowloris** — Exhausts the connection capacity of a web service. Same as the previous mapping. CAPEC: HTTP DoS (keeping HTTP sessions alive) is the CAPEC entry linked to T1499.002 that matches the scenario's description of Slowloris.
- **DDoS_attacks_LOIC_HTTP** — HTTP requests flooding a web service. Same as the previous mapping. CAPEC: The label names HTTP requests; HTTP Flood is the CAPEC entry linked to T1499.002 for floods of HTTP requests.
- **DDOS_attack_LOIC_UDP** — Every flow of this label is UDP, i.e. a flood of network traffic rather than requests to a service's application logic. Direct Network Flood names UDP floods. CHANGED from the previous mapping T1499.002 (Service Exhaustion Flood), which describes attacks on the services themselves. CAPEC: Of the CAPEC entries linked to T1498.001 (125, 482, 528, 666), Flooding is the general one; 482 is TCP-specific, 528 is XML-specific and 666 is Bluetooth-specific.
- **DDOS_attack_HOIC** — The tool attacks URLs, i.e. a web service. Same as the previous mapping. CAPEC: Attacking URLs is an HTTP-level flood; HTTP Flood is the CAPEC entry linked to T1499.002 for it.
- **Brute_Force_Web** — Brute force against a web login guesses passwords. Same as the previous mapping. CAPEC: CAPEC-49 is the entry CAPEC links to T1110.001. CAPEC-16 (dictionary) is not chosen here because the dataset page does not describe the password list used against the web login; the dictionary sentence belongs to the FTP and SSH scenario.
- **Brute_Force_XSS** — The captured flows are the attacker's automated requests to the vulnerable web application, i.e. an attempt to exploit a public-facing application. Drive-by Compromise (T1189) describes a user's browser visiting a compromised site, which is not the traffic captured in this scenario. CHANGED from the previous mapping T1189. SQL injection and XSS share T1190 and are told apart by their CAPEC entries. CAPEC: CAPEC v3.9 links no attack pattern to T1190; CAPEC-63 Cross-Site Scripting is chosen because it describes the attack named by the label.
- **SQL_Injection** — SQL injection exploits a weakness of a public-facing web application. Same as the previous mapping. CAPEC: CAPEC v3.9 links no attack pattern to T1190; CAPEC-66 SQL Injection is chosen because it describes the attack named by the label.
- **Infiltration** — Of the scenario's stages, the internal scan is the one described on the dataset page as network activity against many hosts and ports (no single destination port carries half of the label's flows); the delivery of the malicious file and the exploitation run on the victim host. CHANGED from the previous mapping T1204.002 + T1046: the user-execution stage is not kept because the report must describe network flows. CAPEC: Port Scanning is the CAPEC entry linked to T1046.
- **Bot** — The network-visible part of the scenario is the controller's commands and the infected hosts' replies; nearly all of the label's flows use TCP port 8080, registered by IANA as HTTP Alternate. CHANGED from the previous mapping T1583.005 (Botnet) + T1113 (Screen Capture): the first describes the attacker acquiring infrastructure and the second an action on the host, neither of which is observable in a flow. CAPEC: CAPEC v3.9 links no attack pattern to T1071.001 and has no entry for botnet command and control; no CAPEC passage.

## 4. Every author-written sentence next to its evidence

Code checks that each quote exists verbatim; a reviewer checks that the sentence says no more than its evidence.

### A-SSH_Bruteforce

> Reference label SSH_Bruteforce (CSE-CIC-IDS2018): Dictionary password guessing against the SSH service of the victim server.

- [unb-ids2018 | 1. Introduction, Brute force attacks] "by running a dictionary brute force attack against the main server."
- [claim | ssh_port22_ge999] At least 99.9% of SSH_Bruteforce flows go to port 22. — holds on every source: yes (5 sources)
- [decision T1110.001] The scenario guesses account passwords from a dictionary; Password Guessing is the ATT&CK sub-technique for guessing passwords from a list. Same as the previous mapping.

### A-FTP_BruteForce

> Reference label FTP_BruteForce (CSE-CIC-IDS2018): The dataset page describes a dictionary password attack against the FTP service of the victim server. In the published CSV files every flow with this label goes to TCP port 21 and carries no payload in either direction. An independent analysis of the original capture reports that the target answered the connection attempts only with reset packets, so port 21 was closed and no credentials could be tried.

- [unb-ids2018 | 2.2.1 Brute-force attack] "For a list of passwords, we use a large dictionary that contains 90 million words."
- [claim | twin_tcp21_nopayload__FTP_BruteForce] Every FTP_BruteForce flow goes to TCP port 21 and carries no payload. — holds on every source: yes (5 sources)
- [distrinet-2018 | 1. FTP Patator, 14-02-2018] "In response to the SYN packets being sent out, only [RST, ACK] packets are sent in response, suggesting that port 21 is closed, and so the attacker has no opportunity to actually "brute force" with credentials."
- [decision T1110.001] Same scenario as SSH_Bruteforce on the FTP service; the label keeps the technique named by the scenario, while the label card states that the data show a failed attempt. Same as the previous mapping.

### A-DoS_attacks_GoldenEye

> Reference label DoS_attacks_GoldenEye (CSE-CIC-IDS2018): Denial-of-service attack on the victim's Apache web server.

- [unb-ids2018 | Table 1] "DoS attack Hulk, GoldenEye, Slowloris, Slowhttptest One day Kali linux Ubuntu 16.4 (Apache)"
- [claim | port80_all__DoS_attacks_GoldenEye] Every DoS_attacks_GoldenEye flow goes to port 80. — holds on every source: yes (5 sources)
- [decision T1499.002] Denial of service against a web service (UNB: DoS attack on Apache; every flow of the label goes to port 80). Same as the previous mapping.

### A-DoS_attacks_Hulk

> Reference label DoS_attacks_Hulk (CSE-CIC-IDS2018): Denial-of-service attack on the victim's Apache web server.

- [unb-ids2018 | Table 1] "DoS attack Hulk, GoldenEye, Slowloris, Slowhttptest One day Kali linux Ubuntu 16.4 (Apache)"
- [claim | port80_all__DoS_attacks_Hulk] Every DoS_attacks_Hulk flow goes to port 80. — holds on every source: yes (5 sources)
- [decision T1499.002] Denial of service against a web service (UNB: DoS attack on Apache; every flow of the label goes to port 80). Same as the previous mapping.

### A-DoS_attacks_SlowHTTPTest

> Reference label DoS_attacks_SlowHTTPTest (CSE-CIC-IDS2018): The dataset page lists Slowhttptest among the denial-of-service tools run against an Apache web server. In the published CSV files every flow with this label goes to TCP port 21 (FTP) and carries no payload in either direction. An independent analysis of the original capture found no successful Slowhttptest attack and reports that the tool appears to have been launched against port 21 by mistake.

- [unb-ids2018 | Table 1] "DoS attack Hulk, GoldenEye, Slowloris, Slowhttptest One day Kali linux Ubuntu 16.4 (Apache)"
- [claim | twin_tcp21_nopayload__DoS_attacks_SlowHTTPTest] Every DoS_attacks_SlowHTTPTest flow goes to TCP port 21 and carries no payload. — holds on every source: yes (5 sources)
- [distrinet-2018 | 6. DoS Slowhttptest, 16-02-2018] "No succesful DoS Slowhttptest attack was found in this dataset. Instead, it looks like the tool was mistakenly launched against port 21."
- [decision T1499.002] Denial of service against a web service per the dataset page; the label keeps the technique named by the scenario, while the label card states that the data show a failed attempt on port 21. Same as the previous mapping.

### A-DoS_attacks_Slowloris

> Reference label DoS_attacks_Slowloris (CSE-CIC-IDS2018): Slow HTTP denial of service: connections to the victim web server are held open by sending valid but incomplete HTTP requests at regular intervals, until the server has no free connections.

- [unb-ids2018 | 1. Introduction, HTTP denial of service] "The tool holds the connection open by sending valid, incomplete HTTP requests to the server at regular intervals to keep the sockets from closing. Since any Web server has a finite ability to serve connections, it will only be a matter of time before all sockets are used up and no other connection can be made."
- [claim | port80_all__DoS_attacks_Slowloris] Every DoS_attacks_Slowloris flow goes to port 80. — holds on every source: yes (5 sources)
- [decision T1499.002] Exhausts the connection capacity of a web service. Same as the previous mapping.

### A-DDoS_attacks_LOIC_HTTP

> Reference label DDoS_attacks_LOIC_HTTP (CSE-CIC-IDS2018): Distributed denial of service with HTTP requests against the victim web server.

- [unb-ids2018 | Table 1] "Low Orbit Ion Canon (LOIC) for UDP, TCP, or HTTP requests"
- [claim | port80_all__DDoS_attacks_LOIC_HTTP] Every DDoS_attacks_LOIC_HTTP flow goes to port 80. — holds on every source: yes (5 sources)
- [decision T1499.002] HTTP requests flooding a web service. Same as the previous mapping.

### A-DDOS_attack_LOIC_UDP

> Reference label DDOS_attack_LOIC_UDP (CSE-CIC-IDS2018): Distributed denial of service with UDP traffic sent to the victim server.

- [unb-ids2018 | Table 1] "Low Orbit Ion Canon (LOIC) for UDP, TCP, or HTTP requests"
- [claim | loic_udp_udp_port80] Every DDOS_attack_LOIC_UDP flow is UDP to port 80. — holds on every source: yes (5 sources)
- [decision T1498.001] Every flow of this label is UDP, i.e. a flood of network traffic rather than requests to a service's application logic. Direct Network Flood names UDP floods. CHANGED from the previous mapping T1499.002 (Service Exhaustion Flood), which describes attacks on the services themselves.

### A-DDOS_attack_HOIC

> Reference label DDOS_attack_HOIC (CSE-CIC-IDS2018): Distributed denial of service with HTTP requests against the victim web server, sent from several attacking computers.

- [unb-ids2018 | 2.2.5 Distributed Denial-of-Service] "designed to attack as many as 256 URLs at the same time."
- [unb-ids2018 | 2.2.5 Distributed Denial-of-Service] "In this scenario, we use free HOIC tool to conduct DDoS attack by using 4 different computers."
- [claim | port80_all__DDOS_attack_HOIC] Every DDOS_attack_HOIC flow goes to port 80. — holds on every source: yes (5 sources)
- [decision T1499.002] The tool attacks URLs, i.e. a web service. Same as the previous mapping.

### A-Brute_Force_Web

> Reference label Brute_Force_Web (CSE-CIC-IDS2018): The dataset page describes automated attacks, scripted with the Selenium framework, on the brute-force section of a deliberately vulnerable web application (DVWA) on the victim web server; an independent analysis of the capture describes repeated password login attempts on a web page. Not every flow carrying this label is HTTP: the label also contains UDP flows and flows to destination ports other than 80, although port 80 is its most common destination port.

- [unb-ids2018 | 2.2.6 Web Attacks] "DVWA is a PHP/MySQL web application that is vulnerable."
- [unb-ids2018 | 2.2.6 Web Attacks] "To automate the attacks in XSS and Brute-force section we developed an automation code with Selenium framework."
- [distrinet-2018 | 10. Web Attack - Brute Force, 22-02-2018] "This attack looks mostly like a web-page based attack with repeated password login attempts."
- [claim | bfw_has_udp_and_other_ports] The Brute_Force_Web label contains UDP flows and flows to ports other than 80 (label level: training split, validation split, and the published CSV files taken together). — holds on every source: yes (3 sources)
- [claim | bfw_port80_top] Port 80 is the most common destination port of Brute_Force_Web flows. — holds on every source: yes (8 sources)
- [decision T1110.001] Brute force against a web login guesses passwords. Same as the previous mapping.

### A-Brute_Force_XSS

> Reference label Brute_Force_XSS (CSE-CIC-IDS2018): Automated cross-site scripting (XSS) attempts against a deliberately vulnerable web application (DVWA) on the victim web server.

- [unb-ids2018 | 2.2.6 Web Attacks] "DVWA is a PHP/MySQL web application that is vulnerable."
- [unb-ids2018 | 2.2.6 Web Attacks] "To automate the attacks in XSS and Brute-force section we developed an automation code with Selenium framework."
- [claim | xss_port80_ge90] At least 90% of Brute_Force_XSS flows go to port 80. — holds on every source: yes (8 sources)
- [decision T1190] The captured flows are the attacker's automated requests to the vulnerable web application, i.e. an attempt to exploit a public-facing application. Drive-by Compromise (T1189) describes a user's browser visiting a compromised site, which is not the traffic captured in this scenario. CHANGED from the previous mapping T1189. SQL injection and XSS share T1190 and are told apart by their CAPEC entries.

### A-SQL_Injection

> Reference label SQL_Injection (CSE-CIC-IDS2018): SQL injection attempts against a deliberately vulnerable web application (DVWA) on the victim web server.

- [unb-ids2018 | 1. Introduction, Collection of web application attacks] "we conduct different types of web attacks on the vulnerable website, including SQL injection, command injection, and unrestricted file upload."
- [unb-ids2018 | 2.2.6 Web Attacks] "DVWA is a PHP/MySQL web application that is vulnerable."
- [claim | port80_all__SQL_Injection] Every SQL_Injection flow goes to port 80. — holds on every source: yes (6 sources)
- [decision T1190] SQL injection exploits a weakness of a public-facing web application. Same as the previous mapping.

### A-Infiltration

> Reference label Infiltration (CSE-CIC-IDS2018): The dataset page describes this label as the internal stage of an infiltration scenario: a compromised victim host was used for an IP sweep, a full port scan and service enumeration of the internal network with Nmap. The label is not limited to scan traffic: at least a tenth of its flows go to DNS (UDP port 53), at least a tenth go to HTTPS (TCP port 443), and no single destination port carries half of its flows.

- [unb-ids2018 | 2.2.7 Infiltration of the network from inside] "IP sweep, full port scan and service enumerations using Nmap."
- [unb-ids2018 | 1. Introduction, Infiltration of the network from inside] "we use his computer to scan the internal network for other vulnerable boxes and exploit them if possible."
- [claim | infil_udp53_ge10] At least 10% of Infiltration flows are UDP to port 53 (DNS). — holds on every source: yes (8 sources)
- [claim | infil_tcp443_ge10] At least 10% of Infiltration flows are TCP to port 443 (HTTPS). — holds on every source: yes (8 sources)
- [claim | infil_max_port_lt50] No single destination port carries half of the Infiltration flows. — holds on every source: yes (8 sources)
- [decision T1046] Of the scenario's stages, the internal scan is the one described on the dataset page as network activity against many hosts and ports (no single destination port carries half of the label's flows); the delivery of the malicious file and the exploitation run on the victim host. CHANGED from the previous mapping T1204.002 + T1046: the user-execution stage is not kept because the report must describe network flows.

### A-Bot

> Reference label Bot (CSE-CIC-IDS2018): The dataset page describes victim machines infected with botnet malware (it names Zeus and Ares), with the controller requesting screenshots from the infected hosts every 400 seconds. It lists remote shell, file upload/download, screenshot capture and key logging as capabilities of the Ares tool, not as actions recorded in the capture. An independent analysis of the capture found no evidence of the Zeus botnet. Nearly all flows with this label go to TCP port 8080.

- [unb-ids2018 | 2.2.3 Botnet] "In this scenario, we infect machines with two different botnets (Zeus and Ares), also every 400 seconds we request screenshots from the zombies."
- [unb-ids2018 | Table 1] "Ares (developed by Python): remote shell, file upload/download, capturing screenshots and key logging"
- [distrinet-2018 | 12. Botnet Ares] "All botnet traffic occurs on port 8080, and we found no evidence of the Zeus botnet in this trace."
- [claim | bot_tcp8080_gt95] More than 95% of Bot flows go to TCP port 8080. — holds on every source: yes (5 sources)
- [decision T1071.001] The network-visible part of the scenario is the controller's commands and the infected hosts' replies; nearly all of the label's flows use TCP port 8080, registered by IANA as HTTP Alternate. CHANGED from the previous mapping T1583.005 (Botnet) + T1113 (Screen Capture): the first describes the attacker acquiring infrastructure and the second an action on the host, neither of which is observable in a flow.

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

### H1-Fwd_PSH_Flags

> Fwd PSH Flags: In CSE-CIC-IDS2018 this column is 1 if the first packet of the flow has the PSH flag set and 0 otherwise. It describes that packet only and is not a count of packets: the extractor reads flags only from the first packet of a flow.

- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "public void firstPacket(BasicPacketInfo packet){ updateFlowBulk(packet); detectUpdateSubflows(packet); checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), forward branch] "if(packet.hasFlagPSH()){ this.fPSH_cnt++; } if(packet.hasFlagURG()){ this.fURG_cnt++; }"
- [claim | flags_binary] The eight flag-count columns and Fwd PSH/URG Flags take only the values 0 and 1. — holds on every source: yes (12 sources)
- [cicflowmeter-readme | ReadMe.txt, list of extracted features] "Fwd PSH flags Number of times the PSH flag was set in packets travelling in the forward direction (0 for UDP)"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //37] "fw_psh_flag("Fwd PSH Flags","FwPSH""

### H1-Bwd_PSH_Flags

> Bwd PSH Flags: In the published CSE-CIC-IDS2018 CSV files this column is always 0: the backward PSH and URG counters are incremented only in the first-packet routine, and the first packet always defines the forward direction.

- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "if(this.src==null){ this.src = packet.getSrc();"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), backward branch] "if(packet.hasFlagPSH()){ this.bPSH_cnt++; } if(packet.hasFlagURG()){ this.bURG_cnt++; }"
- [claim | bwd_dir_flags_zero] Bwd PSH Flags and Bwd URG Flags are 0 in every row. — holds on every source: yes (12 sources)
- [cicflowmeter-readme | ReadMe.txt, list of extracted features] "Bwd PSH Flags Number of times the PSH flag was set in packets travelling in the backward direction (0 for UDP)"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //38] "bw_psh_flag("Bwd PSH Flags","BwPSH""

### H1-Fwd_URG_Flags

> Fwd URG Flags: In CSE-CIC-IDS2018 this column is 1 if the first packet of the flow has the URG flag set and 0 otherwise. It describes that packet only and is not a count of packets: the extractor reads flags only from the first packet of a flow.

- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "public void firstPacket(BasicPacketInfo packet){ updateFlowBulk(packet); detectUpdateSubflows(packet); checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), forward branch] "if(packet.hasFlagPSH()){ this.fPSH_cnt++; } if(packet.hasFlagURG()){ this.fURG_cnt++; }"
- [claim | flags_binary] The eight flag-count columns and Fwd PSH/URG Flags take only the values 0 and 1. — holds on every source: yes (12 sources)
- [cicflowmeter-readme | ReadMe.txt, list of extracted features] "Fwd URG Flags Number of times the URG flag was set in packets travelling in the forward direction (0 for UDP)"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //39] "fw_urg_flag("Fwd URG Flags","FwURG""

### H1-Bwd_URG_Flags

> Bwd URG Flags: In the published CSE-CIC-IDS2018 CSV files this column is always 0: the backward PSH and URG counters are incremented only in the first-packet routine, and the first packet always defines the forward direction.

- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "if(this.src==null){ this.src = packet.getSrc();"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), backward branch] "if(packet.hasFlagPSH()){ this.bPSH_cnt++; } if(packet.hasFlagURG()){ this.bURG_cnt++; }"
- [claim | bwd_dir_flags_zero] Bwd PSH Flags and Bwd URG Flags are 0 in every row. — holds on every source: yes (12 sources)
- [cicflowmeter-readme | ReadMe.txt, list of extracted features] "Bwd URG Flags Number of times the URG flag was set in packets travelling in the backward direction (0 for UDP)"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //40] "bw_urg_flag("Bwd URG Flags","BwURG""

### H1-FIN_Flag_Cnt

> FIN Flag Cnt: In CSE-CIC-IDS2018 this column is 1 if the first packet of the flow has the RST flag set and 0 otherwise. It describes that packet only and is not a count of packets: the extractor reads flags only from the first packet of a flow. The column name does not match its content: the extractor writes the eight flag values in the iteration order of a Java HashMap, which does not follow the order of the column names, so this column holds the RST flag and not the FIN flag.

- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "public void firstPacket(BasicPacketInfo packet){ updateFlowBulk(packet); detectUpdateSubflows(packet); checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), forward branch] "if(packet.hasFlagPSH()){ this.fPSH_cnt++; } if(packet.hasFlagURG()){ this.fURG_cnt++; }"
- [claim | flags_binary] The eight flag-count columns and Fwd PSH/URG Flags take only the values 0 and 1. — holds on every source: yes (12 sources)
- [cicflowmeter-src-2018 | BasicFlow.java, field declaration] "private HashMap<String, MutableInt> flagCounts;"
- [cicflowmeter-src-2018 | BasicFlow.java, initFlags()] "flagCounts.put("FIN", new MutableInt()); flagCounts.put("SYN", new MutableInt()); flagCounts.put("RST", new MutableInt()); flagCounts.put("PSH", new MutableInt()); flagCounts.put("ACK", new MutableInt()); flagCounts.put("URG", new MutableInt()); flagCounts.put("CWR", new MutableInt()); flagCounts.put("ECE", new MutableInt());"
- [cicflowmeter-src-2018 | BasicFlow.java, CSV row writer] "for(String key: flagCounts.keySet()){ dump.append(flagCounts.get(key).value).append(separator); //50,51,52,53,54,55,56,57"
- [java8-hashmap | class description] "This class makes no guarantees as to the order of the map; in particular, it does not guarantee that the order will remain constant over time."
- [data_facts | flag_column_holds] = {"FIN Flag Cnt": "RST", "SYN Flag Cnt": "PSH", "RST Flag Cnt": "ECE", "PSH Flag Cnt": "SYN", "ACK Flag Cnt": "ACK", "URG Flag Cnt": "FIN", "CWE Flag Count": "URG", "ECE Flag Cnt": "CWR"}
- [claim | identical__Fwd PSH Flags__SYN Flag Cnt] 'Fwd PSH Flags' equals 'SYN Flag Cnt' in every row. — holds on every source: yes (12 sources)
- [claim | identical__Fwd URG Flags__CWE Flag Count] 'Fwd URG Flags' equals 'CWE Flag Count' in every row. — holds on every source: yes (12 sources)
- [claim | near_identical__RST Flag Cnt__ECE Flag Cnt] 'RST Flag Cnt' equals 'ECE Flag Cnt' in at least 99.9% of rows. — holds on every source: yes (12 sources)
- [cicflowmeter-readme | ReadMe.txt, list of extracted features] "FIN Flag Count Number of packets with FIN"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //50] "fin_cnt("FIN Flag Cnt","FINCT""

### H1-SYN_Flag_Cnt

> SYN Flag Cnt: In CSE-CIC-IDS2018 this column is 1 if the first packet of the flow has the PSH flag set and 0 otherwise. It describes that packet only and is not a count of packets: the extractor reads flags only from the first packet of a flow. The column name does not match its content: the extractor writes the eight flag values in the iteration order of a Java HashMap, which does not follow the order of the column names, so this column holds the PSH flag and not the SYN flag.

- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "public void firstPacket(BasicPacketInfo packet){ updateFlowBulk(packet); detectUpdateSubflows(packet); checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), forward branch] "if(packet.hasFlagPSH()){ this.fPSH_cnt++; } if(packet.hasFlagURG()){ this.fURG_cnt++; }"
- [claim | flags_binary] The eight flag-count columns and Fwd PSH/URG Flags take only the values 0 and 1. — holds on every source: yes (12 sources)
- [cicflowmeter-src-2018 | BasicFlow.java, field declaration] "private HashMap<String, MutableInt> flagCounts;"
- [cicflowmeter-src-2018 | BasicFlow.java, initFlags()] "flagCounts.put("FIN", new MutableInt()); flagCounts.put("SYN", new MutableInt()); flagCounts.put("RST", new MutableInt()); flagCounts.put("PSH", new MutableInt()); flagCounts.put("ACK", new MutableInt()); flagCounts.put("URG", new MutableInt()); flagCounts.put("CWR", new MutableInt()); flagCounts.put("ECE", new MutableInt());"
- [cicflowmeter-src-2018 | BasicFlow.java, CSV row writer] "for(String key: flagCounts.keySet()){ dump.append(flagCounts.get(key).value).append(separator); //50,51,52,53,54,55,56,57"
- [java8-hashmap | class description] "This class makes no guarantees as to the order of the map; in particular, it does not guarantee that the order will remain constant over time."
- [data_facts | flag_column_holds] = {"FIN Flag Cnt": "RST", "SYN Flag Cnt": "PSH", "RST Flag Cnt": "ECE", "PSH Flag Cnt": "SYN", "ACK Flag Cnt": "ACK", "URG Flag Cnt": "FIN", "CWE Flag Count": "URG", "ECE Flag Cnt": "CWR"}
- [claim | identical__Fwd PSH Flags__SYN Flag Cnt] 'Fwd PSH Flags' equals 'SYN Flag Cnt' in every row. — holds on every source: yes (12 sources)
- [claim | identical__Fwd URG Flags__CWE Flag Count] 'Fwd URG Flags' equals 'CWE Flag Count' in every row. — holds on every source: yes (12 sources)
- [claim | near_identical__RST Flag Cnt__ECE Flag Cnt] 'RST Flag Cnt' equals 'ECE Flag Cnt' in at least 99.9% of rows. — holds on every source: yes (12 sources)
- [cicflowmeter-readme | ReadMe.txt, list of extracted features] "SYN Flag Count Number of packets with SYN"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //51] "syn_cnt("SYN Flag Cnt","SYNCT""

### H1-RST_Flag_Cnt

> RST Flag Cnt: In CSE-CIC-IDS2018 this column is 1 if the first packet of the flow has the ECE flag set and 0 otherwise. It describes that packet only and is not a count of packets: the extractor reads flags only from the first packet of a flow. The column name does not match its content: the extractor writes the eight flag values in the iteration order of a Java HashMap, which does not follow the order of the column names, so this column holds the ECE flag and not the RST flag.

- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "public void firstPacket(BasicPacketInfo packet){ updateFlowBulk(packet); detectUpdateSubflows(packet); checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), forward branch] "if(packet.hasFlagPSH()){ this.fPSH_cnt++; } if(packet.hasFlagURG()){ this.fURG_cnt++; }"
- [claim | flags_binary] The eight flag-count columns and Fwd PSH/URG Flags take only the values 0 and 1. — holds on every source: yes (12 sources)
- [cicflowmeter-src-2018 | BasicFlow.java, field declaration] "private HashMap<String, MutableInt> flagCounts;"
- [cicflowmeter-src-2018 | BasicFlow.java, initFlags()] "flagCounts.put("FIN", new MutableInt()); flagCounts.put("SYN", new MutableInt()); flagCounts.put("RST", new MutableInt()); flagCounts.put("PSH", new MutableInt()); flagCounts.put("ACK", new MutableInt()); flagCounts.put("URG", new MutableInt()); flagCounts.put("CWR", new MutableInt()); flagCounts.put("ECE", new MutableInt());"
- [cicflowmeter-src-2018 | BasicFlow.java, CSV row writer] "for(String key: flagCounts.keySet()){ dump.append(flagCounts.get(key).value).append(separator); //50,51,52,53,54,55,56,57"
- [java8-hashmap | class description] "This class makes no guarantees as to the order of the map; in particular, it does not guarantee that the order will remain constant over time."
- [data_facts | flag_column_holds] = {"FIN Flag Cnt": "RST", "SYN Flag Cnt": "PSH", "RST Flag Cnt": "ECE", "PSH Flag Cnt": "SYN", "ACK Flag Cnt": "ACK", "URG Flag Cnt": "FIN", "CWE Flag Count": "URG", "ECE Flag Cnt": "CWR"}
- [claim | identical__Fwd PSH Flags__SYN Flag Cnt] 'Fwd PSH Flags' equals 'SYN Flag Cnt' in every row. — holds on every source: yes (12 sources)
- [claim | identical__Fwd URG Flags__CWE Flag Count] 'Fwd URG Flags' equals 'CWE Flag Count' in every row. — holds on every source: yes (12 sources)
- [claim | near_identical__RST Flag Cnt__ECE Flag Cnt] 'RST Flag Cnt' equals 'ECE Flag Cnt' in at least 99.9% of rows. — holds on every source: yes (12 sources)
- [cicflowmeter-readme | ReadMe.txt, list of extracted features] "RST Flag Count Number of packets with RST"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //52] "rst_cnt("RST Flag Cnt","RSTCT""

### H1-PSH_Flag_Cnt

> PSH Flag Cnt: In CSE-CIC-IDS2018 this column is 1 if the first packet of the flow has the SYN flag set and 0 otherwise. It describes that packet only and is not a count of packets: the extractor reads flags only from the first packet of a flow. The column name does not match its content: the extractor writes the eight flag values in the iteration order of a Java HashMap, which does not follow the order of the column names, so this column holds the SYN flag and not the PSH flag.

- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "public void firstPacket(BasicPacketInfo packet){ updateFlowBulk(packet); detectUpdateSubflows(packet); checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), forward branch] "if(packet.hasFlagPSH()){ this.fPSH_cnt++; } if(packet.hasFlagURG()){ this.fURG_cnt++; }"
- [claim | flags_binary] The eight flag-count columns and Fwd PSH/URG Flags take only the values 0 and 1. — holds on every source: yes (12 sources)
- [cicflowmeter-src-2018 | BasicFlow.java, field declaration] "private HashMap<String, MutableInt> flagCounts;"
- [cicflowmeter-src-2018 | BasicFlow.java, initFlags()] "flagCounts.put("FIN", new MutableInt()); flagCounts.put("SYN", new MutableInt()); flagCounts.put("RST", new MutableInt()); flagCounts.put("PSH", new MutableInt()); flagCounts.put("ACK", new MutableInt()); flagCounts.put("URG", new MutableInt()); flagCounts.put("CWR", new MutableInt()); flagCounts.put("ECE", new MutableInt());"
- [cicflowmeter-src-2018 | BasicFlow.java, CSV row writer] "for(String key: flagCounts.keySet()){ dump.append(flagCounts.get(key).value).append(separator); //50,51,52,53,54,55,56,57"
- [java8-hashmap | class description] "This class makes no guarantees as to the order of the map; in particular, it does not guarantee that the order will remain constant over time."
- [data_facts | flag_column_holds] = {"FIN Flag Cnt": "RST", "SYN Flag Cnt": "PSH", "RST Flag Cnt": "ECE", "PSH Flag Cnt": "SYN", "ACK Flag Cnt": "ACK", "URG Flag Cnt": "FIN", "CWE Flag Count": "URG", "ECE Flag Cnt": "CWR"}
- [claim | identical__Fwd PSH Flags__SYN Flag Cnt] 'Fwd PSH Flags' equals 'SYN Flag Cnt' in every row. — holds on every source: yes (12 sources)
- [claim | identical__Fwd URG Flags__CWE Flag Count] 'Fwd URG Flags' equals 'CWE Flag Count' in every row. — holds on every source: yes (12 sources)
- [claim | near_identical__RST Flag Cnt__ECE Flag Cnt] 'RST Flag Cnt' equals 'ECE Flag Cnt' in at least 99.9% of rows. — holds on every source: yes (12 sources)
- [cicflowmeter-readme | ReadMe.txt, list of extracted features] "PSH Flag Count Number of packets with PUSH"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //53] "pst_cnt("PSH Flag Cnt","PSHCT""

### H1-ACK_Flag_Cnt

> ACK Flag Cnt: In CSE-CIC-IDS2018 this column is 1 if the first packet of the flow has the ACK flag set and 0 otherwise. It describes that packet only and is not a count of packets: the extractor reads flags only from the first packet of a flow.

- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "public void firstPacket(BasicPacketInfo packet){ updateFlowBulk(packet); detectUpdateSubflows(packet); checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), forward branch] "if(packet.hasFlagPSH()){ this.fPSH_cnt++; } if(packet.hasFlagURG()){ this.fURG_cnt++; }"
- [claim | flags_binary] The eight flag-count columns and Fwd PSH/URG Flags take only the values 0 and 1. — holds on every source: yes (12 sources)
- [cicflowmeter-readme | ReadMe.txt, list of extracted features] "ACK Flag Count Number of packets with ACK"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //54] "ack_cnt("ACK Flag Cnt","ACKCT""

### H1-URG_Flag_Cnt

> URG Flag Cnt: In CSE-CIC-IDS2018 this column is 1 if the first packet of the flow has the FIN flag set and 0 otherwise. It describes that packet only and is not a count of packets: the extractor reads flags only from the first packet of a flow. The column name does not match its content: the extractor writes the eight flag values in the iteration order of a Java HashMap, which does not follow the order of the column names, so this column holds the FIN flag and not the URG flag.

- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "public void firstPacket(BasicPacketInfo packet){ updateFlowBulk(packet); detectUpdateSubflows(packet); checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), forward branch] "if(packet.hasFlagPSH()){ this.fPSH_cnt++; } if(packet.hasFlagURG()){ this.fURG_cnt++; }"
- [claim | flags_binary] The eight flag-count columns and Fwd PSH/URG Flags take only the values 0 and 1. — holds on every source: yes (12 sources)
- [cicflowmeter-src-2018 | BasicFlow.java, field declaration] "private HashMap<String, MutableInt> flagCounts;"
- [cicflowmeter-src-2018 | BasicFlow.java, initFlags()] "flagCounts.put("FIN", new MutableInt()); flagCounts.put("SYN", new MutableInt()); flagCounts.put("RST", new MutableInt()); flagCounts.put("PSH", new MutableInt()); flagCounts.put("ACK", new MutableInt()); flagCounts.put("URG", new MutableInt()); flagCounts.put("CWR", new MutableInt()); flagCounts.put("ECE", new MutableInt());"
- [cicflowmeter-src-2018 | BasicFlow.java, CSV row writer] "for(String key: flagCounts.keySet()){ dump.append(flagCounts.get(key).value).append(separator); //50,51,52,53,54,55,56,57"
- [java8-hashmap | class description] "This class makes no guarantees as to the order of the map; in particular, it does not guarantee that the order will remain constant over time."
- [data_facts | flag_column_holds] = {"FIN Flag Cnt": "RST", "SYN Flag Cnt": "PSH", "RST Flag Cnt": "ECE", "PSH Flag Cnt": "SYN", "ACK Flag Cnt": "ACK", "URG Flag Cnt": "FIN", "CWE Flag Count": "URG", "ECE Flag Cnt": "CWR"}
- [claim | identical__Fwd PSH Flags__SYN Flag Cnt] 'Fwd PSH Flags' equals 'SYN Flag Cnt' in every row. — holds on every source: yes (12 sources)
- [claim | identical__Fwd URG Flags__CWE Flag Count] 'Fwd URG Flags' equals 'CWE Flag Count' in every row. — holds on every source: yes (12 sources)
- [claim | near_identical__RST Flag Cnt__ECE Flag Cnt] 'RST Flag Cnt' equals 'ECE Flag Cnt' in at least 99.9% of rows. — holds on every source: yes (12 sources)
- [cicflowmeter-readme | ReadMe.txt, list of extracted features] "URG Flag Count Number of packets with URG"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //55] "urg_cnt("URG Flag Cnt","URGCT""

### H1-CWE_Flag_Count

> CWE Flag Count: In CSE-CIC-IDS2018 this column is 1 if the first packet of the flow has the URG flag set and 0 otherwise. It describes that packet only and is not a count of packets: the extractor reads flags only from the first packet of a flow. The column name does not match its content: the extractor writes the eight flag values in the iteration order of a Java HashMap, which does not follow the order of the column names, so this column holds the URG flag and not the CWR flag.

- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "public void firstPacket(BasicPacketInfo packet){ updateFlowBulk(packet); detectUpdateSubflows(packet); checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), forward branch] "if(packet.hasFlagPSH()){ this.fPSH_cnt++; } if(packet.hasFlagURG()){ this.fURG_cnt++; }"
- [claim | flags_binary] The eight flag-count columns and Fwd PSH/URG Flags take only the values 0 and 1. — holds on every source: yes (12 sources)
- [cicflowmeter-src-2018 | BasicFlow.java, field declaration] "private HashMap<String, MutableInt> flagCounts;"
- [cicflowmeter-src-2018 | BasicFlow.java, initFlags()] "flagCounts.put("FIN", new MutableInt()); flagCounts.put("SYN", new MutableInt()); flagCounts.put("RST", new MutableInt()); flagCounts.put("PSH", new MutableInt()); flagCounts.put("ACK", new MutableInt()); flagCounts.put("URG", new MutableInt()); flagCounts.put("CWR", new MutableInt()); flagCounts.put("ECE", new MutableInt());"
- [cicflowmeter-src-2018 | BasicFlow.java, CSV row writer] "for(String key: flagCounts.keySet()){ dump.append(flagCounts.get(key).value).append(separator); //50,51,52,53,54,55,56,57"
- [java8-hashmap | class description] "This class makes no guarantees as to the order of the map; in particular, it does not guarantee that the order will remain constant over time."
- [data_facts | flag_column_holds] = {"FIN Flag Cnt": "RST", "SYN Flag Cnt": "PSH", "RST Flag Cnt": "ECE", "PSH Flag Cnt": "SYN", "ACK Flag Cnt": "ACK", "URG Flag Cnt": "FIN", "CWE Flag Count": "URG", "ECE Flag Cnt": "CWR"}
- [claim | identical__Fwd PSH Flags__SYN Flag Cnt] 'Fwd PSH Flags' equals 'SYN Flag Cnt' in every row. — holds on every source: yes (12 sources)
- [claim | identical__Fwd URG Flags__CWE Flag Count] 'Fwd URG Flags' equals 'CWE Flag Count' in every row. — holds on every source: yes (12 sources)
- [claim | near_identical__RST Flag Cnt__ECE Flag Cnt] 'RST Flag Cnt' equals 'ECE Flag Cnt' in at least 99.9% of rows. — holds on every source: yes (12 sources)
- [cicflowmeter-readme | ReadMe.txt, list of extracted features] "CWR Flag Count Number of packets with CWR"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //56] "cwe_cnt("CWE Flag Count","CWECT""

### H1-ECE_Flag_Cnt

> ECE Flag Cnt: In CSE-CIC-IDS2018 this column is 1 if the first packet of the flow has the CWR flag set and 0 otherwise. It describes that packet only and is not a count of packets: the extractor reads flags only from the first packet of a flow. The column name does not match its content: the extractor writes the eight flag values in the iteration order of a Java HashMap, which does not follow the order of the column names, so this column holds the CWR flag and not the ECE flag.

- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "public void firstPacket(BasicPacketInfo packet){ updateFlowBulk(packet); detectUpdateSubflows(packet); checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), forward branch] "if(packet.hasFlagPSH()){ this.fPSH_cnt++; } if(packet.hasFlagURG()){ this.fURG_cnt++; }"
- [claim | flags_binary] The eight flag-count columns and Fwd PSH/URG Flags take only the values 0 and 1. — holds on every source: yes (12 sources)
- [cicflowmeter-src-2018 | BasicFlow.java, field declaration] "private HashMap<String, MutableInt> flagCounts;"
- [cicflowmeter-src-2018 | BasicFlow.java, initFlags()] "flagCounts.put("FIN", new MutableInt()); flagCounts.put("SYN", new MutableInt()); flagCounts.put("RST", new MutableInt()); flagCounts.put("PSH", new MutableInt()); flagCounts.put("ACK", new MutableInt()); flagCounts.put("URG", new MutableInt()); flagCounts.put("CWR", new MutableInt()); flagCounts.put("ECE", new MutableInt());"
- [cicflowmeter-src-2018 | BasicFlow.java, CSV row writer] "for(String key: flagCounts.keySet()){ dump.append(flagCounts.get(key).value).append(separator); //50,51,52,53,54,55,56,57"
- [java8-hashmap | class description] "This class makes no guarantees as to the order of the map; in particular, it does not guarantee that the order will remain constant over time."
- [data_facts | flag_column_holds] = {"FIN Flag Cnt": "RST", "SYN Flag Cnt": "PSH", "RST Flag Cnt": "ECE", "PSH Flag Cnt": "SYN", "ACK Flag Cnt": "ACK", "URG Flag Cnt": "FIN", "CWE Flag Count": "URG", "ECE Flag Cnt": "CWR"}
- [claim | identical__Fwd PSH Flags__SYN Flag Cnt] 'Fwd PSH Flags' equals 'SYN Flag Cnt' in every row. — holds on every source: yes (12 sources)
- [claim | identical__Fwd URG Flags__CWE Flag Count] 'Fwd URG Flags' equals 'CWE Flag Count' in every row. — holds on every source: yes (12 sources)
- [claim | near_identical__RST Flag Cnt__ECE Flag Cnt] 'RST Flag Cnt' equals 'ECE Flag Cnt' in at least 99.9% of rows. — holds on every source: yes (12 sources)
- [cicflowmeter-readme | ReadMe.txt, list of extracted features] "ECE Flag Count Number of packets with ECE"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //57] "ece_cnt("ECE Flag Cnt","ECECT""

### H1-Fwd_Byts_per_b_Avg

> Fwd Byts/b Avg: In the published CSE-CIC-IDS2018 CSV files this column is always 0: bulk statistics are updated only for the first packet of a flow (the update is disabled for later packets), so no bulk transfer is ever recorded.

- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [claim | bulk_zero] The six bulk columns are 0 in every row. — holds on every source: yes (12 sources)
- [cicflowmeter-readme | ReadMe.txt, list of extracted features] "Fwd Bytes/Bulk Avg Average number of bytes bulk rate in the forward direction"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //63] "fw_byt_blk_avg("Fwd Byts/b Avg","FwB/BAG""

### H1-Fwd_Pkts_per_b_Avg

> Fwd Pkts/b Avg: In the published CSE-CIC-IDS2018 CSV files this column is always 0: bulk statistics are updated only for the first packet of a flow (the update is disabled for later packets), so no bulk transfer is ever recorded.

- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [claim | bulk_zero] The six bulk columns are 0 in every row. — holds on every source: yes (12 sources)
- [cicflowmeter-readme | ReadMe.txt, list of extracted features] "Fwd Packet/Bulk Avg Average number of packets bulk rate in the forward direction"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //64] "fw_pkt_blk_avg("Fwd Pkts/b Avg","FwP/BAG""

### H1-Fwd_Blk_Rate_Avg

> Fwd Blk Rate Avg: In the published CSE-CIC-IDS2018 CSV files this column is always 0: bulk statistics are updated only for the first packet of a flow (the update is disabled for later packets), so no bulk transfer is ever recorded.

- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [claim | bulk_zero] The six bulk columns are 0 in every row. — holds on every source: yes (12 sources)
- [cicflowmeter-readme | ReadMe.txt, list of extracted features] "Fwd Bulk Rate Avg Average number of bulk rate in the forward direction"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //65] "fw_blk_rate_avg("Fwd Blk Rate Avg","FwBRAG""

### H1-Bwd_Byts_per_b_Avg

> Bwd Byts/b Avg: In the published CSE-CIC-IDS2018 CSV files this column is always 0: bulk statistics are updated only for the first packet of a flow (the update is disabled for later packets), so no bulk transfer is ever recorded.

- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [claim | bulk_zero] The six bulk columns are 0 in every row. — holds on every source: yes (12 sources)
- [cicflowmeter-readme | ReadMe.txt, list of extracted features] "Bwd Bytes/Bulk Avg Average number of bytes bulk rate in the backward direction"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //66] "bw_byt_blk_avg("Bwd Byts/b Avg","BwB/BAG""

### H1-Bwd_Pkts_per_b_Avg

> Bwd Pkts/b Avg: In the published CSE-CIC-IDS2018 CSV files this column is always 0: bulk statistics are updated only for the first packet of a flow (the update is disabled for later packets), so no bulk transfer is ever recorded.

- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [claim | bulk_zero] The six bulk columns are 0 in every row. — holds on every source: yes (12 sources)
- [cicflowmeter-readme | ReadMe.txt, list of extracted features] "Bwd Packet/Bulk Avg Average number of packets bulk rate in the backward direction"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //67] "bw_pkt_blk_avg("Bwd Pkts/b Avg","BwP/BAG""

### H1-Bwd_Blk_Rate_Avg

> Bwd Blk Rate Avg: In the published CSE-CIC-IDS2018 CSV files this column is always 0: bulk statistics are updated only for the first packet of a flow (the update is disabled for later packets), so no bulk transfer is ever recorded.

- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [claim | bulk_zero] The six bulk columns are 0 in every row. — holds on every source: yes (12 sources)
- [cicflowmeter-readme | ReadMe.txt, list of extracted features] "Bwd Bulk Rate Avg Average number of bulk rate in the backward direction"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //68] "bw_blk_rate_avg("Bwd Blk Rate Avg","BwBRAG""

### H1-Init_Fwd_Win_Byts

> Init Fwd Win Byts: In CSE-CIC-IDS2018 this column is the raw TCP window field of a single packet header, not a byte count: for the forward direction it comes from the first packet of the flow, and for the backward direction from the most recent backward packet (the value is overwritten on each one). The field is recorded as read, without the window scale factor, so unless that packet is a SYN or SYN-ACK segment the real window can be larger than the recorded value. -1 is the extractor's default, kept when no TCP window was read: every UDP flow has -1 in the forward column, no TCP flow has -1 in the forward column, and no TCP flow with at least one backward packet has -1 in the backward column.

- [cicflowmeter-src-2018 | BasicPacketInfo.java, field declaration] "private int TCPWindow=-1;"
- [cicflowmeter-src-2018 | BasicFlow.java, field declarations] "private int Init_Win_bytes_forward=-1; private int Init_Win_bytes_backward=-1;"
- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setTCPWindow(tcp.window());"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), forward branch] "this.min_seg_size_forward = packet.getHeaderBytes(); Init_Win_bytes_forward = packet.getTCPWindow();"
- [cicflowmeter-src-2018 | BasicFlow.java, addPacket(), backward branch] "this.bwdPktStats.addValue((double)packet.getPayloadBytes()); Init_Win_bytes_backward = packet.getTCPWindow();"
- [rfc7323 | Section 2.2] "The window field in a segment where the SYN bit is set (i.e., a <SYN> or <SYN,ACK>) MUST NOT be scaled."
- [rfc9293 | Section 3.1, Header Format] "The value is shifted when the window scaling extension is used [47]."
- [claim | init_fwd_win_tcp_16bit] Init Fwd Win Byts of TCP rows lies in [0, 65535], the range of a 16-bit field. — holds on every source: yes (12 sources)
- [claim | udp_flags_zero_init_fwd_minus1] Every UDP row has all flag-count columns 0 and Init Fwd Win Byts = -1. — holds on every source: yes (12 sources)
- [claim | tcp_init_fwd_never_minus1] No TCP row has Init Fwd Win Byts = -1. — holds on every source: yes (12 sources)
- [claim | tcp_with_bwd_init_bwd_never_minus1] No TCP row with at least one backward packet has Init Bwd Win Byts = -1. — holds on every source: yes (12 sources)
- [cicflowmeter-readme | ReadMe.txt, list of extracted features] "Fwd Init Win bytes The total number of bytes sent in initial window in the forward direction"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //73] "fw_win_byt("Init Fwd Win Byts","FwWB""

### H1-Init_Bwd_Win_Byts

> Init Bwd Win Byts: In CSE-CIC-IDS2018 this column is the raw TCP window field of a single packet header, not a byte count: for the forward direction it comes from the first packet of the flow, and for the backward direction from the most recent backward packet (the value is overwritten on each one). The field is recorded as read, without the window scale factor, so unless that packet is a SYN or SYN-ACK segment the real window can be larger than the recorded value. -1 is the extractor's default, kept when no TCP window was read: every UDP flow has -1 in the forward column, no TCP flow has -1 in the forward column, and no TCP flow with at least one backward packet has -1 in the backward column.

- [cicflowmeter-src-2018 | BasicPacketInfo.java, field declaration] "private int TCPWindow=-1;"
- [cicflowmeter-src-2018 | BasicFlow.java, field declarations] "private int Init_Win_bytes_forward=-1; private int Init_Win_bytes_backward=-1;"
- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setTCPWindow(tcp.window());"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), forward branch] "this.min_seg_size_forward = packet.getHeaderBytes(); Init_Win_bytes_forward = packet.getTCPWindow();"
- [cicflowmeter-src-2018 | BasicFlow.java, addPacket(), backward branch] "this.bwdPktStats.addValue((double)packet.getPayloadBytes()); Init_Win_bytes_backward = packet.getTCPWindow();"
- [rfc7323 | Section 2.2] "The window field in a segment where the SYN bit is set (i.e., a <SYN> or <SYN,ACK>) MUST NOT be scaled."
- [rfc9293 | Section 3.1, Header Format] "The value is shifted when the window scaling extension is used [47]."
- [claim | init_fwd_win_tcp_16bit] Init Fwd Win Byts of TCP rows lies in [0, 65535], the range of a 16-bit field. — holds on every source: yes (12 sources)
- [claim | udp_flags_zero_init_fwd_minus1] Every UDP row has all flag-count columns 0 and Init Fwd Win Byts = -1. — holds on every source: yes (12 sources)
- [claim | tcp_init_fwd_never_minus1] No TCP row has Init Fwd Win Byts = -1. — holds on every source: yes (12 sources)
- [claim | tcp_with_bwd_init_bwd_never_minus1] No TCP row with at least one backward packet has Init Bwd Win Byts = -1. — holds on every source: yes (12 sources)
- [cicflowmeter-readme | ReadMe.txt, list of extracted features] "Bwd Init Win bytes The total number of bytes sent in initial window in the backward direction"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //74] "bw_win_byt("Init Bwd Win Byts","BwWB""

### H1-Fwd_Act_Data_Pkts

> Fwd Act Data Pkts: In CSE-CIC-IDS2018 this column counts the forward packets after the first one that carry at least 1 byte of payload; the first packet of the flow is never counted. The value is therefore at most the forward packet count minus 1, and a flow whose only forward payload is in its first packet shows 0.

- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "if(packet.getPayloadBytes() >=1){ this.Act_data_pkt_forward++; }"
- [claim | act_data_le_fwd_minus1] Fwd Act Data Pkts is at most Tot Fwd Pkts - 1 in every row. — holds on every source: yes (12 sources)
- [cicflowmeter-readme | ReadMe.txt, list of extracted features] "Fwd Act Data Pkts Count of packets with at least 1 byte of TCP data payload in the forward direction"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //75] "Fw_act_pkt("Fwd Act Data Pkts","FwAP""

### H1-Fwd_Seg_Size_Min

> Fwd Seg Size Min: In CSE-CIC-IDS2018 this column is the smallest transport-header length, in bytes, among the forward packets of the flow (between 20 and 60 for TCP, 8 for UDP); it is not a payload or segment size.

- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "this.min_seg_size_forward = Math.min(packet.getHeaderBytes(),this.min_seg_size_forward);"
- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setHeaderBytes(tcp.getHeaderLength());"
- [claim | seg_size_min_is_header] Fwd Seg Size Min is between 20 and 60 for TCP rows and 8 for UDP rows. — holds on every source: yes (12 sources)
- [cicflowmeter-readme | ReadMe.txt, list of extracted features] "Fwd Seg Size Min Minimum segment size observed in the forward direction"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //76] "fw_seg_min("Fwd Seg Size Min","FwSgMI""

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

### H3-NOWIN

> A value of -1 in an initial-window column is not a window size: the TCP window field is a 16-bit count of octets. It is the extractor's default, kept when no TCP window was read for that direction: every UDP flow has -1 in the forward column, and no TCP flow with at least one backward packet has -1 in the backward column.

- [rfc9293 | Section 3.1, Header Format] "Window: 16 bits The number of data octets beginning with the one indicated in the acknowledgment field that the sender of this segment is willing to accept. The value is shifted when the window scaling extension is used [47]."
- [cicflowmeter-src-2018 | BasicPacketInfo.java, field declaration] "private int TCPWindow=-1;"
- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setTCPWindow(tcp.window());"
- [claim | udp_flags_zero_init_fwd_minus1] Every UDP row has all flag-count columns 0 and Init Fwd Win Byts = -1. — holds on every source: yes (12 sources)
- [claim | tcp_with_bwd_init_bwd_never_minus1] No TCP row with at least one backward packet has Init Bwd Win Byts = -1. — holds on every source: yes (12 sources)

### H3-WINSCALE

> The first packet of this TCP flow is not a SYN segment, so the initial-window columns hold window fields of segments sent after the connection was opened. Such a field is not in bytes when window scaling is in use: the scale factor is agreed only in the SYN segments and every later window field is shifted by it, so the real window can be larger than the recorded value. Only a window field carried in a SYN or SYN-ACK segment is never scaled.

- [rfc7323 | Section 2.1] "This option is sent only in a <SYN> segment (a segment with the SYN bit on), hence the window scale is fixed in each direction when a connection is opened."
- [rfc7323 | Section 2.2] "The window field in a segment where the SYN bit is set (i.e., a <SYN> or <SYN,ACK>) MUST NOT be scaled."
- [rfc9293 | Section 3.1, Header Format] "The value is shifted when the window scaling extension is used [47]."
- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setTCPWindow(tcp.window());"
- [data_facts | flag_column_holds] = {"FIN Flag Cnt": "RST", "SYN Flag Cnt": "PSH", "RST Flag Cnt": "ECE", "PSH Flag Cnt": "SYN", "ACK Flag Cnt": "ACK", "URG Flag Cnt": "FIN", "CWE Flag Count": "URG", "ECE Flag Cnt": "CWR"}
- [claim | init_fwd_win_tcp_16bit] Init Fwd Win Byts of TCP rows lies in [0, 65535], the range of a 16-bit field. — holds on every source: yes (12 sources)

### H3-TIMEOUT

> The flow lasted until the extractor's 120 s flow timeout. A longer connection is split into several consecutive flows, so this duration does not cover the whole session.

- [cicflowmeter-src-2018 | CICFlowMeter.java, main()] "flowGen = new FlowGenerator(true,120000000L, 5000000L);"
- [cicflowmeter-src-2018 | FlowGenerator.java, addPacket()] "// Flow finished due flowtimeout: // 1.- we move the flow to finished flow list // 2.- we eliminate the flow from the current flow list // 3.- we create a new flow with the packet-in-process"
- [claim | duration_le_120s] Flow Duration never exceeds 120,000,000 microseconds. — holds on every source: yes (12 sources)

### H3-UDP

> This is a UDP flow. The extractor reads flags and the window only from TCP headers, so for UDP the flag columns stay 0 and the initial-window columns stay -1; they carry no information about this flow.

- [cicflowmeter-readme | ReadMe.txt, list of extracted features] "Fwd PSH flags Number of times the PSH flag was set in packets travelling in the forward direction (0 for UDP)"
- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setFlagRST(tcp.flags_RST());"
- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setTCPWindow(tcp.window());"
- [claim | udp_flags_zero_init_fwd_minus1] Every UDP row has all flag-count columns 0 and Init Fwd Win Byts = -1. — holds on every source: yes (12 sources)

### H3-NOTRANSPORT

> Protocol 0 here means the packets carried neither a TCP nor a UDP header: the extractor sets the protocol only to 6 (TCP) or 17 (UDP) and otherwise keeps the Java default value 0, and ports and payload are then also 0. It does not refer to IANA protocol number 0 (HOPOPT).

- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setProtocol(6);"
- [cicflowmeter-src-2018 | PacketReader.java, UDP branch] "packetInfo.setProtocol(17);"
- [cicflowmeter-src-2018 | BasicPacketInfo.java, field declarations] "private int protocol;"
- [java8-jls | Section 4.12.5, Initial Values of Variables] "For type int, the default value is zero, that is, 0."
- [iana-protocols | protocol-numbers-1.csv, row 0] "0,HOPOPT,IPv6 Hop-by-Hop Option"
- [claim | protocol0_port0_no_payload] Every row with Protocol 0 has Dst Port 0 and no payload. — holds on every source: yes (12 sources)

### Notes attached to H1 and H2 passages

**XN-TIME-MICROSECONDS** (on: Flow Duration, Flow IAT Mean, Flow IAT Std, Flow IAT Max, Flow IAT Min, Fwd IAT Tot, Fwd IAT Mean, Fwd IAT Std, Fwd IAT Max, Fwd IAT Min, Bwd IAT Tot, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Bwd IAT Min, Active Mean, Active Std, Active Max, Active Min, Idle Mean, Idle Std, Idle Max, Idle Min)

> Times are in microseconds: the extractor reads packet timestamps in microseconds.

- [cicflowmeter-src-2018 | PacketReader.java, IPv4 packet parsing] "packetInfo.setTimeStamp(packet.getCaptureHeader().timestampInMicros());"
- [cicflowmeter-readme | ReadMe.txt, list of extracted features] "Flow duration Duration of the flow in Microsecond"

**XN-FLOW-TIMEOUT** (on: Flow Duration)

> The command-line extractor closes a flow 120 s (120,000,000 microseconds) after its first packet and starts a new flow; no flow in the published CSE-CIC-IDS2018 CSV files lasts longer than that.

- [cicflowmeter-src-2018 | CICFlowMeter.java, main()] "flowGen = new FlowGenerator(true,120000000L, 5000000L);"
- [cicflowmeter-src-2018 | FlowGenerator.java, addPacket()] "if((currentTimestamp -flow.getFlowStartTime())>flowTimeOut){"
- [claim | duration_le_120s] Flow Duration never exceeds 120,000,000 microseconds. — holds on every source: yes (12 sources)

**DATA-IDENTICAL** (on: Tot Fwd Pkts)

> In the published CSE-CIC-IDS2018 CSV files this column is identical to 'Subflow Fwd Pkts' in every row.

- [claim | identical__Tot Fwd Pkts__Subflow Fwd Pkts] 'Tot Fwd Pkts' equals 'Subflow Fwd Pkts' in every row. — holds on every source: yes (12 sources)

**DATA-IDENTICAL** (on: Tot Bwd Pkts)

> In the published CSE-CIC-IDS2018 CSV files this column is identical to 'Subflow Bwd Pkts' in every row.

- [claim | identical__Tot Bwd Pkts__Subflow Bwd Pkts] 'Tot Bwd Pkts' equals 'Subflow Bwd Pkts' in every row. — holds on every source: yes (12 sources)

**XN-PAYLOAD-BYTES** (on: TotLen Fwd Pkts, TotLen Bwd Pkts, Fwd Pkt Len Max, Fwd Pkt Len Min, Fwd Pkt Len Mean, Fwd Pkt Len Std, Bwd Pkt Len Max, Bwd Pkt Len Min, Bwd Pkt Len Mean, Bwd Pkt Len Std, Pkt Len Min, Pkt Len Max, Pkt Len Mean, Pkt Len Std, Pkt Len Var, Pkt Size Avg, Fwd Seg Size Avg, Bwd Seg Size Avg, Subflow Fwd Byts, Subflow Bwd Byts)

> Lengths count transport-layer payload bytes (the TCP or UDP payload), not packet headers.

- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setPayloadBytes(tcp.getPayloadLength());"
- [cicflowmeter-src-2018 | PacketReader.java, UDP branch] "packetInfo.setPayloadBytes(udp.getPayloadLength());"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "this.fwdPktStats.addValue((double)packet.getPayloadBytes());"
- [cicflowmeter-src-2018 | BasicFlow.java, getAvgPacketSize()] "return (this.flowLengthStats.getSum()/this.packetCount());"
- [cicflowmeter-src-2018 | BasicFlow.java, fAvgSegmentSize()] "return (this.fwdPktStats.getSum() / (double)this.forward.size());"

**DATA-IDENTICAL** (on: TotLen Fwd Pkts)

> In the published CSE-CIC-IDS2018 CSV files this column is identical to 'Subflow Fwd Byts' in every row.

- [claim | identical__TotLen Fwd Pkts__Subflow Fwd Byts] 'TotLen Fwd Pkts' equals 'Subflow Fwd Byts' in every row. — holds on every source: yes (12 sources)

**DATA-NEAR-IDENTICAL** (on: TotLen Bwd Pkts)

> In the published CSE-CIC-IDS2018 CSV files this column equals 'Subflow Bwd Byts' in at least 99.9% of rows.

- [claim | near_identical__TotLen Bwd Pkts__Subflow Bwd Byts] 'TotLen Bwd Pkts' equals 'Subflow Bwd Byts' in at least 99.9% of rows. — holds on every source: yes (12 sources)

**DATA-IDENTICAL** (on: Fwd PSH Flags)

> In the published CSE-CIC-IDS2018 CSV files this column is identical to 'SYN Flag Cnt' in every row.

- [claim | identical__Fwd PSH Flags__SYN Flag Cnt] 'Fwd PSH Flags' equals 'SYN Flag Cnt' in every row. — holds on every source: yes (12 sources)

**DATA-IDENTICAL** (on: Fwd URG Flags)

> In the published CSE-CIC-IDS2018 CSV files this column is identical to 'CWE Flag Count' in every row.

- [claim | identical__Fwd URG Flags__CWE Flag Count] 'Fwd URG Flags' equals 'CWE Flag Count' in every row. — holds on every source: yes (12 sources)

**XN-FIRST-PACKET-TWICE** (on: Pkt Len Mean, Pkt Len Std, Pkt Len Var, Pkt Size Avg)

> The extractor adds the payload length of the flow's first packet twice to the statistics over all packets, so the first packet counts double: the mean, standard deviation and variance are taken over one more value than the flow has packets, and the average packet size divides that sum by the real packet count.

- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "this.endActiveTime = packet.getTimeStamp(); this.flowLengthStats.addValue((double)packet.getPayloadBytes());"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), forward branch] "Init_Win_bytes_forward = packet.getTCPWindow(); this.flowLengthStats.addValue((double)packet.getPayloadBytes());"
- [cicflowmeter-src-2018 | BasicFlow.java, getAvgPacketSize()] "return (this.flowLengthStats.getSum()/this.packetCount());"
- [claim | pkt_len_double_count] Pkt Size Avg x (Tot Fwd Pkts + Tot Bwd Pkts) equals Pkt Len Mean x (Tot Fwd Pkts + Tot Bwd Pkts + 1) in every row. — holds on every source: yes (12 sources)

**DATA-IDENTICAL** (on: SYN Flag Cnt)

> In the published CSE-CIC-IDS2018 CSV files this column is identical to 'Fwd PSH Flags' in every row.

- [claim | identical__Fwd PSH Flags__SYN Flag Cnt] 'Fwd PSH Flags' equals 'SYN Flag Cnt' in every row. — holds on every source: yes (12 sources)

**DATA-NEAR-IDENTICAL** (on: RST Flag Cnt)

> In the published CSE-CIC-IDS2018 CSV files this column equals 'ECE Flag Cnt' in at least 99.9% of rows.

- [claim | near_identical__RST Flag Cnt__ECE Flag Cnt] 'RST Flag Cnt' equals 'ECE Flag Cnt' in at least 99.9% of rows. — holds on every source: yes (12 sources)

**XN-CWE-NAME** (on: CWE Flag Count)

> 'CWE' in this column name is the 2018 extractor's name for its CWR counter; later versions of the extractor call the column 'CWR Flag Count'.

- [cicflowmeter-src-2018 | FlowFeature.java, feature //56] "cwe_cnt("CWE Flag Count","CWECT""
- [cicflowmeter-readme | FlowFeature.java (master), feature //56] "CWR_cnt("CWR Flag Count","CWRCT""

**DATA-IDENTICAL** (on: CWE Flag Count)

> In the published CSE-CIC-IDS2018 CSV files this column is identical to 'Fwd URG Flags' in every row.

- [claim | identical__Fwd URG Flags__CWE Flag Count] 'Fwd URG Flags' equals 'CWE Flag Count' in every row. — holds on every source: yes (12 sources)

**DATA-NEAR-IDENTICAL** (on: ECE Flag Cnt)

> In the published CSE-CIC-IDS2018 CSV files this column equals 'RST Flag Cnt' in at least 99.9% of rows.

- [claim | near_identical__RST Flag Cnt__ECE Flag Cnt] 'RST Flag Cnt' equals 'ECE Flag Cnt' in at least 99.9% of rows. — holds on every source: yes (12 sources)

**XN-DOWNUP-INTEGER** (on: Down/Up Ratio)

> The extractor divides the backward packet count by the forward packet count with integer division, so the fraction is dropped: the value is 0 whenever the flow has fewer backward than forward packets, and it never has a fractional part.

- [cicflowmeter-src-2018 | BasicFlow.java, getDownUpRatio()] "return (double)(this.backward.size()/this.forward.size());"
- [claim | downup_integer] Down/Up Ratio equals floor(Tot Bwd Pkts / Tot Fwd Pkts) in every row. — holds on every source: yes (12 sources)

**XN-SUBFLOW-ONE** (on: Subflow Fwd Pkts, Subflow Fwd Byts, Subflow Bwd Pkts, Subflow Bwd Byts)

> Subflow detection runs only for the first packet of a flow, so every flow counts as a single subflow and these columns repeat the flow totals.

- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, getSflow_fpackets()] "return this.forward.size()/sfCount;"
- [claim | identical__Tot Fwd Pkts__Subflow Fwd Pkts] 'Tot Fwd Pkts' equals 'Subflow Fwd Pkts' in every row. — holds on every source: yes (12 sources)
- [claim | identical__Tot Bwd Pkts__Subflow Bwd Pkts] 'Tot Bwd Pkts' equals 'Subflow Bwd Pkts' in every row. — holds on every source: yes (12 sources)
- [claim | identical__TotLen Fwd Pkts__Subflow Fwd Byts] 'TotLen Fwd Pkts' equals 'Subflow Fwd Byts' in every row. — holds on every source: yes (12 sources)
- [claim | near_identical__TotLen Bwd Pkts__Subflow Bwd Byts] 'TotLen Bwd Pkts' equals 'Subflow Bwd Byts' in at least 99.9% of rows. — holds on every source: yes (12 sources)

**DATA-IDENTICAL** (on: Subflow Fwd Pkts)

> In the published CSE-CIC-IDS2018 CSV files this column is identical to 'Tot Fwd Pkts' in every row.

- [claim | identical__Tot Fwd Pkts__Subflow Fwd Pkts] 'Tot Fwd Pkts' equals 'Subflow Fwd Pkts' in every row. — holds on every source: yes (12 sources)

**DATA-IDENTICAL** (on: Subflow Fwd Byts)

> In the published CSE-CIC-IDS2018 CSV files this column is identical to 'TotLen Fwd Pkts' in every row.

- [claim | identical__TotLen Fwd Pkts__Subflow Fwd Byts] 'TotLen Fwd Pkts' equals 'Subflow Fwd Byts' in every row. — holds on every source: yes (12 sources)

**DATA-IDENTICAL** (on: Subflow Bwd Pkts)

> In the published CSE-CIC-IDS2018 CSV files this column is identical to 'Tot Bwd Pkts' in every row.

- [claim | identical__Tot Bwd Pkts__Subflow Bwd Pkts] 'Tot Bwd Pkts' equals 'Subflow Bwd Pkts' in every row. — holds on every source: yes (12 sources)

**DATA-NEAR-IDENTICAL** (on: Subflow Bwd Byts)

> In the published CSE-CIC-IDS2018 CSV files this column equals 'TotLen Bwd Pkts' in at least 99.9% of rows.

- [claim | near_identical__TotLen Bwd Pkts__Subflow Bwd Byts] 'TotLen Bwd Pkts' equals 'Subflow Bwd Byts' in at least 99.9% of rows. — holds on every source: yes (12 sources)

**PORT-NOT-SERVICE** (on: H2-PORT-21, H2-PORT-22, H2-PORT-53, H2-PORT-80, H2-PORT-135, H2-PORT-443, H2-PORT-445, H2-PORT-500, H2-PORT-3389, H2-PORT-8080)

> The registration does not show which service a flow actually uses: the fact that network traffic is flowing to or from an assigned port number does not mean that it is "good" traffic, or even that it is used by the assigned service.

- [rfc6335 | Section 9, Security Considerations] "the fact that network traffic is flowing to or from an assigned port number does not mean that it is "good" traffic, or even that it is used by the assigned service."

## 5. H3 conditions: share of flows per label (train_full.csv)

Descriptive only: training split, not checked for stationarity, not used as evidence in any passage.

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

## 6b. ReadMe definitions that do not hold for CSE-CIC-IDS2018

These columns start with their real meaning; the ReadMe definition is kept only in authored/feature_names.yaml and in the passage support.

| column | ReadMe definition (not sent to the VLM) |
|---|---|
| Fwd PSH Flags | Number of times the PSH flag was set in packets travelling in the forward direction (0 for UDP) |
| Bwd PSH Flags | Number of times the PSH flag was set in packets travelling in the backward direction (0 for UDP) |
| Fwd URG Flags | Number of times the URG flag was set in packets travelling in the forward direction (0 for UDP) |
| Bwd URG Flags | Number of times the URG flag was set in packets travelling in the backward direction (0 for UDP) |
| FIN Flag Cnt | Number of packets with FIN |
| SYN Flag Cnt | Number of packets with SYN |
| RST Flag Cnt | Number of packets with RST |
| PSH Flag Cnt | Number of packets with PUSH |
| ACK Flag Cnt | Number of packets with ACK |
| URG Flag Cnt | Number of packets with URG |
| CWE Flag Count | Number of packets with CWR |
| ECE Flag Cnt | Number of packets with ECE |
| Fwd Byts/b Avg | Average number of bytes bulk rate in the forward direction |
| Fwd Pkts/b Avg | Average number of packets bulk rate in the forward direction |
| Fwd Blk Rate Avg | Average number of bulk rate in the forward direction |
| Bwd Byts/b Avg | Average number of bytes bulk rate in the backward direction |
| Bwd Pkts/b Avg | Average number of packets bulk rate in the backward direction |
| Bwd Blk Rate Avg | Average number of bulk rate in the backward direction |
| Init Fwd Win Byts | The total number of bytes sent in initial window in the forward direction |
| Init Bwd Win Byts | The total number of bytes sent in initial window in the backward direction |
| Fwd Act Data Pkts | Count of packets with at least 1 byte of TCP data payload in the forward direction |
| Fwd Seg Size Min | Minimum segment size observed in the forward direction |

## 7. Claims and stationarity (build/claims.py)

A number enters a passage sent to the VLM only as a claim that holds on the training split, the validation split and every published CSV file (and both halves of each label within a file, by row order).

| claim | statement | test | train | val | min / max over raw files and halves | holds |
|---|---|---|---|---|---|---|
| flags_binary | The eight flag-count columns and Fwd PSH/URG Flags take only the values 0 and 1. | = 1 | 1 | 1 | 1 / 1 | yes |
| bulk_zero | The six bulk columns are 0 in every row. | = 1 | 1 | 1 | 1 / 1 | yes |
| bwd_dir_flags_zero | Bwd PSH Flags and Bwd URG Flags are 0 in every row. | = 1 | 1 | 1 | 1 / 1 | yes |
| pkt_len_double_count | Pkt Size Avg x (Tot Fwd Pkts + Tot Bwd Pkts) equals Pkt Len Mean x (Tot Fwd Pkts + Tot Bwd Pkts + 1) in every row. | = 1 | 1 | 1 | 1 / 1 | yes |
| downup_integer | Down/Up Ratio equals floor(Tot Bwd Pkts / Tot Fwd Pkts) in every row. | = 1 | 1 | 1 | 1 / 1 | yes |
| duration_le_120s | Flow Duration never exceeds 120,000,000 microseconds. | = 1 | 1 | 1 | 1 / 1 | yes |
| protocol0_port0_no_payload | Every row with Protocol 0 has Dst Port 0 and no payload. | = 1 | 1 | 1 | 1 / 1 | yes |
| tcp_init_fwd_never_minus1 | No TCP row has Init Fwd Win Byts = -1. | = 1 | 1 | 1 | 1 / 1 | yes |
| tcp_with_bwd_init_bwd_never_minus1 | No TCP row with at least one backward packet has Init Bwd Win Byts = -1. | = 1 | 1 | 1 | 1 / 1 | yes |
| udp_flags_zero_init_fwd_minus1 | Every UDP row has all flag-count columns 0 and Init Fwd Win Byts = -1. | = 1 | 1 | 1 | 1 / 1 | yes |
| act_data_le_fwd_minus1 | Fwd Act Data Pkts is at most Tot Fwd Pkts - 1 in every row. | = 1 | 1 | 1 | 1 / 1 | yes |
| seg_size_min_is_header | Fwd Seg Size Min is between 20 and 60 for TCP rows and 8 for UDP rows. | = 1 | 1 | 1 | 1 / 1 | yes |
| init_fwd_win_tcp_16bit | Init Fwd Win Byts of TCP rows lies in [0, 65535], the range of a 16-bit field. | = 1 | 1 | 1 | 1 / 1 | yes |
| identical__Tot Fwd Pkts__Subflow Fwd Pkts | 'Tot Fwd Pkts' equals 'Subflow Fwd Pkts' in every row. | = 1 | 1 | 1 | 1 / 1 | yes |
| identical__Tot Bwd Pkts__Subflow Bwd Pkts | 'Tot Bwd Pkts' equals 'Subflow Bwd Pkts' in every row. | = 1 | 1 | 1 | 1 / 1 | yes |
| identical__TotLen Fwd Pkts__Subflow Fwd Byts | 'TotLen Fwd Pkts' equals 'Subflow Fwd Byts' in every row. | = 1 | 1 | 1 | 1 / 1 | yes |
| identical__Fwd PSH Flags__SYN Flag Cnt | 'Fwd PSH Flags' equals 'SYN Flag Cnt' in every row. | = 1 | 1 | 1 | 1 / 1 | yes |
| identical__Fwd URG Flags__CWE Flag Count | 'Fwd URG Flags' equals 'CWE Flag Count' in every row. | = 1 | 1 | 1 | 1 / 1 | yes |
| near_identical__TotLen Bwd Pkts__Subflow Bwd Byts | 'TotLen Bwd Pkts' equals 'Subflow Bwd Byts' in at least 99.9% of rows. | >= 0.999 | 1 | 1 | 0.9999 / 1 | yes |
| near_identical__RST Flag Cnt__ECE Flag Cnt | 'RST Flag Cnt' equals 'ECE Flag Cnt' in at least 99.9% of rows. | >= 0.999 | 1 | 1 | 1 / 1 | yes |
| constant__Bwd PSH Flags | 'Bwd PSH Flags' is 0 in every row. | = 1 | 1 | 1 | 1 / 1 | yes |
| constant__Bwd URG Flags | 'Bwd URG Flags' is 0 in every row. | = 1 | 1 | 1 | 1 / 1 | yes |
| constant__Fwd Byts/b Avg | 'Fwd Byts/b Avg' is 0 in every row. | = 1 | 1 | 1 | 1 / 1 | yes |
| constant__Fwd Pkts/b Avg | 'Fwd Pkts/b Avg' is 0 in every row. | = 1 | 1 | 1 | 1 / 1 | yes |
| constant__Fwd Blk Rate Avg | 'Fwd Blk Rate Avg' is 0 in every row. | = 1 | 1 | 1 | 1 / 1 | yes |
| constant__Bwd Byts/b Avg | 'Bwd Byts/b Avg' is 0 in every row. | = 1 | 1 | 1 | 1 / 1 | yes |
| constant__Bwd Pkts/b Avg | 'Bwd Pkts/b Avg' is 0 in every row. | = 1 | 1 | 1 | 1 / 1 | yes |
| constant__Bwd Blk Rate Avg | 'Bwd Blk Rate Avg' is 0 in every row. | = 1 | 1 | 1 | 1 / 1 | yes |
| twin_tcp21_nopayload__FTP_BruteForce | Every FTP_BruteForce flow goes to TCP port 21 and carries no payload. | = 1 | 1 | 1 | 1 / 1 | yes |
| twin_tcp21_nopayload__DoS_attacks_SlowHTTPTest | Every DoS_attacks_SlowHTTPTest flow goes to TCP port 21 and carries no payload. | = 1 | 1 | 1 | 1 / 1 | yes |
| loic_udp_udp_port80 | Every DDOS_attack_LOIC_UDP flow is UDP to port 80. | = 1 | 1 | 1 | 1 / 1 | yes |
| port80_all__DoS_attacks_GoldenEye | Every DoS_attacks_GoldenEye flow goes to port 80. | = 1 | 1 | 1 | 1 / 1 | yes |
| port80_all__DoS_attacks_Hulk | Every DoS_attacks_Hulk flow goes to port 80. | = 1 | 1 | 1 | 1 / 1 | yes |
| port80_all__DoS_attacks_Slowloris | Every DoS_attacks_Slowloris flow goes to port 80. | = 1 | 1 | 1 | 1 / 1 | yes |
| port80_all__DDoS_attacks_LOIC_HTTP | Every DDoS_attacks_LOIC_HTTP flow goes to port 80. | = 1 | 1 | 1 | 1 / 1 | yes |
| port80_all__DDOS_attack_HOIC | Every DDOS_attack_HOIC flow goes to port 80. | = 1 | 1 | 1 | 1 / 1 | yes |
| port80_all__SQL_Injection | Every SQL_Injection flow goes to port 80. | = 1 | 1 | 1 | 1 / 1 | yes |
| ssh_port22_ge999 | At least 99.9% of SSH_Bruteforce flows go to port 22. | >= 0.999 | 1 | 0.9998 | 0.9999 / 1 | yes |
| bot_tcp8080_gt95 | More than 95% of Bot flows go to TCP port 8080. | > 0.95 | 0.9667 | 0.9708 | 0.961 / 0.976 | yes |
| xss_port80_ge90 | At least 90% of Brute_Force_XSS flows go to port 80. | >= 0.90 | 0.9835 | 0.9565 | 0.9333 / 1 | yes |
| bfw_port80_top | Port 80 is the most common destination port of Brute_Force_Web flows. | = 80 | 80 | 80 | 80 / 80 | yes |
| bfw_has_udp_and_other_ports | The Brute_Force_Web label contains UDP flows and flows to ports other than 80 (label level: training split, validation split, and the published CSV files taken together). | = 1 | 1 | 1 | 1 / 1 | yes |
| infil_udp53_ge10 | At least 10% of Infiltration flows are UDP to port 53 (DNS). | >= 0.10 | 0.2916 | 0.303 | 0.2225 / 0.3636 | yes |
| infil_tcp443_ge10 | At least 10% of Infiltration flows are TCP to port 443 (HTTPS). | >= 0.10 | 0.2029 | 0.194 | 0.1277 / 0.3026 | yes |
| infil_max_port_lt50 | No single destination port carries half of the Infiltration flows. | < 0.5 | 0.2917 | 0.304 | 0.2228 / 0.3637 | yes |
