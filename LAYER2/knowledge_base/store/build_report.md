# Knowledge base build report

Built 2026-11-14 | passages.jsonl sha256 `1e846acf4aaea9b9d0d74d855eaf59cb90d07c9907e729dc943204c8d38d67d6`

Validation: **passed** (every quote verbatim in its locked source, Borisenko key matches, coverage complete)


## 1. Sources

| id | tier | version | retrieved | files (sha256) |
|---|---|---|---|---|
| unb-ids2018 | dataset | page retrieved on the date in sources.lock.json | 2026-11-13 | page.html `c1258503f336` |
| cicflowmeter-readme | tool | git 98a5ebad0df579cc8b43eedd3421b3ae87699901 (master, 2022-01-05) | 2026-11-13 | ReadMe.txt `c8919171add3`; LICENSE.txt `f60c54b998ca`; FlowFeature.java `fadddf4ff18d` |
| cicflowmeter-src-2018 | tool | git 50d668db9bdb0a5fae457916dae4538ba8b41876 (2018-03-07) | 2026-11-13 | FlowFeature.java `941611e67a7a`; BasicFlow.java `7519720889fe`; BasicPacketInfo.java `eff86c02d123`; PacketReader.java `776a9234941c`; FlowGenerator.java `c760789268ba`; CICFlowMeter.java `9b74a25ae292`; DateFormatter.java `b50d3489f41e`; FlowOfflinePane.java `37bf07e7d4ac`; ReadPcapFileWorker.java `049bd42c3c8c`; TrafficFlowWorker.java `201da46d4d66`; build.gradle `2f09973fe4ec` |
| cicflowmeter-src-2018-06 | tool | git 6833ccd4ff2953fdd2ab703060d43e349494151e (2018-06-04) | 2026-11-13 | BasicFlow.java `797fb5370c35` |
| openjdk8-src | platform | jdk8-b132 | 2026-11-13 | HashMap.java `78dc23b1dc93`; String.java `af751f86c9bd` |
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
| layer1-code | system | git: commit of each file recorded in sources.lock.json | 2026-11-14 | autoencoder.py `0321d6c1fd5a`; train_protocol_b.py `a873c3c549c2` |
| borisenko-2022 | mapping | published version (IEEE Xplore) | 2026-11-14 | borisenko2022.pdf `d99c624b042f` |
| engelen-2021-doc | audit | page retrieved on the date in sources.lock.json | 2026-11-14 | extended_doc.html `770d19e58d78` |

## 2. Passages per part

| part | passages | verbatim | rendered | defined | sent to VLM |
|---|---|---|---|---|---|
| M  label -> technique (Borisenko et al. 2022, Table II) | 14 | 0 | 14 | 0 | yes |
| B  technique knowledge (ATT&CK v20.1, CAPEC v3.9) | 19 | 16 | 3 | 0 | yes |
| H1 feature definitions (CICFlowMeter-V3 2018) | 81 | 1 | 54 | 26 | yes |
| H2 protocol / port / TCP flag (IANA, IETF) | 6077 | 10 | 6067 | 0 | yes |
| H3 conditional extractor artefacts | 7 | 0 | 0 | 7 | yes |
| G  mitigation candidates (ATT&CK mitigates + CAPEC Mitigations), chosen by the VLM | 40 | 15 | 25 | 0 | yes |

## 3. Label -> technique key (Borisenko et al. 2022, Table II)

| label | Table II row (verbatim) | technique(s) used | CAPEC linked | network analytics |
|---|---|---|---|---|
| FTP_BruteForce | 1 FTP- BruteForce Credential Access (TA0006) Brute Force (T1110) Password Guessing (T1110.001) FTP Patator | T1110.001 Password Guessing | T1110.001: CAPEC-49 | T1110.001: **none** |
| SSH_Bruteforce | 2 SSH- Bruteforce Credential Access (TA0006) Brute Force (T1110) Password Guessing (T1110.001) SSH Patator | T1110.001 Password Guessing | T1110.001: CAPEC-49 | T1110.001: **none** |
| DoS_attacks_GoldenEye | 3 DoS- GoldenEye Impact (TA0040) Endpoint Denial of Service (T1499) Service Exhaustion Flood (T1499.002) GoldenEye | T1499.002 Service Exhaustion Flood | T1499.002: CAPEC-469, CAPEC-482, CAPEC-488, CAPEC-489, CAPEC-528 | T1499.002: yes |
| DoS_attacks_Slowloris | 4 DoS- Slowloris Impact (TA0040) Endpoint Denial of Service (T1499) Service Exhaustion Flood (T1499.002) Slowloris | T1499.002 Service Exhaustion Flood | T1499.002: CAPEC-469, CAPEC-482, CAPEC-488, CAPEC-489, CAPEC-528 | T1499.002: yes |
| DoS_attacks_SlowHTTPTest | 5 DoS- SlowHTT PTest Impact (TA0040) Endpoint Denial of Service (T1499) Service Exhaustion Flood (T1499.002) Slowhttptest | T1499.002 Service Exhaustion Flood | T1499.002: CAPEC-469, CAPEC-482, CAPEC-488, CAPEC-489, CAPEC-528 | T1499.002: yes |
| DoS_attacks_Hulk | 6 DoS-Hulk Impact (TA0040) Endpoint Denial of Service (T1499) Service Exhaustion Flood (T1499.002) Hulk | T1499.002 Service Exhaustion Flood | T1499.002: CAPEC-469, CAPEC-482, CAPEC-488, CAPEC-489, CAPEC-528 | T1499.002: yes |
| DDoS_attacks_LOIC_HTTP | 7 DDoS attacks- LOIC- HTTP Impact (TA0040) Endpoint Denial of Service (T1499) Service Exhaustion Flood (T1499.002) LOIC | T1499.002 Service Exhaustion Flood | T1499.002: CAPEC-469, CAPEC-482, CAPEC-488, CAPEC-489, CAPEC-528 | T1499.002: yes |
| DDOS_attack_LOIC_UDP | 8 DDoS- LOIC- UDP Impact (TA0040) Endpoint Denial of Service (T1499) Service Exhaustion Flood (T1499.002) LOIC | T1499.002 Service Exhaustion Flood | T1499.002: CAPEC-469, CAPEC-482, CAPEC-488, CAPEC-489, CAPEC-528 | T1499.002: yes |
| DDOS_attack_HOIC | 9 DDOS- HOIC Impact (TA0040) Endpoint Denial of Service (T1499) Service Exhaustion Flood (T1499.002) HOIC | T1499.002 Service Exhaustion Flood | T1499.002: CAPEC-469, CAPEC-482, CAPEC-488, CAPEC-489, CAPEC-528 | T1499.002: yes |
| Brute_Force_Web | 10 Brute Force - Web Credential Access (TA0006) Brute Force (T1110) Password Guessing (T1110.001) DVWA, Selenium Framework | T1110.001 Password Guessing | T1110.001: CAPEC-49 | T1110.001: **none** |
| Brute_Force_XSS | 11 Brute Force - XSS Initial Access (TA0001) Drive-by Compromise (T1189) - DVWA, Selenium Framework | T1189 Drive-by Compromise | T1189: none | T1189: **none** |
| SQL_Injection | 12 SQL Injection Initial Access (TA0001) Exploit Public- Facing Application (T1190) - DVWA, sqlmap | T1190 Exploit Public-Facing Application | T1190: none | T1190: yes |
| Infiltration | 13 Infiltration Execution (TA0002) User Execution (T1204) Malicious File (T1204.002) Dropbox, Adobe Acrobat Reader 9 Discovery (TA0007) Network Service Scanning (T1046) - Nmap, portscan | T1204.002 Malicious File, T1046 Network Service Discovery | T1204.002: none; T1046: CAPEC-300 | T1204.002: **none**; T1046: yes |
| Bot | 14 Bot Resource Developm ent (TA0042) Acquire Infrastructur e (T1583) Botnet (T1583.005) Ares, Zeus Collection (TA0009) Screen Capture (T1113) - screenshots | T1583.005 Botnet, T1113 Screen Capture | T1583.005: none; T1113: CAPEC-648 | T1583.005: **none**; T1113: **none** |

### Gaps

Techniques without any ATT&CK v20.1 analytic on a network data component (no B-DET passage): T1110.001 Password Guessing, T1113 Screen Capture, T1189 Drive-by Compromise, T1204.002 Malicious File, T1583.005 Botnet.

## 4. Defined sentences next to their evidence

Code checks that each quote exists verbatim; a reviewer checks that the sentence says no more than its evidence.

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
- [engelen-2021-doc | extended_doc.html, Flow Construction, TCP flag count] "This was due to a bug in the CICFlowMeter tool which only incremented the count of TCP flags for the first packet in a flow."
- [engelen-2021-doc | extended_doc.html, Flow Construction, TCP flag count] "As a result, this issue may affect any dataset that was generated with the CICFlowMeter-V3 tool before this fix was implemented."
- [cicflowmeter-src-2018 | FlowFeature.java, feature //37] "fw_psh_flag("Fwd PSH Flags","FwPSH""

### H1-Bwd_PSH_Flags

> Bwd PSH Flags: In CSE-CIC-IDS2018 this column is always 0: the backward PSH and URG counters are incremented only in the first-packet routine, and the first packet always defines the forward direction.

- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "if(this.src==null){ this.src = packet.getSrc();"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), backward branch] "if(packet.hasFlagPSH()){ this.bPSH_cnt++; } if(packet.hasFlagURG()){ this.bURG_cnt++; }"
- [engelen-2021-doc | extended_doc.html, Flow Construction, TCP flag count] "Similarly, the values for Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags and Bwd URG Flags were not being correctly incremented in the CICFlowMeter tool."
- [cicflowmeter-src-2018 | FlowFeature.java, feature //38] "bw_psh_flag("Bwd PSH Flags","BwPSH""

### H1-Fwd_URG_Flags

> Fwd URG Flags: In CSE-CIC-IDS2018 this column is 1 if the first packet of the flow has the URG flag set and 0 otherwise. It describes that packet only and is not a count of packets: the extractor reads flags only from the first packet of a flow.

- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "public void firstPacket(BasicPacketInfo packet){ updateFlowBulk(packet); detectUpdateSubflows(packet); checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), forward branch] "if(packet.hasFlagPSH()){ this.fPSH_cnt++; } if(packet.hasFlagURG()){ this.fURG_cnt++; }"
- [engelen-2021-doc | extended_doc.html, Flow Construction, TCP flag count] "This was due to a bug in the CICFlowMeter tool which only incremented the count of TCP flags for the first packet in a flow."
- [engelen-2021-doc | extended_doc.html, Flow Construction, TCP flag count] "As a result, this issue may affect any dataset that was generated with the CICFlowMeter-V3 tool before this fix was implemented."
- [cicflowmeter-src-2018 | FlowFeature.java, feature //39] "fw_urg_flag("Fwd URG Flags","FwURG""

### H1-Bwd_URG_Flags

> Bwd URG Flags: In CSE-CIC-IDS2018 this column is always 0: the backward PSH and URG counters are incremented only in the first-packet routine, and the first packet always defines the forward direction.

- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "if(this.src==null){ this.src = packet.getSrc();"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), backward branch] "if(packet.hasFlagPSH()){ this.bPSH_cnt++; } if(packet.hasFlagURG()){ this.bURG_cnt++; }"
- [engelen-2021-doc | extended_doc.html, Flow Construction, TCP flag count] "Similarly, the values for Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags and Bwd URG Flags were not being correctly incremented in the CICFlowMeter tool."
- [cicflowmeter-src-2018 | FlowFeature.java, feature //40] "bw_urg_flag("Bwd URG Flags","BwURG""

### H1-FIN_Flag_Cnt

> FIN Flag Cnt: In CSE-CIC-IDS2018 this column is 1 if the first packet of the flow has the RST flag set and 0 otherwise. It describes that packet only and is not a count of packets: the extractor reads flags only from the first packet of a flow. The column name does not match its content: the extractor writes the eight flag values in the iteration order of a Java HashMap, which does not follow the order of the column names, so this column holds the RST flag and not the FIN flag.

- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "public void firstPacket(BasicPacketInfo packet){ updateFlowBulk(packet); detectUpdateSubflows(packet); checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), forward branch] "if(packet.hasFlagPSH()){ this.fPSH_cnt++; } if(packet.hasFlagURG()){ this.fURG_cnt++; }"
- [engelen-2021-doc | extended_doc.html, Flow Construction, TCP flag count] "This was due to a bug in the CICFlowMeter tool which only incremented the count of TCP flags for the first packet in a flow."
- [engelen-2021-doc | extended_doc.html, Flow Construction, TCP flag count] "As a result, this issue may affect any dataset that was generated with the CICFlowMeter-V3 tool before this fix was implemented."
- [cicflowmeter-src-2018 | BasicFlow.java, field declaration] "private HashMap<String, MutableInt> flagCounts;"
- [cicflowmeter-src-2018 | BasicFlow.java, initFlags()] "flagCounts.put("FIN", new MutableInt()); flagCounts.put("SYN", new MutableInt()); flagCounts.put("RST", new MutableInt()); flagCounts.put("PSH", new MutableInt()); flagCounts.put("ACK", new MutableInt()); flagCounts.put("URG", new MutableInt()); flagCounts.put("CWR", new MutableInt()); flagCounts.put("ECE", new MutableInt());"
- [cicflowmeter-src-2018 | BasicFlow.java, CSV row writer] "for(String key: flagCounts.keySet()){ dump.append(flagCounts.get(key).value).append(separator); //50,51,52,53,54,55,56,57"
- [java8-hashmap | class description] "This class makes no guarantees as to the order of the map; in particular, it does not guarantee that the order will remain constant over time."
- [openjdk8-src | HashMap.java, hash()] "return (key == null) ? 0 : (h = key.hashCode()) ^ (h >>> 16);"
- [openjdk8-src | HashMap.java, putVal()] "if ((p = tab[i = (n - 1) & hash]) == null)"
- [openjdk8-src | String.java, hashCode()] "h = 31 * h + val[i];"
- [cicflowmeter-src-2018 | build.gradle] "sourceCompatibility = 1.8"
- [cicflowmeter-src-2018-06 | BasicFlow.java, CSV row writer] "dump.append(flagCounts.get("FIN").value).append(separator); //50 dump.append(flagCounts.get("SYN").value).append(separator); //51"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //50] "fin_cnt("FIN Flag Cnt","FINCT""

### H1-SYN_Flag_Cnt

> SYN Flag Cnt: In CSE-CIC-IDS2018 this column is 1 if the first packet of the flow has the PSH flag set and 0 otherwise. It describes that packet only and is not a count of packets: the extractor reads flags only from the first packet of a flow. The column name does not match its content: the extractor writes the eight flag values in the iteration order of a Java HashMap, which does not follow the order of the column names, so this column holds the PSH flag and not the SYN flag.

- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "public void firstPacket(BasicPacketInfo packet){ updateFlowBulk(packet); detectUpdateSubflows(packet); checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), forward branch] "if(packet.hasFlagPSH()){ this.fPSH_cnt++; } if(packet.hasFlagURG()){ this.fURG_cnt++; }"
- [engelen-2021-doc | extended_doc.html, Flow Construction, TCP flag count] "This was due to a bug in the CICFlowMeter tool which only incremented the count of TCP flags for the first packet in a flow."
- [engelen-2021-doc | extended_doc.html, Flow Construction, TCP flag count] "As a result, this issue may affect any dataset that was generated with the CICFlowMeter-V3 tool before this fix was implemented."
- [cicflowmeter-src-2018 | BasicFlow.java, field declaration] "private HashMap<String, MutableInt> flagCounts;"
- [cicflowmeter-src-2018 | BasicFlow.java, initFlags()] "flagCounts.put("FIN", new MutableInt()); flagCounts.put("SYN", new MutableInt()); flagCounts.put("RST", new MutableInt()); flagCounts.put("PSH", new MutableInt()); flagCounts.put("ACK", new MutableInt()); flagCounts.put("URG", new MutableInt()); flagCounts.put("CWR", new MutableInt()); flagCounts.put("ECE", new MutableInt());"
- [cicflowmeter-src-2018 | BasicFlow.java, CSV row writer] "for(String key: flagCounts.keySet()){ dump.append(flagCounts.get(key).value).append(separator); //50,51,52,53,54,55,56,57"
- [java8-hashmap | class description] "This class makes no guarantees as to the order of the map; in particular, it does not guarantee that the order will remain constant over time."
- [openjdk8-src | HashMap.java, hash()] "return (key == null) ? 0 : (h = key.hashCode()) ^ (h >>> 16);"
- [openjdk8-src | HashMap.java, putVal()] "if ((p = tab[i = (n - 1) & hash]) == null)"
- [openjdk8-src | String.java, hashCode()] "h = 31 * h + val[i];"
- [cicflowmeter-src-2018 | build.gradle] "sourceCompatibility = 1.8"
- [cicflowmeter-src-2018-06 | BasicFlow.java, CSV row writer] "dump.append(flagCounts.get("FIN").value).append(separator); //50 dump.append(flagCounts.get("SYN").value).append(separator); //51"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //51] "syn_cnt("SYN Flag Cnt","SYNCT""

### H1-RST_Flag_Cnt

> RST Flag Cnt: In CSE-CIC-IDS2018 this column is 1 if the first packet of the flow has the ECE flag set and 0 otherwise. It describes that packet only and is not a count of packets: the extractor reads flags only from the first packet of a flow. The column name does not match its content: the extractor writes the eight flag values in the iteration order of a Java HashMap, which does not follow the order of the column names, so this column holds the ECE flag and not the RST flag.

- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "public void firstPacket(BasicPacketInfo packet){ updateFlowBulk(packet); detectUpdateSubflows(packet); checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), forward branch] "if(packet.hasFlagPSH()){ this.fPSH_cnt++; } if(packet.hasFlagURG()){ this.fURG_cnt++; }"
- [engelen-2021-doc | extended_doc.html, Flow Construction, TCP flag count] "This was due to a bug in the CICFlowMeter tool which only incremented the count of TCP flags for the first packet in a flow."
- [engelen-2021-doc | extended_doc.html, Flow Construction, TCP flag count] "As a result, this issue may affect any dataset that was generated with the CICFlowMeter-V3 tool before this fix was implemented."
- [cicflowmeter-src-2018 | BasicFlow.java, field declaration] "private HashMap<String, MutableInt> flagCounts;"
- [cicflowmeter-src-2018 | BasicFlow.java, initFlags()] "flagCounts.put("FIN", new MutableInt()); flagCounts.put("SYN", new MutableInt()); flagCounts.put("RST", new MutableInt()); flagCounts.put("PSH", new MutableInt()); flagCounts.put("ACK", new MutableInt()); flagCounts.put("URG", new MutableInt()); flagCounts.put("CWR", new MutableInt()); flagCounts.put("ECE", new MutableInt());"
- [cicflowmeter-src-2018 | BasicFlow.java, CSV row writer] "for(String key: flagCounts.keySet()){ dump.append(flagCounts.get(key).value).append(separator); //50,51,52,53,54,55,56,57"
- [java8-hashmap | class description] "This class makes no guarantees as to the order of the map; in particular, it does not guarantee that the order will remain constant over time."
- [openjdk8-src | HashMap.java, hash()] "return (key == null) ? 0 : (h = key.hashCode()) ^ (h >>> 16);"
- [openjdk8-src | HashMap.java, putVal()] "if ((p = tab[i = (n - 1) & hash]) == null)"
- [openjdk8-src | String.java, hashCode()] "h = 31 * h + val[i];"
- [cicflowmeter-src-2018 | build.gradle] "sourceCompatibility = 1.8"
- [cicflowmeter-src-2018-06 | BasicFlow.java, CSV row writer] "dump.append(flagCounts.get("FIN").value).append(separator); //50 dump.append(flagCounts.get("SYN").value).append(separator); //51"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //52] "rst_cnt("RST Flag Cnt","RSTCT""

### H1-PSH_Flag_Cnt

> PSH Flag Cnt: In CSE-CIC-IDS2018 this column is 1 if the first packet of the flow has the SYN flag set and 0 otherwise. It describes that packet only and is not a count of packets: the extractor reads flags only from the first packet of a flow. The column name does not match its content: the extractor writes the eight flag values in the iteration order of a Java HashMap, which does not follow the order of the column names, so this column holds the SYN flag and not the PSH flag.

- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "public void firstPacket(BasicPacketInfo packet){ updateFlowBulk(packet); detectUpdateSubflows(packet); checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), forward branch] "if(packet.hasFlagPSH()){ this.fPSH_cnt++; } if(packet.hasFlagURG()){ this.fURG_cnt++; }"
- [engelen-2021-doc | extended_doc.html, Flow Construction, TCP flag count] "This was due to a bug in the CICFlowMeter tool which only incremented the count of TCP flags for the first packet in a flow."
- [engelen-2021-doc | extended_doc.html, Flow Construction, TCP flag count] "As a result, this issue may affect any dataset that was generated with the CICFlowMeter-V3 tool before this fix was implemented."
- [cicflowmeter-src-2018 | BasicFlow.java, field declaration] "private HashMap<String, MutableInt> flagCounts;"
- [cicflowmeter-src-2018 | BasicFlow.java, initFlags()] "flagCounts.put("FIN", new MutableInt()); flagCounts.put("SYN", new MutableInt()); flagCounts.put("RST", new MutableInt()); flagCounts.put("PSH", new MutableInt()); flagCounts.put("ACK", new MutableInt()); flagCounts.put("URG", new MutableInt()); flagCounts.put("CWR", new MutableInt()); flagCounts.put("ECE", new MutableInt());"
- [cicflowmeter-src-2018 | BasicFlow.java, CSV row writer] "for(String key: flagCounts.keySet()){ dump.append(flagCounts.get(key).value).append(separator); //50,51,52,53,54,55,56,57"
- [java8-hashmap | class description] "This class makes no guarantees as to the order of the map; in particular, it does not guarantee that the order will remain constant over time."
- [openjdk8-src | HashMap.java, hash()] "return (key == null) ? 0 : (h = key.hashCode()) ^ (h >>> 16);"
- [openjdk8-src | HashMap.java, putVal()] "if ((p = tab[i = (n - 1) & hash]) == null)"
- [openjdk8-src | String.java, hashCode()] "h = 31 * h + val[i];"
- [cicflowmeter-src-2018 | build.gradle] "sourceCompatibility = 1.8"
- [cicflowmeter-src-2018-06 | BasicFlow.java, CSV row writer] "dump.append(flagCounts.get("FIN").value).append(separator); //50 dump.append(flagCounts.get("SYN").value).append(separator); //51"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //53] "pst_cnt("PSH Flag Cnt","PSHCT""

### H1-ACK_Flag_Cnt

> ACK Flag Cnt: In CSE-CIC-IDS2018 this column is 1 if the first packet of the flow has the ACK flag set and 0 otherwise. It describes that packet only and is not a count of packets: the extractor reads flags only from the first packet of a flow.

- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "public void firstPacket(BasicPacketInfo packet){ updateFlowBulk(packet); detectUpdateSubflows(packet); checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), forward branch] "if(packet.hasFlagPSH()){ this.fPSH_cnt++; } if(packet.hasFlagURG()){ this.fURG_cnt++; }"
- [engelen-2021-doc | extended_doc.html, Flow Construction, TCP flag count] "This was due to a bug in the CICFlowMeter tool which only incremented the count of TCP flags for the first packet in a flow."
- [engelen-2021-doc | extended_doc.html, Flow Construction, TCP flag count] "As a result, this issue may affect any dataset that was generated with the CICFlowMeter-V3 tool before this fix was implemented."
- [cicflowmeter-src-2018 | FlowFeature.java, feature //54] "ack_cnt("ACK Flag Cnt","ACKCT""

### H1-URG_Flag_Cnt

> URG Flag Cnt: In CSE-CIC-IDS2018 this column is 1 if the first packet of the flow has the FIN flag set and 0 otherwise. It describes that packet only and is not a count of packets: the extractor reads flags only from the first packet of a flow. The column name does not match its content: the extractor writes the eight flag values in the iteration order of a Java HashMap, which does not follow the order of the column names, so this column holds the FIN flag and not the URG flag.

- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "public void firstPacket(BasicPacketInfo packet){ updateFlowBulk(packet); detectUpdateSubflows(packet); checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), forward branch] "if(packet.hasFlagPSH()){ this.fPSH_cnt++; } if(packet.hasFlagURG()){ this.fURG_cnt++; }"
- [engelen-2021-doc | extended_doc.html, Flow Construction, TCP flag count] "This was due to a bug in the CICFlowMeter tool which only incremented the count of TCP flags for the first packet in a flow."
- [engelen-2021-doc | extended_doc.html, Flow Construction, TCP flag count] "As a result, this issue may affect any dataset that was generated with the CICFlowMeter-V3 tool before this fix was implemented."
- [cicflowmeter-src-2018 | BasicFlow.java, field declaration] "private HashMap<String, MutableInt> flagCounts;"
- [cicflowmeter-src-2018 | BasicFlow.java, initFlags()] "flagCounts.put("FIN", new MutableInt()); flagCounts.put("SYN", new MutableInt()); flagCounts.put("RST", new MutableInt()); flagCounts.put("PSH", new MutableInt()); flagCounts.put("ACK", new MutableInt()); flagCounts.put("URG", new MutableInt()); flagCounts.put("CWR", new MutableInt()); flagCounts.put("ECE", new MutableInt());"
- [cicflowmeter-src-2018 | BasicFlow.java, CSV row writer] "for(String key: flagCounts.keySet()){ dump.append(flagCounts.get(key).value).append(separator); //50,51,52,53,54,55,56,57"
- [java8-hashmap | class description] "This class makes no guarantees as to the order of the map; in particular, it does not guarantee that the order will remain constant over time."
- [openjdk8-src | HashMap.java, hash()] "return (key == null) ? 0 : (h = key.hashCode()) ^ (h >>> 16);"
- [openjdk8-src | HashMap.java, putVal()] "if ((p = tab[i = (n - 1) & hash]) == null)"
- [openjdk8-src | String.java, hashCode()] "h = 31 * h + val[i];"
- [cicflowmeter-src-2018 | build.gradle] "sourceCompatibility = 1.8"
- [cicflowmeter-src-2018-06 | BasicFlow.java, CSV row writer] "dump.append(flagCounts.get("FIN").value).append(separator); //50 dump.append(flagCounts.get("SYN").value).append(separator); //51"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //55] "urg_cnt("URG Flag Cnt","URGCT""

### H1-CWE_Flag_Count

> CWE Flag Count: In CSE-CIC-IDS2018 this column is 1 if the first packet of the flow has the URG flag set and 0 otherwise. It describes that packet only and is not a count of packets: the extractor reads flags only from the first packet of a flow. The column name does not match its content: the extractor writes the eight flag values in the iteration order of a Java HashMap, which does not follow the order of the column names, so this column holds the URG flag and not the CWR flag.

- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "public void firstPacket(BasicPacketInfo packet){ updateFlowBulk(packet); detectUpdateSubflows(packet); checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), forward branch] "if(packet.hasFlagPSH()){ this.fPSH_cnt++; } if(packet.hasFlagURG()){ this.fURG_cnt++; }"
- [engelen-2021-doc | extended_doc.html, Flow Construction, TCP flag count] "This was due to a bug in the CICFlowMeter tool which only incremented the count of TCP flags for the first packet in a flow."
- [engelen-2021-doc | extended_doc.html, Flow Construction, TCP flag count] "As a result, this issue may affect any dataset that was generated with the CICFlowMeter-V3 tool before this fix was implemented."
- [cicflowmeter-src-2018 | BasicFlow.java, field declaration] "private HashMap<String, MutableInt> flagCounts;"
- [cicflowmeter-src-2018 | BasicFlow.java, initFlags()] "flagCounts.put("FIN", new MutableInt()); flagCounts.put("SYN", new MutableInt()); flagCounts.put("RST", new MutableInt()); flagCounts.put("PSH", new MutableInt()); flagCounts.put("ACK", new MutableInt()); flagCounts.put("URG", new MutableInt()); flagCounts.put("CWR", new MutableInt()); flagCounts.put("ECE", new MutableInt());"
- [cicflowmeter-src-2018 | BasicFlow.java, CSV row writer] "for(String key: flagCounts.keySet()){ dump.append(flagCounts.get(key).value).append(separator); //50,51,52,53,54,55,56,57"
- [java8-hashmap | class description] "This class makes no guarantees as to the order of the map; in particular, it does not guarantee that the order will remain constant over time."
- [openjdk8-src | HashMap.java, hash()] "return (key == null) ? 0 : (h = key.hashCode()) ^ (h >>> 16);"
- [openjdk8-src | HashMap.java, putVal()] "if ((p = tab[i = (n - 1) & hash]) == null)"
- [openjdk8-src | String.java, hashCode()] "h = 31 * h + val[i];"
- [cicflowmeter-src-2018 | build.gradle] "sourceCompatibility = 1.8"
- [cicflowmeter-src-2018-06 | BasicFlow.java, CSV row writer] "dump.append(flagCounts.get("FIN").value).append(separator); //50 dump.append(flagCounts.get("SYN").value).append(separator); //51"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //56] "cwe_cnt("CWE Flag Count","CWECT""

### H1-ECE_Flag_Cnt

> ECE Flag Cnt: In CSE-CIC-IDS2018 this column is 1 if the first packet of the flow has the CWR flag set and 0 otherwise. It describes that packet only and is not a count of packets: the extractor reads flags only from the first packet of a flow. The column name does not match its content: the extractor writes the eight flag values in the iteration order of a Java HashMap, which does not follow the order of the column names, so this column holds the CWR flag and not the ECE flag.

- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "public void firstPacket(BasicPacketInfo packet){ updateFlowBulk(packet); detectUpdateSubflows(packet); checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), forward branch] "if(packet.hasFlagPSH()){ this.fPSH_cnt++; } if(packet.hasFlagURG()){ this.fURG_cnt++; }"
- [engelen-2021-doc | extended_doc.html, Flow Construction, TCP flag count] "This was due to a bug in the CICFlowMeter tool which only incremented the count of TCP flags for the first packet in a flow."
- [engelen-2021-doc | extended_doc.html, Flow Construction, TCP flag count] "As a result, this issue may affect any dataset that was generated with the CICFlowMeter-V3 tool before this fix was implemented."
- [cicflowmeter-src-2018 | BasicFlow.java, field declaration] "private HashMap<String, MutableInt> flagCounts;"
- [cicflowmeter-src-2018 | BasicFlow.java, initFlags()] "flagCounts.put("FIN", new MutableInt()); flagCounts.put("SYN", new MutableInt()); flagCounts.put("RST", new MutableInt()); flagCounts.put("PSH", new MutableInt()); flagCounts.put("ACK", new MutableInt()); flagCounts.put("URG", new MutableInt()); flagCounts.put("CWR", new MutableInt()); flagCounts.put("ECE", new MutableInt());"
- [cicflowmeter-src-2018 | BasicFlow.java, CSV row writer] "for(String key: flagCounts.keySet()){ dump.append(flagCounts.get(key).value).append(separator); //50,51,52,53,54,55,56,57"
- [java8-hashmap | class description] "This class makes no guarantees as to the order of the map; in particular, it does not guarantee that the order will remain constant over time."
- [openjdk8-src | HashMap.java, hash()] "return (key == null) ? 0 : (h = key.hashCode()) ^ (h >>> 16);"
- [openjdk8-src | HashMap.java, putVal()] "if ((p = tab[i = (n - 1) & hash]) == null)"
- [openjdk8-src | String.java, hashCode()] "h = 31 * h + val[i];"
- [cicflowmeter-src-2018 | build.gradle] "sourceCompatibility = 1.8"
- [cicflowmeter-src-2018-06 | BasicFlow.java, CSV row writer] "dump.append(flagCounts.get("FIN").value).append(separator); //50 dump.append(flagCounts.get("SYN").value).append(separator); //51"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //57] "ece_cnt("ECE Flag Cnt","ECECT""

### H1-Fwd_Byts_per_b_Avg

> Fwd Byts/b Avg: In CSE-CIC-IDS2018 this column is always 0: bulk statistics are updated only for the first packet of a flow (the update is disabled for later packets), so no bulk transfer is ever recorded.

- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //63] "fw_byt_blk_avg("Fwd Byts/b Avg","FwB/BAG""

### H1-Fwd_Pkts_per_b_Avg

> Fwd Pkts/b Avg: In CSE-CIC-IDS2018 this column is always 0: bulk statistics are updated only for the first packet of a flow (the update is disabled for later packets), so no bulk transfer is ever recorded.

- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //64] "fw_pkt_blk_avg("Fwd Pkts/b Avg","FwP/BAG""

### H1-Fwd_Blk_Rate_Avg

> Fwd Blk Rate Avg: In CSE-CIC-IDS2018 this column is always 0: bulk statistics are updated only for the first packet of a flow (the update is disabled for later packets), so no bulk transfer is ever recorded.

- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //65] "fw_blk_rate_avg("Fwd Blk Rate Avg","FwBRAG""

### H1-Bwd_Byts_per_b_Avg

> Bwd Byts/b Avg: In CSE-CIC-IDS2018 this column is always 0: bulk statistics are updated only for the first packet of a flow (the update is disabled for later packets), so no bulk transfer is ever recorded.

- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //66] "bw_byt_blk_avg("Bwd Byts/b Avg","BwB/BAG""

### H1-Bwd_Pkts_per_b_Avg

> Bwd Pkts/b Avg: In CSE-CIC-IDS2018 this column is always 0: bulk statistics are updated only for the first packet of a flow (the update is disabled for later packets), so no bulk transfer is ever recorded.

- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //67] "bw_pkt_blk_avg("Bwd Pkts/b Avg","BwP/BAG""

### H1-Bwd_Blk_Rate_Avg

> Bwd Blk Rate Avg: In CSE-CIC-IDS2018 this column is always 0: bulk statistics are updated only for the first packet of a flow (the update is disabled for later packets), so no bulk transfer is ever recorded.

- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //68] "bw_blk_rate_avg("Bwd Blk Rate Avg","BwBRAG""

### H1-Init_Fwd_Win_Byts

> Init Fwd Win Byts: In CSE-CIC-IDS2018 this column is the raw TCP window field of a single packet header, not a byte count: for the forward direction it comes from the first packet of the flow, and for the backward direction from the most recent backward packet (the value is overwritten on each one). The field is recorded as read, without the window scale factor, so unless that packet is a SYN or SYN-ACK segment the real window can be larger than the recorded value. -1 is the extractor's default, kept when no TCP window was read, for example in a UDP flow or when there is no backward packet.

- [cicflowmeter-src-2018 | BasicPacketInfo.java, field declaration] "private int TCPWindow=-1;"
- [cicflowmeter-src-2018 | BasicFlow.java, field declarations] "private int Init_Win_bytes_forward=-1; private int Init_Win_bytes_backward=-1;"
- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setTCPWindow(tcp.window());"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), forward branch] "this.min_seg_size_forward = packet.getHeaderBytes(); Init_Win_bytes_forward = packet.getTCPWindow();"
- [cicflowmeter-src-2018 | BasicFlow.java, addPacket(), backward branch] "this.bwdPktStats.addValue((double)packet.getPayloadBytes()); Init_Win_bytes_backward = packet.getTCPWindow();"
- [rfc7323 | Section 2.2] "The window field in a segment where the SYN bit is set (i.e., a <SYN> or <SYN,ACK>) MUST NOT be scaled."
- [rfc9293 | Section 3.1, Header Format] "The value is shifted when the window scaling extension is used [47]."
- [cicflowmeter-src-2018 | FlowFeature.java, feature //73] "fw_win_byt("Init Fwd Win Byts","FwWB""

### H1-Init_Bwd_Win_Byts

> Init Bwd Win Byts: In CSE-CIC-IDS2018 this column is the raw TCP window field of a single packet header, not a byte count: for the forward direction it comes from the first packet of the flow, and for the backward direction from the most recent backward packet (the value is overwritten on each one). The field is recorded as read, without the window scale factor, so unless that packet is a SYN or SYN-ACK segment the real window can be larger than the recorded value. -1 is the extractor's default, kept when no TCP window was read, for example in a UDP flow or when there is no backward packet.

- [cicflowmeter-src-2018 | BasicPacketInfo.java, field declaration] "private int TCPWindow=-1;"
- [cicflowmeter-src-2018 | BasicFlow.java, field declarations] "private int Init_Win_bytes_forward=-1; private int Init_Win_bytes_backward=-1;"
- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setTCPWindow(tcp.window());"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), forward branch] "this.min_seg_size_forward = packet.getHeaderBytes(); Init_Win_bytes_forward = packet.getTCPWindow();"
- [cicflowmeter-src-2018 | BasicFlow.java, addPacket(), backward branch] "this.bwdPktStats.addValue((double)packet.getPayloadBytes()); Init_Win_bytes_backward = packet.getTCPWindow();"
- [rfc7323 | Section 2.2] "The window field in a segment where the SYN bit is set (i.e., a <SYN> or <SYN,ACK>) MUST NOT be scaled."
- [rfc9293 | Section 3.1, Header Format] "The value is shifted when the window scaling extension is used [47]."
- [cicflowmeter-src-2018 | FlowFeature.java, feature //74] "bw_win_byt("Init Bwd Win Byts","BwWB""

### H1-Fwd_Act_Data_Pkts

> Fwd Act Data Pkts: In CSE-CIC-IDS2018 this column counts the forward packets after the first one that carry at least 1 byte of payload; the first packet of the flow is never counted, so a flow whose only forward payload is in its first packet shows 0.

- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "if(packet.getPayloadBytes() >=1){ this.Act_data_pkt_forward++; }"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //75] "Fw_act_pkt("Fwd Act Data Pkts","FwAP""

### H1-Fwd_Seg_Size_Min

> Fwd Seg Size Min: In CSE-CIC-IDS2018 this column is the smallest transport-header length, in bytes, among the forward packets of the flow; it is not a payload or segment size.

- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "this.min_seg_size_forward = Math.min(packet.getHeaderBytes(),this.min_seg_size_forward);"
- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setHeaderBytes(tcp.getHeaderLength());"
- [cicflowmeter-src-2018 | FlowFeature.java, feature //76] "fw_seg_min("Fwd Seg Size Min","FwSgMI""

### H1-ae_recon_error

> Reconstruction error of the autoencoder: the mean squared error between the flow's scaled features and the autoencoder's reconstruction of them.

- [layer1-code | autoencoder.py, reconstruction_error()] "return ((x - x_hat) ** 2).mean(dim=1)"
- [layer1-code | train_protocol_b.py, AE features] "names = feats + [f"ae_latent_{i}" for i in range(AE_BOTTLENECK)] + ["ae_recon_error"]"

### H1-ae_latent

> One of the 16 coordinates of the autoencoder's latent (bottleneck) layer, the layer that encodes the flow's scaled features.

- [layer1-code | autoencoder.py, module docstring] "Encoder: input_dim -> 64 -> 32 -> bottleneck_dim"
- [layer1-code | train_protocol_b.py, configuration] "AE_BOTTLENECK = 16"
- [layer1-code | train_protocol_b.py, AE features] "names = feats + [f"ae_latent_{i}" for i in range(AE_BOTTLENECK)] + ["ae_recon_error"]"

### H3-LABELNOISE

> The reference label of this alert comes from the labels published with CSE-CIC-IDS2018, and an independent re-analysis of the original packet captures found labelling errors in that release: on one day a large share of the flows labelled as web brute force are mislabelled, including a block of UDP flows to port 500 rather than web traffic; every DDoS-LOIC-UDP flow of that day carries the label DDoS-LOIC-HTTP; and the Infiltration flows between the two hosts named on the dataset page are labelled Benign. Where the flow evidence contradicts the background of the reference label, report the contradiction instead of resolving it in favour of either side.

- [distrinet-2018 | CSECICIDS2018.html, 23-02-2018 (Brute Force -Web)] "On this day, the attack has significant label contamination."
- [distrinet-2018 | CSECICIDS2018.html, 23-02-2018 (Brute Force -Web)] "Besides these, there are 151 additional flows that are also mislabelled wrong (i.e 41.7% mislabeled)."
- [distrinet-2018 | CSECICIDS2018.html, 23-02-2018 (Brute Force -Web)] "Of these, 128 flows have protocol = 17 (i.e. not tcp) with port = 500."
- [distrinet-2018 | CSECICIDS2018.html, DDoS LOIC UDP] "all DDoS-LOIC-UDP attacks are incorrectly labelled as DDoS-LOIC-HTTP"
- [distrinet-2018 | CSECICIDS2018.html, Infiltration] "in the originally released dataset, all flows going between the 2 IP's described above are labelled as Benign"

### H3-NOPAYLOAD

> This flow carries no transport-layer payload in either direction, so application content (for example an HTTP request or an SQL statement) cannot be observed in it. The extractor ends a TCP flow at the first FIN packet, and a packet that arrives when no flow is open starts a new flow, so the last packets of a connection can form a flow of their own.

- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setPayloadBytes(tcp.getPayloadLength());"
- [cicflowmeter-src-2018 | FlowGenerator.java, addPacket()] "// Flow finished due FIN flag (tcp only):"
- [cicflowmeter-src-2018 | FlowGenerator.java, addPacket()] "}else if(packet.hasFlagFIN()){"
- [cicflowmeter-src-2018 | FlowGenerator.java, addPacket(), packet without an open flow] "currentFlows.put(packet.getFlowId(), new BasicFlow(bidirectional,packet));"

### H3-NOWIN

> A value of -1 in an initial-window column is not a window size: the TCP window field is a 16-bit count of octets. It is the extractor's default, kept when no TCP window was read for that direction, for example in a UDP flow or in a flow without backward packets.

- [rfc9293 | Section 3.1, Header Format] "Window: 16 bits The number of data octets beginning with the one indicated in the acknowledgment field that the sender of this segment is willing to accept. The value is shifted when the window scaling extension is used [47]."
- [cicflowmeter-src-2018 | BasicPacketInfo.java, field declaration] "private int TCPWindow=-1;"
- [cicflowmeter-src-2018 | BasicFlow.java, field declarations] "private int Init_Win_bytes_forward=-1; private int Init_Win_bytes_backward=-1;"
- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setTCPWindow(tcp.window());"

### H3-WINSCALE

> The first packet of this TCP flow is not a SYN segment, so the initial-window columns hold window fields of segments sent after the connection was opened. Such a field is not in bytes when window scaling is in use: the scale factor is agreed only in the SYN segments and every later window field is shifted by it, so the real window can be larger than the recorded value. Only a window field carried in a SYN or SYN-ACK segment is never scaled.

- [rfc7323 | Section 2.1] "This option is sent only in a <SYN> segment (a segment with the SYN bit on), hence the window scale is fixed in each direction when a connection is opened."
- [rfc7323 | Section 2.2] "The window field in a segment where the SYN bit is set (i.e., a <SYN> or <SYN,ACK>) MUST NOT be scaled."
- [rfc9293 | Section 3.1, Header Format] "The value is shifted when the window scaling extension is used [47]."
- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setTCPWindow(tcp.window());"
- [cicflowmeter-src-2018 | BasicFlow.java, CSV row writer] "for(String key: flagCounts.keySet()){ dump.append(flagCounts.get(key).value).append(separator); //50,51,52,53,54,55,56,57"

### H3-TIMEOUT

> The flow lasted until the extractor's default 120 s flow timeout. A longer connection is split into several consecutive flows, so this duration does not cover the whole session.

- [cicflowmeter-src-2018 | ReadPcapFileWorker.java, constructor] "flowTimeout = 120000000L;"
- [cicflowmeter-src-2018 | TrafficFlowWorker.java, doInBackground()] "FlowGenerator flowGen = new FlowGenerator(true,120000000L, 5000000L);"
- [cicflowmeter-src-2018 | FlowGenerator.java, addPacket()] "// Flow finished due flowtimeout: // 1.- we move the flow to finished flow list // 2.- we eliminate the flow from the current flow list // 3.- we create a new flow with the packet-in-process"
- [distrinet-2018 | CSECICIDS2018.html, DoS Slowloris] "With the timeout value of the CICFlowMeter being set to 120 seconds when processing this dataset, a single DoS Slowloris TCP connection will be split over multiple flows."

### H3-UDP

> This is a UDP flow. The extractor reads flags and the window only from TCP headers, so for UDP the flag columns stay 0 and the initial-window columns stay -1; they carry no information about this flow.

- [cicflowmeter-readme | ReadMe.txt, list of extracted features] "Fwd PSH flags Number of times the PSH flag was set in packets travelling in the forward direction (0 for UDP)"
- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setFlagRST(tcp.flags_RST());"
- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setTCPWindow(tcp.window());"
- [cicflowmeter-src-2018 | BasicPacketInfo.java, field declaration] "private int TCPWindow=-1;"

### H3-NOTRANSPORT

> Protocol 0 here means the packets carried neither a TCP nor a UDP header: the extractor sets the protocol only to 6 (TCP) or 17 (UDP) and otherwise keeps the Java default value 0. It does not refer to IANA protocol number 0 (HOPOPT).

- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setProtocol(6);"
- [cicflowmeter-src-2018 | PacketReader.java, UDP branch] "packetInfo.setProtocol(17);"
- [cicflowmeter-src-2018 | BasicPacketInfo.java, field declarations] "private int protocol;"
- [java8-jls | Section 4.12.5, Initial Values of Variables] "For type int, the default value is zero, that is, 0."
- [iana-protocols | protocol-numbers-1.csv, row 0] "0,HOPOPT,IPv6 Hop-by-Hop Option"

### Notes attached to H1 passages

**XN-TIME-MICROSECONDS** (on: Flow Duration, Flow IAT Mean, Flow IAT Std, Flow IAT Max, Flow IAT Min, Fwd IAT Tot, Fwd IAT Mean, Fwd IAT Std, Fwd IAT Max, Fwd IAT Min, Bwd IAT Tot, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Bwd IAT Min, Active Mean, Active Std, Active Max, Active Min, Idle Mean, Idle Std, Idle Max, Idle Min)

> Times are in microseconds: the extractor reads packet timestamps in microseconds.

- [cicflowmeter-src-2018 | PacketReader.java, IPv4 packet parsing] "packetInfo.setTimeStamp(packet.getCaptureHeader().timestampInMicros());"
- [cicflowmeter-readme | ReadMe.txt, list of extracted features] "Flow duration Duration of the flow in Microsecond"

**XN-FLOW-TIMEOUT** (on: Flow Duration)

> The extractor's default flow timeout is 120 s (120,000,000 microseconds): a flow is closed when a packet arrives more than that after the flow's first packet, and that packet starts a new flow.

- [cicflowmeter-src-2018 | FlowOfflinePane.java, initSettingPane()] "param1Ele.add(120000000L);"
- [cicflowmeter-src-2018 | ReadPcapFileWorker.java, constructor] "flowTimeout = 120000000L;"
- [cicflowmeter-src-2018 | FlowGenerator.java, addPacket()] "if((currentTimestamp -flow.getFlowStartTime())>flowTimeOut){"
- [distrinet-2018 | CSECICIDS2018.html, DoS Slowloris] "With the timeout value of the CICFlowMeter being set to 120 seconds when processing this dataset, a single DoS Slowloris TCP connection will be split over multiple flows."

**XN-PAYLOAD-BYTES** (on: TotLen Fwd Pkts, TotLen Bwd Pkts, Fwd Pkt Len Max, Fwd Pkt Len Min, Fwd Pkt Len Mean, Fwd Pkt Len Std, Bwd Pkt Len Max, Bwd Pkt Len Min, Bwd Pkt Len Mean, Bwd Pkt Len Std, Pkt Len Min, Pkt Len Max, Pkt Len Mean, Pkt Len Std, Pkt Len Var, Pkt Size Avg, Fwd Seg Size Avg, Bwd Seg Size Avg, Subflow Fwd Byts, Subflow Bwd Byts)

> Lengths count transport-layer payload bytes (the TCP or UDP payload), not packet headers.

- [cicflowmeter-src-2018 | PacketReader.java, TCP branch] "packetInfo.setPayloadBytes(tcp.getPayloadLength());"
- [cicflowmeter-src-2018 | PacketReader.java, UDP branch] "packetInfo.setPayloadBytes(udp.getPayloadLength());"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "this.fwdPktStats.addValue((double)packet.getPayloadBytes());"

**XN-FIRST-PACKET-TWICE** (on: Pkt Len Mean, Pkt Len Std, Pkt Len Var, Pkt Size Avg)

> The extractor adds the payload length of the flow's first packet twice to the statistics over all packets, so the first packet counts double: the mean, standard deviation and variance are taken over one more value than the flow has packets, and the average packet size divides that sum by the real packet count.

- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket()] "this.endActiveTime = packet.getTimeStamp(); this.flowLengthStats.addValue((double)packet.getPayloadBytes());"
- [cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), forward branch] "Init_Win_bytes_forward = packet.getTCPWindow(); this.flowLengthStats.addValue((double)packet.getPayloadBytes());"
- [cicflowmeter-src-2018 | BasicFlow.java, getAvgPacketSize()] "return (this.flowLengthStats.getSum()/this.packetCount());"

**XN-CWE-NAME** (on: CWE Flag Count)

> 'CWE' in this column name is the 2018 extractor's name for its CWR counter; later versions of the extractor call the column 'CWR Flag Count'.

- [cicflowmeter-src-2018 | FlowFeature.java, feature //56] "cwe_cnt("CWE Flag Count","CWECT""
- [cicflowmeter-readme | FlowFeature.java (master), feature //56] "CWR_cnt("CWR Flag Count","CWRCT""

**XN-DOWNUP-INTEGER** (on: Down/Up Ratio)

> The extractor divides the backward packet count by the forward packet count with integer division, so the fraction is dropped: the value is 0 whenever the flow has fewer backward than forward packets, and it never has a fractional part.

- [cicflowmeter-src-2018 | BasicFlow.java, getDownUpRatio()] "return (double)(this.backward.size()/this.forward.size());"

**XN-SUBFLOW-ONE** (on: Subflow Fwd Pkts, Subflow Fwd Byts, Subflow Bwd Pkts, Subflow Bwd Byts)

> Subflow detection runs only for the first packet of a flow, so every flow counts as a single subflow and these columns repeat the flow totals.

- [cicflowmeter-src-2018 | BasicFlow.java, addPacket()] "public void addPacket(BasicPacketInfo packet){ //updateFlowBulk(packet); //detectUpdateSubflows(packet); //checkFlags(packet);"
- [cicflowmeter-src-2018 | BasicFlow.java, getSflow_fpackets()] "return this.forward.size()/sfCount;"

## 5. Flag columns (CICFlowMeter-V3 2018, Java HashMap order)

| CSV column | flag it actually holds (first packet only) |
|---|---|
| FIN Flag Cnt | RST |
| SYN Flag Cnt | PSH |
| RST Flag Cnt | ECE |
| PSH Flag Cnt | SYN |
| ACK Flag Cnt | ACK |
| URG Flag Cnt | FIN |
| CWE Flag Count | URG |
| ECE Flag Cnt | CWR |
| Fwd PSH Flags | PSH |
| Fwd URG Flags | URG |

## 6. Short quotations (fewer than five words) for manual review

Each is verbatim in its locked source (and in the file named by its locator).

| source | locator | quote |
|---|---|---|
| cicflowmeter-readme | FlowFeature.java, feature //56 | `CWR_cnt("CWR Flag Count","CWRCT"` |
| cicflowmeter-readme | FlowFeature.java, feature //75 | `Fw_act_pkt("Fwd Act Data Pkts","FwAP"` |
| cicflowmeter-readme | FlowFeature.java, feature //54 | `ack_cnt("ACK Flag Count","ACKCT"` |
| cicflowmeter-readme | FlowFeature.java, feature //77 | `atv_avg("Active Mean","AcAG"` |
| cicflowmeter-readme | FlowFeature.java, feature //79 | `atv_max("Active Max","AcMA"` |
| cicflowmeter-readme | FlowFeature.java, feature //80 | `atv_min("Active Min","AcMI"` |
| cicflowmeter-readme | FlowFeature.java, feature //78 | `atv_std("Active Std","AcSD"` |
| cicflowmeter-readme | FlowFeature.java, feature //68 | `bw_blk_rate_avg("Bwd Bulk Rate Avg","BwBRAG"` |
| cicflowmeter-readme | FlowFeature.java, feature //66 | `bw_byt_blk_avg("Bwd Bytes/Bulk Avg","BwB/BAG"` |
| cicflowmeter-readme | FlowFeature.java, feature //42 | `bw_hdr_len("Bwd Header Length","BwHL"` |
| cicflowmeter-readme | FlowFeature.java, feature //33 | `bw_iat_avg("Bwd IAT Mean","BwIATAG"` |
| cicflowmeter-readme | FlowFeature.java, feature //35 | `bw_iat_max("Bwd IAT Max","BwIATMA"` |
| cicflowmeter-readme | FlowFeature.java, feature //36 | `bw_iat_min("Bwd IAT Min","BwIATMI"` |
| cicflowmeter-readme | FlowFeature.java, feature //34 | `bw_iat_std("Bwd IAT Std","BwIATSD"` |
| cicflowmeter-readme | FlowFeature.java, feature //32 | `bw_iat_tot("Bwd IAT Total","BwIATTO"` |
| cicflowmeter-readme | FlowFeature.java, feature //67 | `bw_pkt_blk_avg("Bwd Packet/Bulk Avg","BwP/BAG"` |
| cicflowmeter-readme | FlowFeature.java, feature //19 | `bw_pkt_l_avg("Bwd Packet Length Mean","BwPLAG"` |
| cicflowmeter-readme | FlowFeature.java, feature //17 | `bw_pkt_l_max("Bwd Packet Length Max","BwPLMA"` |
| cicflowmeter-readme | FlowFeature.java, feature //18 | `bw_pkt_l_min("Bwd Packet Length Min","BwPLMI"` |
| cicflowmeter-readme | FlowFeature.java, feature //20 | `bw_pkt_l_std("Bwd Packet Length Std","BwPLSD"` |
| cicflowmeter-readme | FlowFeature.java, feature //44 | `bw_pkt_s("Bwd Packets/s","Bwp/s"` |
| cicflowmeter-readme | FlowFeature.java, feature //38 | `bw_psh_flag("Bwd PSH Flags","BwPSH"` |
| cicflowmeter-readme | FlowFeature.java, feature //61 | `bw_seg_avg("Bwd Segment Size Avg","BwSgAG"` |
| cicflowmeter-readme | FlowFeature.java, feature //40 | `bw_urg_flag("Bwd URG Flags","BwURG"` |
| cicflowmeter-readme | FlowFeature.java, feature //74 | `bw_win_byt("Bwd Init Win Bytes","BwWB"` |
| cicflowmeter-readme | FlowFeature.java, feature //58 | `down_up_ratio("Down/Up Ratio","D/URO"` |
| cicflowmeter-readme | FlowFeature.java, feature //5 | `dst_pot("Dst Port","DPT"` |
| cicflowmeter-readme | FlowFeature.java, feature //57 | `ece_cnt("ECE Flag Count","ECECT"` |
| cicflowmeter-readme | FlowFeature.java, feature //50 | `fin_cnt("FIN Flag Count","FINCT"` |
| cicflowmeter-readme | FlowFeature.java, feature //21 | `fl_byt_s("Flow Bytes/s","FB/s"` |
| cicflowmeter-readme | FlowFeature.java, feature //8 | `fl_dur("Flow Duration","DUR"` |
| cicflowmeter-readme | FlowFeature.java, feature //23 | `fl_iat_avg("Flow IAT Mean","FLIATAG"` |
| cicflowmeter-readme | FlowFeature.java, feature //25 | `fl_iat_max("Flow IAT Max","FLIATMA"` |
| cicflowmeter-readme | FlowFeature.java, feature //26 | `fl_iat_min("Flow IAT Min","FLIATMI"` |
| cicflowmeter-readme | FlowFeature.java, feature //24 | `fl_iat_std("Flow IAT Std","FLIATSD"` |
| cicflowmeter-readme | FlowFeature.java, feature //22 | `fl_pkt_s("Flow Packets/s","FP/s"` |
| cicflowmeter-readme | FlowFeature.java, feature //65 | `fw_blk_rate_avg("Fwd Bulk Rate Avg","FwBRAG"` |
| cicflowmeter-readme | FlowFeature.java, feature //63 | `fw_byt_blk_avg("Fwd Bytes/Bulk Avg","FwB/BAG"` |
| cicflowmeter-readme | FlowFeature.java, feature //41 | `fw_hdr_len("Fwd Header Length","FwHL"` |
| cicflowmeter-readme | FlowFeature.java, feature //28 | `fw_iat_avg("Fwd IAT Mean","FwIATAG"` |
| cicflowmeter-readme | FlowFeature.java, feature //30 | `fw_iat_max("Fwd IAT Max","FwIATMA"` |
| cicflowmeter-readme | FlowFeature.java, feature //31 | `fw_iat_min("Fwd IAT Min","FwIATMI"` |
| cicflowmeter-readme | FlowFeature.java, feature //29 | `fw_iat_std("Fwd IAT Std","FwIATSD"` |
| cicflowmeter-readme | FlowFeature.java, feature //27 | `fw_iat_tot("Fwd IAT Total","FwIATTO"` |
| cicflowmeter-readme | FlowFeature.java, feature //64 | `fw_pkt_blk_avg("Fwd Packet/Bulk Avg","FwP/BAG"` |
| cicflowmeter-readme | FlowFeature.java, feature //15 | `fw_pkt_l_avg("Fwd Packet Length Mean","FwPLAG"` |
| cicflowmeter-readme | FlowFeature.java, feature //13 | `fw_pkt_l_max("Fwd Packet Length Max","FwPLMA"` |
| cicflowmeter-readme | FlowFeature.java, feature //14 | `fw_pkt_l_min("Fwd Packet Length Min","FwPLMI"` |
| cicflowmeter-readme | FlowFeature.java, feature //16 | `fw_pkt_l_std("Fwd Packet Length Std","FwPLSD"` |
| cicflowmeter-readme | FlowFeature.java, feature //43 | `fw_pkt_s("Fwd Packets/s","FwP/s"` |
| cicflowmeter-readme | FlowFeature.java, feature //37 | `fw_psh_flag("Fwd PSH Flags","FwPSH"` |
| cicflowmeter-readme | FlowFeature.java, feature //60 | `fw_seg_avg("Fwd Segment Size Avg","FwSgAG"` |
| cicflowmeter-readme | FlowFeature.java, feature //76 | `fw_seg_min("Fwd Seg Size Min","FwSgMI"` |
| cicflowmeter-readme | FlowFeature.java, feature //39 | `fw_urg_flag("Fwd URG Flags","FwURG"` |
| cicflowmeter-readme | FlowFeature.java, feature //73 | `fw_win_byt("FWD Init Win Bytes","FwWB"` |
| cicflowmeter-readme | FlowFeature.java, feature //81 | `idl_avg("Idle Mean","IlAG"` |
| cicflowmeter-readme | FlowFeature.java, feature //83 | `idl_max("Idle Max","IlMA"` |
| cicflowmeter-readme | FlowFeature.java, feature //84 | `idl_min("Idle Min","IlMI"` |
| cicflowmeter-readme | FlowFeature.java, feature //82 | `idl_std("Idle Std","IlSD"` |
| cicflowmeter-readme | FlowFeature.java, feature //47 | `pkt_len_avg("Packet Length Mean","PLAG"` |
| cicflowmeter-readme | FlowFeature.java, feature //46 | `pkt_len_max("Packet Length Max","PLMA"` |
| cicflowmeter-readme | FlowFeature.java, feature //45 | `pkt_len_min("Packet Length Min","PLMI"` |
| cicflowmeter-readme | FlowFeature.java, feature //48 | `pkt_len_std("Packet Length Std","PLSD"` |
| cicflowmeter-readme | FlowFeature.java, feature //49 | `pkt_len_var("Packet Length Variance","PLVA"` |
| cicflowmeter-readme | FlowFeature.java, feature //59 | `pkt_size_avg("Average Packet Size","PSAG"` |
| cicflowmeter-readme | FlowFeature.java, feature //6 | `prot("Protocol","PROT"` |
| cicflowmeter-readme | FlowFeature.java, feature //53 | `pst_cnt("PSH Flag Count","PSHCT"` |
| cicflowmeter-readme | FlowFeature.java, feature //52 | `rst_cnt("RST Flag Count","RSTCT"` |
| cicflowmeter-readme | FlowFeature.java, feature //72 | `subfl_bw_byt("Subflow Bwd Bytes","SFBwB"` |
| cicflowmeter-readme | FlowFeature.java, feature //71 | `subfl_bw_pkt("Subflow Bwd Packets","SFBwP"` |
| cicflowmeter-readme | FlowFeature.java, feature //70 | `subfl_fw_byt("Subflow Fwd Bytes","SFFwB"` |
| cicflowmeter-readme | FlowFeature.java, feature //69 | `subfl_fw_pkt("Subflow Fwd Packets","SFFwP"` |
| cicflowmeter-readme | FlowFeature.java, feature //51 | `syn_cnt("SYN Flag Count","SYNCT"` |
| cicflowmeter-readme | FlowFeature.java, feature //10 | `tot_bw_pkt("Total Bwd packets","TBwP"` |
| cicflowmeter-readme | FlowFeature.java, feature //9 | `tot_fw_pkt("Total Fwd Packet","TFwP"` |
| cicflowmeter-readme | FlowFeature.java, feature //55 | `urg_cnt("URG Flag Count","URGCT"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //75 | `Fw_act_pkt("Fwd Act Data Pkts","FwAP"` |
| cicflowmeter-src-2018 | BasicFlow.java, firstPacket(), forward branch | `Init_Win_bytes_forward = packet.getTCPWindow(); this.flowLengthStats.addValue((double)packet.getPayloadBytes());` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //54 | `ack_cnt("ACK Flag Cnt","ACKCT"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //77 | `atv_avg("Active Mean","AcAG"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //79 | `atv_max("Active Max","AcMA"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //80 | `atv_min("Active Min","AcMI"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //78 | `atv_std("Active Std","AcSD"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //68 | `bw_blk_rate_avg("Bwd Blk Rate Avg","BwBRAG"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //66 | `bw_byt_blk_avg("Bwd Byts/b Avg","BwB/BAG"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //42 | `bw_hdr_len("Bwd Header Len","BwHL"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //33 | `bw_iat_avg("Bwd IAT Mean","BwIATAG"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //35 | `bw_iat_max("Bwd IAT Max","BwIATMA"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //36 | `bw_iat_min("Bwd IAT Min","BwIATMI"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //34 | `bw_iat_std("Bwd IAT Std","BwIATSD"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //32 | `bw_iat_tot("Bwd IAT Tot","BwIATTO"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //67 | `bw_pkt_blk_avg("Bwd Pkts/b Avg","BwP/BAG"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //19 | `bw_pkt_l_avg("Bwd Pkt Len Mean","BwPLAG"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //17 | `bw_pkt_l_max("Bwd Pkt Len Max","BwPLMA"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //18 | `bw_pkt_l_min("Bwd Pkt Len Min","BwPLMI"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //20 | `bw_pkt_l_std("Bwd Pkt Len Std","BwPLSD"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //44 | `bw_pkt_s("Bwd Pkts/s","Bwp/s"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //38 | `bw_psh_flag("Bwd PSH Flags","BwPSH"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //61 | `bw_seg_avg("Bwd Seg Size Avg","BwSgAG"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //40 | `bw_urg_flag("Bwd URG Flags","BwURG"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //74 | `bw_win_byt("Init Bwd Win Byts","BwWB"` |
| cicflowmeter-src-2018 | FlowGenerator.java, addPacket(), packet without an open flow | `currentFlows.put(packet.getFlowId(), new BasicFlow(bidirectional,packet));` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //56 | `cwe_cnt("CWE Flag Count","CWECT"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //58 | `down_up_ratio("Down/Up Ratio","D/URO"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //5 | `dst_pot("Dst Port","DPT"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //57 | `ece_cnt("ECE Flag Cnt","ECECT"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //50 | `fin_cnt("FIN Flag Cnt","FINCT"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //21 | `fl_byt_s("Flow Byts/s","FB/s"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //8 | `fl_dur("Flow Duration","DUR"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //23 | `fl_iat_avg("Flow IAT Mean","FLIATAG"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //25 | `fl_iat_max("Flow IAT Max","FLIATMA"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //26 | `fl_iat_min("Flow IAT Min","FLIATMI"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //24 | `fl_iat_std("Flow IAT Std","FLIATSD"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //22 | `fl_pkt_s("Flow Pkts/s","FP/s"` |
| cicflowmeter-src-2018 | ReadPcapFileWorker.java, constructor | `flowTimeout = 120000000L;` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //65 | `fw_blk_rate_avg("Fwd Blk Rate Avg","FwBRAG"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //63 | `fw_byt_blk_avg("Fwd Byts/b Avg","FwB/BAG"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //41 | `fw_hdr_len("Fwd Header Len","FwHL"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //28 | `fw_iat_avg("Fwd IAT Mean","FwIATAG"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //30 | `fw_iat_max("Fwd IAT Max","FwIATMA"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //31 | `fw_iat_min("Fwd IAT Min","FwIATMI"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //29 | `fw_iat_std("Fwd IAT Std","FwIATSD"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //27 | `fw_iat_tot("Fwd IAT Tot","FwIATTO"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //64 | `fw_pkt_blk_avg("Fwd Pkts/b Avg","FwP/BAG"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //15 | `fw_pkt_l_avg("Fwd Pkt Len Mean","FwPLAG"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //13 | `fw_pkt_l_max("Fwd Pkt Len Max","FwPLMA"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //14 | `fw_pkt_l_min("Fwd Pkt Len Min","FwPLMI"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //16 | `fw_pkt_l_std("Fwd Pkt Len Std","FwPLSD"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //43 | `fw_pkt_s("Fwd Pkts/s","FwP/s"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //37 | `fw_psh_flag("Fwd PSH Flags","FwPSH"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //60 | `fw_seg_avg("Fwd Seg Size Avg","FwSgAG"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //76 | `fw_seg_min("Fwd Seg Size Min","FwSgMI"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //39 | `fw_urg_flag("Fwd URG Flags","FwURG"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //73 | `fw_win_byt("Init Fwd Win Byts","FwWB"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //81 | `idl_avg("Idle Mean","IlAG"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //83 | `idl_max("Idle Max","IlMA"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //84 | `idl_min("Idle Min","IlMI"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //82 | `idl_std("Idle Std","IlSD"` |
| cicflowmeter-src-2018 | FlowGenerator.java, addPacket() | `if((currentTimestamp -flow.getFlowStartTime())>flowTimeOut){` |
| cicflowmeter-src-2018 | BasicFlow.java, addPacket() | `if(packet.getPayloadBytes() >=1){ this.Act_data_pkt_forward++; }` |
| cicflowmeter-src-2018 | BasicFlow.java, firstPacket() | `if(this.src==null){ this.src = packet.getSrc();` |
| cicflowmeter-src-2018 | PacketReader.java, TCP branch | `packetInfo.setDstPort(tcp.destination());` |
| cicflowmeter-src-2018 | PacketReader.java, UDP branch | `packetInfo.setDstPort(udp.destination());` |
| cicflowmeter-src-2018 | PacketReader.java, TCP branch | `packetInfo.setFlagRST(tcp.flags_RST());` |
| cicflowmeter-src-2018 | PacketReader.java, TCP branch | `packetInfo.setHeaderBytes(tcp.getHeaderLength());` |
| cicflowmeter-src-2018 | PacketReader.java, TCP branch | `packetInfo.setPayloadBytes(tcp.getPayloadLength());` |
| cicflowmeter-src-2018 | PacketReader.java, UDP branch | `packetInfo.setPayloadBytes(udp.getPayloadLength());` |
| cicflowmeter-src-2018 | PacketReader.java, UDP branch | `packetInfo.setProtocol(17);` |
| cicflowmeter-src-2018 | PacketReader.java, TCP branch | `packetInfo.setProtocol(6);` |
| cicflowmeter-src-2018 | PacketReader.java, TCP branch | `packetInfo.setTCPWindow(tcp.window());` |
| cicflowmeter-src-2018 | PacketReader.java, IPv4 packet parsing | `packetInfo.setTimeStamp(packet.getCaptureHeader().timestampInMicros());` |
| cicflowmeter-src-2018 | FlowOfflinePane.java, initSettingPane() | `param1Ele.add(120000000L);` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //47 | `pkt_len_avg("Pkt Len Mean","PLAG"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //46 | `pkt_len_max("Pkt Len Max","PLMA"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //45 | `pkt_len_min("Pkt Len Min","PLMI"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //48 | `pkt_len_std("Pkt Len Std","PLSD"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //49 | `pkt_len_var("Pkt Len Var","PLVA"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //59 | `pkt_size_avg("Pkt Size Avg","PSAG"` |
| cicflowmeter-src-2018 | BasicFlow.java, field declaration | `private HashMap<String, MutableInt> flagCounts;` |
| cicflowmeter-src-2018 | BasicPacketInfo.java, field declaration | `private int TCPWindow=-1;` |
| cicflowmeter-src-2018 | BasicPacketInfo.java, field declarations | `private int protocol;` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //6 | `prot("Protocol","PROT"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //53 | `pst_cnt("PSH Flag Cnt","PSHCT"` |
| cicflowmeter-src-2018 | BasicFlow.java, getDownUpRatio() | `return (double)(this.backward.size()/this.forward.size());` |
| cicflowmeter-src-2018 | BasicFlow.java, getAvgPacketSize() | `return (this.flowLengthStats.getSum()/this.packetCount());` |
| cicflowmeter-src-2018 | BasicFlow.java, getSflow_fpackets() | `return this.forward.size()/sfCount;` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //52 | `rst_cnt("RST Flag Cnt","RSTCT"` |
| cicflowmeter-src-2018 | build.gradle | `sourceCompatibility = 1.8` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //72 | `subfl_bw_byt("Subflow Bwd Byts","SFBwB"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //71 | `subfl_bw_pkt("Subflow Bwd Pkts","SFBwP"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //70 | `subfl_fw_byt("Subflow Fwd Byts","SFFwB"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //69 | `subfl_fw_pkt("Subflow Fwd Pkts","SFFwP"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //51 | `syn_cnt("SYN Flag Cnt","SYNCT"` |
| cicflowmeter-src-2018 | BasicFlow.java, addPacket(), backward branch | `this.bwdPktStats.addValue((double)packet.getPayloadBytes()); Init_Win_bytes_backward = packet.getTCPWindow();` |
| cicflowmeter-src-2018 | BasicFlow.java, firstPacket() | `this.endActiveTime = packet.getTimeStamp(); this.flowLengthStats.addValue((double)packet.getPayloadBytes());` |
| cicflowmeter-src-2018 | BasicFlow.java, firstPacket() | `this.fwdPktStats.addValue((double)packet.getPayloadBytes());` |
| cicflowmeter-src-2018 | BasicFlow.java, addPacket() | `this.min_seg_size_forward = Math.min(packet.getHeaderBytes(),this.min_seg_size_forward);` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //10 | `tot_bw_pkt("Tot Bwd Pkts","TBwP"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //9 | `tot_fw_pkt("Tot Fwd Pkts","TFwP"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //12 | `tot_l_bw_pkt("TotLen Bwd Pkts","TLBwP"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //11 | `tot_l_fw_pkt("TotLen Fwd Pkts","TLFwP"` |
| cicflowmeter-src-2018 | FlowFeature.java, feature //55 | `urg_cnt("URG Flag Cnt","URGCT"` |
| cicflowmeter-src-2018 | FlowGenerator.java, addPacket() | `}else if(packet.hasFlagFIN()){` |
| cicflowmeter-src-2018-06 | BasicFlow.java, CSV row writer | `dump.append(flagCounts.get("FIN").value).append(separator); //50 dump.append(flagCounts.get("SYN").value).append(separator); //51` |
| layer1-code | train_protocol_b.py, configuration | `AE_BOTTLENECK = 16` |
