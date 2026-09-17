# Emberline glossary

Terms as used in this repository and its documents. Where a term has a
precise meaning in the code, the module is named.

| term | meaning here |
|---|---|
| **Adapter** | A module in `emberline/adapters/` that would fetch real data (USGS 3DEP elevation, LANDFIRE fuel, OpenStreetMap roads) and return the same arrays the synthetic world generator produces. All three are **stubs** that raise `NotImplementedError`. |
| **ADVISORY** | The label on every `RoutePlan` (`foresight/routing.py`): routing output supports a human decision and is never an instruction. |
| **Arrival MAE** | Mean absolute error, in minutes, between the surrogate's and the physics model's first-touch time per cell, over cells both reach within +60 min (10-minute resolution). `surrogate/eval.py`. |
| **BME680 / BME688** | Bosch environmental sensor on the planned node: temperature, humidity, pressure and heater-plate **gas resistance in ohms**. Emberline uses the raw resistance (as a log-ratio against a rolling baseline), not the vendor's VOC index. PLANNED hardware. |
| **BPR** | Bureau of Public Roads congestion function; routing costs each edge as `length × (1 + α·(load/capacity)²)` with α = 0.15. |
| **CA (cellular automaton)** | The grid-based fire model in `emberline/firesim/`: each burning cell ignites neighbours with a probability derived from the rate of spread. |
| **CAD** | Channel Activity Detection: a LoRa radio's listen-before-talk. Modelled in `MeshSim.tx/_attempt` as a single collision domain. |
| **Calibration / ECE** | Whether a predicted probability matches observed frequency. **Expected Calibration Error** is the bin-weighted mean gap between predicted P(burn) and the observed burn fraction. Shipped raw ensemble ECE is 0.060 on val fires. `surrogate/calibration.py`. |
| **CAP** | OASIS Common Alerting Protocol 1.2, the schema behind IPAWS/EAS alerts. Emberline writes CAP-shaped JSON **drafts** with `status: "Exercise"` to `outbox/` and never transmits them. `foresight/cap.py`. |
| **Cascade (Tier 2)** | The full siren cascade: every reachable node sounds. Requires ≥ 2 distinct healthy-enough nodes and a weighted score ≥ 1.60 inside a 120-s window. |
| **Chirp (Tier 0)** | A local audible chirp at one node whose classifier confidence is ≥ 0.60. Nothing else happens beyond a DETECT packet toward the cluster head. |
| **Cluster head** | The alive node maximising degree × health; aggregates DETECT packets and decides escalation. Re-elected when its heartbeats go silent. `mesh/escalation.py`. |
| **Cone** | Per-horizon map of P(burn by +10/+30/+60 min) from an ensemble, drawn at thresholds 0.10/0.30/0.60 and dilated by 4 cells for safety. The **routing mask** is the +30-min map at P ≥ 0.30. `foresight/__init__.py`. |
| **Confounder** | A non-fire event that moves the same sensor channels: bbq, wood_stove, vehicle, fog, dust, aerosol. Signatures are **authored** in `sensors/events.py`. |
| **Corroboration** | Agreement between distinct nodes inside the corroboration window; score = Σ (health weight × confidence). Health weights are floored at 0.2 so a degraded node can contribute but never cascade alone. |
| **DES** | Discrete-event simulation: the mesh advances by scheduled events (transmissions, deliveries, heartbeats) rather than fixed ticks. `mesh/__init__.py`. |
| **Duty cycle** | Fraction of time a node may transmit; 1 % budget per node, a config constant citing FCC Part 15 / ETSI practice. Enforced in `MeshSim.duty_budget_ms`. |
| **Ensemble** | Many perturbed fire runs (ignition jitter, wind direction/speed, ROS roughness); the fraction of members that burned a cell is its probability. Physics ensemble in `firesim/ensemble.py`; batched surrogate ensemble in `surrogate/rollout.py`. |
| **Event-level split** | Train/validation split that keeps all windows of one event (fire or confounder) on the same side, so near-duplicate windows cannot leak. `detect/train.py::event_split`. |
| **F1 / precision / recall** | Standard classification metrics on validation windows: precision = TP/(TP+FP), recall = TP/(TP+FN), F1 their harmonic mean. |
| **False positives per node-day** | Positive classifications per node over a simulated 24-hour fire-free, confounder-rich day, classifier slid every 30 s. Measured 7.33 (CNN). Single-node figure, before corroboration. |
| **Foresight** | The decision-support layer: ignition estimate from bearings, surrogate/physics ensemble cones, evacuation routing, CAP draft. `emberline/foresight/`. |
| **GBM** | Gradient-boosted machine, sklearn `GradientBoostingClassifier` on 23 handcrafted window features; the detection baseline that the CNN did not beat (F1 0.952 vs 0.943). |
| **Held-out wind regime** | Training worlds whose base wind direction falls in 80–130° are excluded from training and validation entirely and reported separately as a never-seen-regime test. |
| **Hindcast** | Replaying a scenario (ignition, wind, start hour, authored first-911 minute) through sensors → classifier → mesh to compute `warning_minutes_gained = report_911_min − tier2_min`. Current scenarios are hand-authored, not fire records. |
| **IoU** | Intersection over union of the surrogate's touched (burned or burning) mask and the physics mask at a horizon. IoU@+30 on val worlds is 0.627. |
| **IPAWS / WEA / EAS** | The US federal alerting pipeline (Integrated Public Alert and Warning System), Wireless Emergency Alerts and the Emergency Alert System. Emberline has **no** integration with any of them. |
| **LoRa** | Long-range, low-power sub-GHz radio modulation (915 MHz in the US ISM band). Modelled at SF10-class sensitivity (−129 dBm); **no real radio exists in this project yet**. |
| **Node-day** | One node observed for one day; the unit for false-positive rates. |
| **OU (Ornstein–Uhlenbeck)** | Mean-reverting random process used for temporally correlated wind gusts and direction wander. `worldgen/wind.py`. |
| **Outbox** | `outbox/` directory where CAP drafts land. Nothing reads it automatically; a human would. |
| **PMS5003** | Plantower optical particulate sensor (PM1/PM2.5/PM10) on the planned node. Optical sensors misread fog droplets as particles, which is why fog is a confounder. PLANNED hardware. |
| **Plume (Gaussian)** | `C = Q/(π σ_y σ_z U)·exp(−y²/2σ_y²)`: ground-level concentration downwind of point sources with full reflection; σ grows linearly with distance. `sensors/plume.py`. |
| **ROS** | Rate of spread of the fire front, m/s. `ROS = R0 · φ_fuel · φ_wind · φ_slope`. |
| **Rothermel** | Rothermel's 1972 surface fire spread model; Emberline keeps its multiplicative structure and slope factor and collapses the rest to per-fuel scalars. Surface spread only: no crown fire, spotting or fire–atmosphere coupling. |
| **Self-heal** | Heartbeat mourning of any silent node and re-election of a silent cluster head; flooding needs no route repair. Demo: N3 killed at t+390 s, mourned at t+570 s. |
| **Shard** | One compressed `.npz` per training world holding all its fires; the unit of storage and of the world-level split. |
| **Sim-to-real** | The gap between behaviour under the repo's own models and behaviour in the field. Every number in this repo is on the "sim" side. |
| **Storm suppression** | Measures that stop 50 simultaneous detections from flooding the channel: duplicate suppression, per-node DETECT rate limit (30 s), head aggregation, and post-Tier-2 relays dropping lower-tier traffic. Side effect: a second fire reported after a cascade is not heard. |
| **Surrogate** | A neural network (`FireUNet`, 371,858 parameters) trained to imitate the physics CA one 10-minute step at a time; +30/+60 forecasts are 3/6 autoregressive steps. |
| **Sustained-single** | Tier-1 path for a lone healthy node: ≥ 2 reports ≥ 0.85 at least 20 s apart, added after a barbecue tail voice-alerted the whole simulated town in development. |
| **Temperature scaling** | Post-hoc calibration `sigmoid(logit(p)/T)`. A T = 2.37 was fitted and passed its disjoint-world check but worsened val ECE, so `foresight.temperature` stays `null`. |
| **Tier 0 / 1 / 2** | Chirp / voice with bearing / full cascade. See *Escalation ladder* in [00_OVERVIEW.md](00_OVERVIEW.md). |
| **Touched** | A cell that is burning or burned; the mask the surrogate predicts and IoU is computed on. |
| **Twin (digital twin)** | The terrain + fuel + town + wind model Foresight forecasts on. Today synthetic (`worldgen`); real twin requires the adapters. |
| **UNet** | Encoder–decoder convolutional network with skip connections; fully convolutional, so it trains on 64×64 crops and evaluates on the 256×256 grid. |
| **Voice + bearing (Tier 1)** | Spoken alert naming the direction smoke comes from. Bearings in the simulation are "upwind of the node", not a measured quantity. |
| **Warning minutes gained** | `report_911_min − tier2_min` in a hindcast; positive means the cascade beat the authored human report. |
| **World** | One seeded synthetic landscape (`World` dataclass): 256 × 256 cells at 10 m. World 0 is the demo world; 1–160 train/val/holdout; 161–172 calibration; 300–309 and 320–324 detection layouts. |
| **WUI** | Wildland-urban interface: where homes meet flammable vegetation; the deployment setting and the insurers' exposure. |

---

Previous: [05_ROADMAP.md](05_ROADMAP.md). Index: [README.md](../README.md).
