# RQ C — Initial Results (Gemini flash vs pro, 19 human-labelled traces)

## Table 1 — Agreement per trace

| Trace | Framework | gemini-2.5-flash | gemini-2.5-pro |
|---|---|---|---|
| AG2_tid12 | AG2 | 71.4 | 78.6 |
| AG2_tid2 | AG2 | 85.7 | 85.7 |
| AG2_tid7 | AG2 | 78.6 | 85.7 |
| AppWorld_tid0 | AppWorld | 50.0 | 78.6 |
| AppWorld_tid11 | AppWorld | 57.1 | 71.4 |
| AppWorld_tid5 | AppWorld | 85.7 | 85.7 |
| ChatDev_tid13 | ChatDev | 85.7 | 78.6 |
| ChatDev_tid15 | ChatDev | 78.6 | 78.6 |
| ChatDev_tid3 | ChatDev | 57.1 | 57.1 |
| ChatDev_tid8 | ChatDev | 71.4 | 71.4 |
| GAIA_tid17 | GAIA | 85.7 | 71.4 |
| GAIA_tid18 | GAIA | 64.3 | 71.4 |
| HyperAgent_tid1 | HyperAgent | 35.7 | 28.6 |
| HyperAgent_tid10 | HyperAgent | 57.1 | 42.9 |
| HyperAgent_tid6 | HyperAgent | 85.7 | 71.4 |
| MetaGPT_tid14 | MetaGPT | 57.1 | 57.1 |
| MetaGPT_tid16 | MetaGPT | 50.0 | 64.3 |
| MetaGPT_tid4 | MetaGPT | 42.9 | 57.1 |
| MetaGPT_tid9 | MetaGPT | 57.1 | 71.4 |
| **MEAN** | | **66.2** | **68.8** |

## Table 2 — Per-framework mean agreement

| Framework | gemini-2.5-flash | gemini-2.5-pro | n |
|---|---|---|---|
| AG2 | 78.6 | 83.3 | 3 |
| AppWorld | 64.3 | 78.6 | 3 |
| ChatDev | 73.2 | 71.4 | 4 |
| GAIA | 75.0 | 71.4 | 2 |
| HyperAgent | 59.5 | 47.6 | 3 |
| MetaGPT | 51.8 | 62.5 | 4 |

## Table 3 — Per-mode error breakdown

| Mode | Accuracy % | HumanYES | JudgeYES | FalsePos | FalseNeg |
|---|---|---|---|---|---|
| 1.1 | 68.4 | 12 | 16 | 8 | 4 |
| 1.2 | 78.9 | 4 | 4 | 4 | 4 |
| 1.3 | 50.0 | 4 | 23 | 19 | 0 |
| 1.4 | 73.7 | 4 | 6 | 6 | 4 |
| 1.5 | 68.4 | 10 | 10 | 6 | 6 |
| 2.1 | 76.3 | 4 | 5 | 5 | 4 |
| 2.2 | 65.8 | 10 | 7 | 5 | 8 |
| 2.3 | 63.2 | 10 | 8 | 6 | 8 |
| 2.4 | 81.6 | 0 | 7 | 7 | 0 |
| 2.5 | 84.2 | 6 | 0 | 0 | 6 |
| 2.6 | 47.4 | 8 | 16 | 14 | 6 |
| 3.1 | 78.9 | 0 | 8 | 8 | 0 |
| 3.2 | 68.4 | 14 | 4 | 1 | 11 |
| 3.3 | 39.5 | 22 | 17 | 9 | 14 |

*Localization map: in progress (step-decomposition recently fixed).*