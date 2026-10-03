# VCM benchmark - Edward Vincent Duero - 20261003-090010

Wake word: **Hey Alfred** - trials: 202 with the wake word + 16 without - shuffle seed: 231 - connection: ssh - holdout: huggingface

Pi log file: ~/vcm_benchmark/alfred_20261003T010348Z_2151.log

## At a glance

|                                   | overall     | real voice  | synthetic voice |
|-----------------------------------|-------------|-------------|-----------------|
| intent accuracy (19)              | 52.5%       | 40.6%       | 63.2%           |
| command accuracy (93)             | 49.5%       | 37.5%       | 60.4%           |
| false accept (out of scope fired) | 6.2% (1/16) | 0.0% (0/10) | 16.7% (1/6)     |
| false reject (command ignored)    | 50.5%       | 65.1%       | 38.0%           |
| false wake (no wake word, fired)  | 0.0% (0/16) | 0.0% (0/5)  | 0.0% (0/11)     |
| slot exact                        | 87.8%       | 76.9%       | 91.7%           |
| latency p95                       | 1.68 s      | 1.63 s      | 1.71 s          |

**Pi:** real-time factor 0.005 (p95 0.006), inference 9 ms, CPU temp max 56.8 C, runtime CPU 12% mean, runtime RAM 543 MB peak, 659 KFLOP per inference

# Detailed metrics

## Classification

| metric                                            | 19 intents (+reject) | 93 commands (+reject) |
|---------------------------------------------------|----------------------|-----------------------|
| accuracy                                          | 52.5%                | 49.5%                 |
| balanced accuracy                                 | 53.3%                | 46.2%                 |
| precision (macro)                                 | 93.7%                | 70.0%                 |
| recall (macro)                                    | 53.3%                | 46.2%                 |
| F1 (macro)                                        | 62.6%                | 53.6%                 |
| F2 (macro)                                        | 55.0%                | 48.4%                 |
| false accept rate (OOS fired)                     | 6.2%                 | 6.2%                  |
| false reject rate (in-scope silent/rejected)      | 50.5%                | 50.5%                 |
| misfire rate (wrong command fired)                | 0.5%                 | 3.8%                  |
| accuracy 95% CI                                   | [46-59%]             | [43-56%]              |
| false accept 95% CI                               | [1-28%] (1/16)       | [1-28%]               |
| false wake rate (command without wake word fired) | 0.0% [0-19%] (0/16)  | 0.0% [0-19%] (0/16)   |

Responses: 90.1% of trials fired a command; no response: 20; extra fires: 0; wake detect rate: 90.1%

## Overall vs real vs synthetic voices

Each group is scored on its own. '-' = the group has no clips of that kind. The holdout's 10 out-of-scope clips are all real recordings (none are synthetic), so there is no false accept rate for synthetic voices.

| metric                         | overall        | real voice     | synthetic voice |
|--------------------------------|----------------|----------------|-----------------|
| clips (with wake word)         | 202            | 96             | 106             |
| **19 intents** accuracy        | 52.5% [46-59%] | 40.6% [31-51%] | 63.2% [54-72%]  |
| balanced accuracy              | 53.3%          | 40.1%          | 64.8%           |
| F1 (macro)                     | 62.6%          | 46.0%          | 72.2%           |
| F2 (macro)                     | 55.0%          | 40.5%          | 66.0%           |
| false accept rate              | 6.2% (1/16)    | 0.0% (0/10)    | 16.7% (1/6)     |
| false reject rate              | 50.5%          | 65.1%          | 38.0%           |
| misfire rate                   | 0.5%           | 1.2%           | 0.0%            |
| **93 commands** accuracy       | 49.5%          | 37.5%          | 60.4%           |
| balanced accuracy              | 46.2%          | 31.0%          | 59.4%           |
| F1 (macro)                     | 53.6%          | 29.8%          | 59.4%           |
| F2 (macro)                     | 48.4%          | 30.2%          | 59.1%           |
| misfire rate                   | 3.8%           | 4.7%           | 3.0%            |
| slot exact (intent right)      | 87.8% (n=49)   | 76.9% (n=13)   | 91.7% (n=36)    |
| latency p50 / p95              | 1.49 / 1.68 s  | 1.47 / 1.63 s  | 1.50 / 1.71 s   |
| false wake rate (no wake word) | 0.0% (0/16)    | 0.0% (0/5)     | 0.0% (0/11)     |

## Slot values (slotted intents, intent right)

abs error = Manhattan (L1) distance in the slot's unit (alarm: minutes, circular over 24 h); rel error = abs error / spread of the 3 schema values; phonetic / char distance = normalised edit distance (0 same, 1 completely different) of simplified-Metaphone keys / spelled-out text.

| intent          | n  | exact  | mean abs error | mean rel error | phonetic dist | char dist |
|-----------------|----|--------|----------------|----------------|---------------|-----------|
| ALARM           | 12 | 91.7%  | 0.0 min        | 0.000          | 0.337         | 0.393     |
| BRIGHTNESS      | 9  | 77.8%  | 10.0 %         | 0.125          | 0.086         | 0.088     |
| COLOR           | 11 | 90.9%  | -              | -              | 0.091         | 0.068     |
| CREATE_REMINDER | 3  | 66.7%  | -              | -              | 0.125         | 0.121     |
| TEMPERATURE     | 7  | 85.7%  | 0.6 deg        | 0.071          | 0.071         | 0.071     |
| TIMER           | 7  | 100.0% | 0.0 s          | 0.000          | 0.000         | 0.010     |
| ALL             | 49 | 87.8%  | -              | 0.048          | 0.136         | 0.147     |

## Raspberry Pi

- **Raspberry Pi 5 Model B Rev 1.1**, 4 cores  up to 2400.0 MHz, RAM 8062.3 MB, Debian GNU/Linux 13 (trixie), kernel 6.18.50+rpt-rpi-2712, Python 3.13.5
- packages: numpy 2.2.4

| metric                                      | mean / p95 / max         |
|---------------------------------------------|--------------------------|
| response latency (command end -> Pi output) | 1.321 / 1.679 / 2.579 s  |
| latency p50 / p99                           | 1.489 / 2.228 s          |
| inference time (Pi-reported)                | 8.8 / 12.4 / 16.8 ms     |
| real-time factor (infer / audio window)     | 0.005 / 0.006 / 0.008    |
| CPU temperature                             | 50.7 / 52.4 / 56.8 C     |
| CPU use, whole Pi                           | 3.3 / 11.9 / 47.8 %      |
| CPU use, your runtime process               | 11.9 / 46.6 / 189.9 %    |
| RAM (RSS), your runtime process             | 364.5 / 373.8 / 543.3 MB |
| RAM used, whole Pi                          | 758.3 / 767.3 / 930.4 MB |
| CPU clock                                   | 1715 / 2400 / 2400 MHz   |
| load average (1 min)                        | 0.15 / 0.29 / 0.43       |
| runtime CPU-seconds per second of speech    | 1.110                    |
| runtime CPU share of wall time              | 11.1%                    |
| throttling flags seen                       | none                     |
| test wall time                              | 60.1 min                 |
| model parameters                            | 1,104,613                |
| model size                                  | 4.34 MB                  |
| model FLOPs per inference                   | 659 KFLOP                |
| effective GFLOP/s (FLOPs / mean infer time) | 0.07                     |

## Most frequent confusions

**intent level:** CREATE_REMINDER -> REJECT (15); TEMPERATURE -> REJECT (11); TIMER -> REJECT (11); BRIGHTNESS -> REJECT (9); COLOR -> REJECT (7); ALARM -> REJECT (6); WEATHER -> REJECT (5); PAUSE -> REJECT (4); MESSAGE -> REJECT (4); NEXT -> REJECT (3)

**command level:** Reminder Drink water -> REJECT (2); Set color to Red -> REJECT (2); Reminder Exercise -> REJECT (2); Reminder Study -> REJECT (2); Set the temperature to 26 degrees -> REJECT (2); Remind me to Exercise -> REJECT (2); Weather -> REJECT (2); Change the temperature to 26 degrees -> REJECT (2); Change the temperature to 22 degrees -> REJECT (2); Alarm 8:00 AM -> REJECT (2)

## Per-intent scores

| class           | n  | precision | recall | F1    | F2    |
|-----------------|----|-----------|--------|-------|-------|
| ALARM           | 18 | 100.0%    | 66.7%  | 80.0% | 71.4% |
| BRIGHTNESS      | 18 | 100.0%    | 50.0%  | 66.7% | 55.6% |
| CALL            | 6  | 100.0%    | 83.3%  | 90.9% | 86.2% |
| COLOR           | 18 | 100.0%    | 61.1%  | 75.9% | 66.3% |
| CREATE_REMINDER | 18 | 100.0%    | 16.7%  | 28.6% | 20.0% |
| LIGHT_OFF       | 6  | 80.0%     | 66.7%  | 72.7% | 69.0% |
| LIGHT_ON        | 6  | 100.0%    | 83.3%  | 90.9% | 86.2% |
| LIST_REMINDERS  | 6  | 100.0%    | 50.0%  | 66.7% | 55.6% |
| MESSAGE         | 6  | 100.0%    | 33.3%  | 50.0% | 38.5% |
| NEXT            | 6  | 100.0%    | 50.0%  | 66.7% | 55.6% |
| PAUSE           | 6  | 100.0%    | 33.3%  | 50.0% | 38.5% |
| PLAY_MUSIC      | 6  | 100.0%    | 66.7%  | 80.0% | 71.4% |
| REJECT          | 16 | 13.8%     | 93.8%  | 24.0% | 43.4% |
| STOP            | 6  | 100.0%    | 50.0%  | 66.7% | 55.6% |
| TEMPERATURE     | 18 | 100.0%    | 38.9%  | 56.0% | 44.3% |
| TIME            | 6  | 100.0%    | 66.7%  | 80.0% | 71.4% |
| TIMER           | 18 | 100.0%    | 38.9%  | 56.0% | 44.3% |
| VOLUME_DOWN     | 6  | 100.0%    | 33.3%  | 50.0% | 38.5% |
| VOLUME_UP       | 6  | 80.0%     | 66.7%  | 72.7% | 69.0% |
| WEATHER         | 6  | 100.0%    | 16.7%  | 28.6% | 20.0% |

Scoring notes: REJECT = out-of-scope truth, or the Pi answered out-of-scope / did not respond. Command level: a prediction matches a variation when intent and slot are right (the Pi does not predict the wording); wrong predictions count against the first variation of their (intent, slot). Macro scores average over classes present in the holdout. False accept rate rests on only the out-of-scope clips in the holdout, so read its confidence interval. False wake rate: in-scope commands played WITHOUT the wake word (as many as the out-of-scope clips); any command the Pi fires for them is a false wake. These trials are not part of the 19/93 scores.
