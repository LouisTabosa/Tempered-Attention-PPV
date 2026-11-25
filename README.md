# Tempered attention as an aggregation law for PPV in multi-delay blasts

Code and results for the article *Tempered Attention as an Aggregation Law for Peak Particle Velocity in Multi-Delay Blasts* (J. J. S. de Figueiredo, L. D. G. Tabosa and S. S. Martins, submitted to IEEE Access).

The USBM scaled-distance law is the numerator of an attention score. For a blast with $N$ delay groups, the proposed law aggregates the per-delay predictions with a tempered log-sum-exp,

$$\ln \mathrm{PPV} = \tau \ln \sum_{i=1}^{N} \exp(s_i/\tau), \qquad s_i = \ln K - \beta \ln D_i + \alpha\beta \ln W_i,$$

where $\tau \to 0$ recovers the USBM law with maximum charge per delay and $\tau = 1$ the in-phase sum of all delays. The temperature may depend on the delay interval measured in cycles of the wave, $\tau(x) = \tau_{\min} + (1-\tau_{\min})\,\sigma(a - b\,x)$ with $x = \Delta t\,\hat f_d$.

## Contents

| File | Content |
|---|---|
| `ppv_attention.py` | Synthetic generator (superposition of signature-hole waveforms), models (USBM, gradient boosting, generic attention, physics-structured attention) and metrics |
| `experiments.py` | Synthetic experiments E1 to E7 |
| `run_all.py` | Runs E1 to E7 and writes `results.json` (and quick-look figures in `fig/`) |
| `results.json` | Results of E1 to E7 used in the article, including the per-seed metrics |
| `make_figures.py` | Figures of the article, from `results.json` and `real_data/` |
| `figures/` | Figures of the article (PDF and PNG) |
| `real_data/real_data_test.py` | Pilot test with field data: per-segment table, site law, equal-delay attention law and metrics |
| `real_data/recalibrated_usbm.py` | Comparison with a USBM law recalibrated with a constant bias, overall and by blast |
| `real_data/duration_check.py` | 5–95% energy duration of the multi-hole blasts against the signature hole |
| `real_data/*.csv`, `real_data/*.json` | Outputs of the three scripts above, as used in the article |

## Requirements

Python 3.10 or later and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

## Reproducing the results

Synthetic experiments (about 16 min on 2 cores):

```
python run_all.py
```

Field data. The records come from the reproduction package [Waveform-Simulation-PPV](https://github.com/LouisTabosa/Waveform-Simulation-PPV) (field data under CC-BY-4.0). Download or clone it and pass its root folder:

```
python real_data/real_data_test.py <path to Waveform-Simulation-PPV>
python real_data/recalibrated_usbm.py
python real_data/duration_check.py <path to Waveform-Simulation-PPV>
```

Figures (uses the stored results; the E5 figure refits the proposed model on the 20 training sets of E2, about 1 min):

```
python make_figures.py
```

All random draws use fixed seeds, so the scripts reproduce the numbers in `results.json` and in the article.

## Field data used

One campaign in an open-pit iron-ore mine: one signature hole (153 kg) and four electronically initiated blasts with 153 kg per delay (Test 2 with 6 holes; Tests 1, 3 and 4 with 15 holes), fired as separate events about 2 s apart and recorded by the same triaxial stations. The firing plan (time and position of each hole) is not available, so the field test uses the equal-delay form of the law, $\ln \mathrm{PPV} = s + \tau \ln N$.

## License and citation

- Code: MIT (see `LICENSE`).
- Field data: CC-BY-4.0, distributed with Waveform-Simulation-PPV.
- If you use this code, please cite the article (see `CITATION.cff`).
