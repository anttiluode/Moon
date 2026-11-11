# Hand-Written Moon

A light-transport network for a translucent quasicrystal moon. Every weight is written by
`build_network.py` from radiative-transfer formulas. No training data, no optimiser.

- `build_network.py`  sets all weights, writes `network.json` (byte-identical on rerun)
- `pt.c`              reference path tracer (delta tracking, HG g=0.5, no absorption); `gcc -O3 -fopenmp pt.c -lm -o pt`
- `pairs.py`          12 reference and 6 held-back camera/sun pairs; `runrefs.py` renders them into `ref/`, `held/`
- `feat.py`           probe stencils for every ray-march sample of a view (cached to `cache/`)
- `exp.py`            the formulas in plain numpy; `netscore.py` runs the weights through a generic interpreter and scores them
- `browser_test.py`   scores the compiled, baked WebGL shader in headless Chromium
- `page/moon.html`    the live page (weights + compiler + renderer in one file); `page/app.js` is its script

Error = mean relative luminance error on moon pixels, sun + sky + ground combined.

|                                   | reference (12) | held back (6) |
|-----------------------------------|----------------|---------------|
| single scattering, sunlight       | 81.6%          | 79.7%         |
| written network, sunlight         | 14.4%          | 14.8%         |
| written network, all lights       | 7.1%           | 7.1%          |
| compiled + baked in the browser   | 6.7%           | 6.9%          |
| Monte Carlo noise in references   | 0.6%           | 0.6%          |

Two constants (BETA, LAM) were set by hand against the reference views. One scene, one medium.
