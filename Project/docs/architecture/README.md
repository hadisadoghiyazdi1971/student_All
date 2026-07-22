# Architecture diagram

`digital_twin_architecture.tex` is the presentation-ready, standalone TikZ overview of the complete project.

Compile from this directory with:

```bash
pdflatex digital_twin_architecture.tex
```

The solid path shows the currently implemented batch workflow. Dashed elements show the closed feedback loop and the recommended upgrade toward an adaptive, continuously recalibrated twin.

The diagram intentionally distinguishes three concepts:

1. **Physical person** — the real system being observed and influenced.
2. **Digital state** — the synchronized five-factor representation (`HI_v2`).
3. **Predictive/decision twin** — the model that forecasts the next state and selects an action whose outcome returns as new evidence.
