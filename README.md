# Porsche Global Performance Intelligence

## Project Overview

Porsche Global Performance Intelligence is a static Business Intelligence dashboard built with Python, Pandas, Plotly, and HTML/CSS. It generates a realistic synthetic dataset inspired by Porsche's publicly reported delivery figures from 2018 through 2024, then publishes a self-contained `index.html` suitable for GitHub Pages.

## Business Problem

Porsche's internal operating data is private, but executives, analysts, and portfolio teams still need a way to understand delivery trends, regional exposure, model profitability, and electrification momentum. This project simulates that executive analysis environment using public annual delivery anchors and transparent synthetic assumptions.

## Dataset Description

The generated CSV lives at `data/porsche_global_performance_synthetic.csv`.

Columns included:

- `Year`
- `Model`
- `Region`
- `Units Sold`
- `Revenue in EUR millions`
- `Average Transaction Price in EUR`
- `Customer Satisfaction Score (1-10)`
- `Dealer Count`
- `Electric vs ICE vs Hybrid classification`

The global annual totals are anchored to Porsche's public delivery figures, including 320,221 vehicles in 2023 and 310,718 vehicles in 2024. Model and regional splits are synthetic but designed to remain directionally consistent with Porsche's reported performance: Cayenne and Macan drive scale, 911 protects premium pricing, Taycan launched in 2019, and China softened materially in 2024.

## Methodology

1. Use public Porsche delivery totals from 2018-2024 as annual control totals.
2. Create a synthetic model mix that follows reported model-line behavior and known launch timing.
3. Allocate units by region using realistic regional weights and 2024 market-share changes.
4. Estimate revenue from model-level average transaction prices, regional price indexes, and powertrain mix.
5. Build Plotly charts for trend, region, profitability, electrification, and forecast analysis.
6. Export a single self-contained `index.html` with Plotly embedded for static hosting.

## Key Findings

- Porsche's 2024 delivery base remains resilient despite a product transition year.
- Cayenne and Macan continue to drive volume, while the 911 anchors pricing power and brand premium.
- North America and Europe are the strongest 2024 revenue contributors in the synthetic model.
- China weakness creates a strategic need for localized EV, software, and dealer-experience moves.
- Electrified models lift revenue per unit, but adoption volatility makes launch execution critical.

## Technologies Used

- Python
- Pandas
- NumPy
- Plotly
- HTML/CSS
- GitHub Pages

## How to Run

Install the Python dependencies:

```bash
pip install pandas numpy plotly
```

Generate the dashboard and CSV:

```bash
python generate_dashboard.py
```

Open `index.html` in a browser. No server or external JavaScript is required because Plotly is embedded directly in the file.

## GitHub Pages

This repo is ready for GitHub Pages. Push the repository to GitHub, then enable Pages from the repository settings with the source set to the main branch root. GitHub Pages will serve `index.html` directly.
