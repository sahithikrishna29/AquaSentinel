AquaSentinel Beas River source compilation — REAL SOURCE EXTRACTS

This package contains source-backed values extracted from:
1) CPCB NWMP 2022 River Beas annual station table
   https://cpcb.nic.in/wqm/2022/WQuality_River-Data-2022.pdf
2) CPCB NWMP 2023 River Beas annual station table
   https://cpcb.nic.in/wqm/2023/WQuality_River-Data-2023.pdf
3) Himachal Pradesh State Pollution Control Board (HPSPCB), MINARS 2021-22 monthly tables
   https://hppcb.nic.in/waterquality/WQ22.pdf

Files:
- beas_real_source_compilation_long.csv: combined long-format compilation.
- beas_cpcb_annual_ranges_2022_2023.csv: station/year/parameter annual minimum and maximum ranges.
- beas_hppcb_monthly_observations_2021_22.csv: monthly point observations for six Beas stations and four parameters.
- beas_parameter_source_inventory.csv: verified parameter coverage and gaps.

Important interpretation:
- CPCB annual table values are min/max ranges for a station and year. They are NOT monthly or single-sample observations. Keep `record_type=annual_min_max_range`.
- HPSPCB MINARS table contains monthly observations. The report's source month columns were reordered to calendar order. The date is represented as YYYY-MM-15 only as a month key; day 15 is a placeholder, not a reported sampling day.
- BDL is stored as a qualifier and numeric value left blank. It must not be converted to zero.
- This package is not a complete full-parameter dataset. No turbidity, COD, TDS or several other required fields were verified for Beas in the sources inspected. Those remain missing.
- Annual min/max records and monthly point records must not be naively merged into one row as if they were the same sampling event.
- This package excludes the previously generated synthetic dataset.
- Some HPSPCB PDF table OCR-extracted entries had ambiguous total-coliform tokens for certain stations; only stations with clear numeric sequences were included in the monthly extract.
- CPCB annual reports list both Himachal Pradesh and Punjab Beas stations; this package is restricted to Himachal Pradesh stations to match the Beas-Himachal project scope.

Use this dataset for source-grounded exploratory analysis and data integration tests. It is not a dense time-series dataset for forecasting. Model training should use a consistent observation grain and validated target labels.
