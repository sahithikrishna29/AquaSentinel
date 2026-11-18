AquaSentinel - Combined Beas River Water Quality Master Dataset

MAIN FILE
- beas_master_water_quality_long.csv

SUPPORT FILES
- beas_parameter_source_inventory.csv: parameter availability by uploaded source.
- beas_data_quality_flags.csv: invalid ranges and conflicts requiring review.
- dataset_summary.json: record and parameter summary.

INCLUDED
1) User-uploaded RiverWaterQualityOverYears_PreprocessedData.csv, filtered to Monitoring Location containing BEAS. Annual min/max records for nine parameters, years 2012, 2013, 2014, 2017-2023. The uploaded file does not include original agency/source URL. Treat its provenance as unverified until traced.
2) Uploaded CPCB chemical CSV, filtered to Station containing BEAS; only non-null numeric values are included.
3) Uploaded CPCB biological CSV, filtered to Station containing BEAS; only non-null numeric values are included.
4) Earlier HPSPCB monthly observations (2021-22) from the AquaSentinel source ZIP. Date day 15 is a month placeholder, not a reported sampling day.
5) Earlier CPCB annual PDF transcription for 2022-2023. Kept separately from the preprocessed annual source. Disagreements are flagged, not silently reconciled.

EXCLUDED
- Uploaded CPCB physical CSV: Beas station rows exist, but no numeric Beas physical values were found.
- Both uploaded CWC physical CSVs: no Beas references were found anywhere in the files, so non-Beas records were not mixed into this Beas dataset.

IMPORTANT
- Long format: each row represents one parameter record.
- record_type separates annual_min_max_range, monthly_observation, and sample_observation. Do not treat them as a single time grain.
- Annual min/max are ranges, not point samples.
- Missing values remain missing; BDL is not converted to zero.
- Station names are preserved; no unverified alias matching was performed.
- The uploaded historical preprocessed file has min>max anomalies and overlaps/conflicts with the earlier CPCB transcription. See quality flags.
- Its Contaminated label is retained as supplied but its definition is not established; do not use as a validated target until verified.
- No synthetic values were created.
- Uploaded CPCB files do not contain source URLs; provenance should be added from original catalog records before formal publication.
