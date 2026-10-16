PY ?= .venv/Scripts/python

data-raw:
	$(PY) -m src.ingest.download_trips --months 12
	$(PY) -m src.ingest.weather --start 2025-10-01 --end 2026-10-01

collect:
	$(PY) -m src.ingest.gbfs_collector

data:
	$(PY) -m src.ingest.station_hour
	$(PY) -m pytest -q tests
