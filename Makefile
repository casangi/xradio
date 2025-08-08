
# Format Python code using black
python-format:
	black --config pyproject.toml src/ tests/ docs/source/ scripts/


# Export JSON schemas
schema-export:
	@for schema in VisibilityXds SpectrumXds; do \
		PYTHONPATH=src python scripts/export_schema.py $$schema schemas/$$schema.json; \
	done
