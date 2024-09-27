
# Format Python code using black
python-format:
	black --config pyproject.toml src/ tests/ docs/source/
