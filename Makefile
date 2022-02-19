# === COMMANDS ================================================================
# Installs the Python dependencies
install:
	pip3 install -r requirements.txt

# Downloads the Python dependencies
download:
	pip3 download -r requirements.txt


# Make index notebook
index:
	python3 scripts/make_index.py
