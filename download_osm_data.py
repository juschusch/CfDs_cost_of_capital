import os
import requests
from pathlib import Path

# Configuration
OSM_VERSION = 0.6
ZENODO_ID = "14144752"
BASE_URL = f"https://zenodo.org/records/{ZENODO_ID}/files/"
OUTPUT_DIR = Path(f"data/osm-prebuilt/{OSM_VERSION}")

FILES = [
    "buses.csv",
    "converters.csv",
    "lines.csv",
    "links.csv",
    "transformers.csv",
    "map.html"
]

def download_file(filename):
    url = BASE_URL + filename
    output_path = OUTPUT_DIR / filename
    
    if output_path.exists():
        print(f"File already exists: {output_path}")
        return

    print(f"Downloading {filename} from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

if __name__ == "__main__":
    print(f"Downloading OSM prebuilt data (v{OSM_VERSION})...")
    for file in FILES:
        download_file(file)
    print("Done.")
