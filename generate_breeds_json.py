import json
import wikipedia
import requests
from bs4 import BeautifulSoup

# Load your config
with open("./dog-breeds-multiclass-image-classification-with-vit/config.json", "r") as f:
    config = json.load(f)

id2label = config["id2label"]

def clean_name(name):
    return name.replace("_", " ").title()

def fetch_wikipedia_data(breed):
    try:
        page = wikipedia.page(breed)
        url = page.url

        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        info = {
            "origin": "Unknown",
            "life_expectancy": "Unknown",
            "weight": "Unknown",
            "height": "Unknown",
            "temperament": "Unknown",
            "bite_force": "Unknown"
        }

        # Extract infobox
        infobox = soup.find("table", {"class": "infobox"})

        if infobox:
            rows = infobox.find_all("tr")

            for row in rows:
                header = row.find("th")
                value = row.find("td")

                if header and value:
                    key = header.text.lower()
                    val = value.text.strip().replace("\n", " ")

                    if "origin" in key:
                        info["origin"] = val
                    elif "life span" in key:
                        info["life_expectancy"] = val
                    elif "weight" in key:
                        info["weight"] = val
                    elif "height" in key:
                        info["height"] = val
                    elif "temperament" in key:
                        info["temperament"] = val

        return info

    except Exception as e:
        print(f"Error fetching {breed}: {e}")
        return None


# Generate dataset
breeds_data = {}

for idx, raw_name in id2label.items():
    breed = clean_name(raw_name)
    print(f"Fetching: {breed}")

    data = fetch_wikipedia_data(breed)

    if data:
        breeds_data[breed] = data
    else:
        breeds_data[breed] = {
            "origin": "Unknown",
            "life_expectancy": "Unknown",
            "weight": "Unknown",
            "height": "Unknown",
            "temperament": "Unknown",
            "bite_force": "Unknown"
        }

# Save JSON
with open("breeds_info.json", "w") as f:
    json.dump(breeds_data, f, indent=2)

print("✅ breeds_info.json generated successfully!")