import json

# Load once
with open("breeds_info.json", "r") as f:
    BREEDS = json.load(f)


def normalize_name(name):
    return name.strip().lower()


def get_breed_info(breed_name):
    LABEL_MAP = {
    "Pekinese": "Pekinese",  # if you keep same
    "Wire": "Wire",
    "Soft": "Soft",
    "Black": "Black",
    "Shih": "Shih",
    "German Short": "German Short",
    }
    breed_name = LABEL_MAP.get(breed_name, breed_name)
    try:
        # Normalize input
        input_name = normalize_name(breed_name)

        # Search in JSON (case-insensitive)
        for key in BREEDS:
            if normalize_name(key) == input_name:
                data = BREEDS[key]

                return {
                    "name": key,
                    "origin": data.get("origin", "N/A"),
                    "life_span": data.get("life_expectancy", "N/A"),
                    "weight": data.get("weight", "N/A"),
                    "height": data.get("height", "N/A"),
                    "temperament": data.get("temperament", "N/A"),
                    "bred_for": data.get("bite_force", "N/A")
                }

        return {"error": "Breed not found"}

    except Exception as e:
        return {"error": str(e)}