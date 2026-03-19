import requests

breed_cache = {}

# 🔥 Fix incorrect labels from model
BREED_FIX = {
    "German Short": "German Shorthaired Pointer",
    "Shih": "Shih Tzu",
    "Flat": "Flat-Coated Retriever",
    "Wire": "Wire Fox Terrier",
    "Soft": "Soft-Coated Wheaten Terrier",
    "Black": "Black Labrador Retriever"
}


def normalize_name(name):
    return BREED_FIX.get(name, name)


def safe_get(value, suffix=""):
    if value is None or value == "":
        return "Unknown"
    return str(value) + suffix


def get_breed_info(breed_name):
    # Normalize name
    breed_name = normalize_name(breed_name)

    # Cache check
    if breed_name in breed_cache:
        return breed_cache[breed_name]

    try:
        url = "https://api.thedogapi.com/v1/breeds/search"

        # 🔥 Try full name first
        response = requests.get(url, params={"q": breed_name})
        data = response.json()

        # 🔥 If not found → try first word
        if len(data) == 0:
            first_word = breed_name.split(" ")[0]
            response = requests.get(url, params={"q": first_word})
            data = response.json()

        # 🔥 STILL NOT FOUND → return fallback (IMPORTANT FIX)
        if len(data) == 0:
            result = {
                "name": breed_name,
                "origin": "Unknown",
                "life_span": "Unknown",
                "weight": "Unknown",
                "height": "Unknown",
                "temperament": "Unknown",
                "bred_for": "Unknown"
            }

            breed_cache[breed_name] = result
            return result

        breed = data[0]

        result = {
            "name": safe_get(breed.get("name")),
            "origin": safe_get(breed.get("origin")),
            "life_span": safe_get(breed.get("life_span")),
            "temperament": safe_get(breed.get("temperament")),
            "weight": safe_get(breed.get("weight", {}).get("metric"), " kg"),
            "height": safe_get(breed.get("height", {}).get("metric"), " cm"),
            "bred_for": safe_get(breed.get("bred_for"))
        }

        breed_cache[breed_name] = result
        return result

    except Exception as e:
        # 🔥 NEVER FAIL UI
        return {
            "name": breed_name,
            "origin": "Unknown",
            "life_span": "Unknown",
            "weight": "Unknown",
            "height": "Unknown",
            "temperament": "Unknown",
            "bred_for": "Unknown"
        }