import os
import httpx

PLANTNET_API_KEY = os.getenv("PLANTNET_API_KEY", "")
PLANTNET_URL = "https://my-api.plantnet.org/v2/identify/all"
PLANTNET_TIMEOUT = 15.0  


async def query_plantnet(image_bytes: bytes, filename: str) -> dict | None:
    if not PLANTNET_API_KEY:
        print("[plantnet] PLANTNET_API_KEY not set — skipping fallback")
        return None

    try:
        async with httpx.AsyncClient(timeout=PLANTNET_TIMEOUT) as client:
            resp = await client.post(
                PLANTNET_URL,
                params={"api-key": PLANTNET_API_KEY, "lang": "en"},
                files={"images": (filename, image_bytes, "image/jpeg")},
                data={"organs": ["auto"]},
            )

        if resp.status_code == 404:
            return None   # PlantNet found nothing

        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            return None

        top = results[0]
        species = top.get("species", {})

        return {
            "species": species.get("commonNames", [None])[0]
                       or species.get("scientificNameWithoutAuthor", "Unknown"),
            "scientific_name": species.get("scientificNameWithoutAuthor"),
            "score": round(top.get("score", 0.0), 4),
            "family": species.get("family", {}).get("scientificNameWithoutAuthor"),
            "genus": species.get("genus", {}).get("scientificNameWithoutAuthor"),
        }

    except httpx.HTTPStatusError as exc:
        print(f"[plantnet] HTTP error: {exc.response.status_code}")
        return None
    except Exception as exc:
        print(f"[plantnet] Error: {exc}")
        return None