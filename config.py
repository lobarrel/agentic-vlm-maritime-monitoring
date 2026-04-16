import os

from ollama import Client

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.com")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
MODEL_NAME = os.environ.get("VLM_MODEL", "gemma4:31b")

STAC_API_URL = "https://earth-search.aws.element84.com/v1"
STAC_COLLECTION = "sentinel-2-l2a"

SEARCH_RADIUS_KM = 10
MAX_IMAGES = 4
MAX_CLOUD_COVER = 30  # percent

INVESTIGATOR_MAX_ITERATIONS = 10
# Longest side of JPEGs sent to the VLM (after cropping the AOI).  Higher =
# more detail and larger API payloads.  If Ollama Cloud returns “request body
# too large”, lower this or ``VLM_JPEG_QUALITY``.  Local Ollama often allows 4096+.
VLM_MAX_IMAGE_PX = int(os.environ.get("VLM_MAX_IMAGE_PX", "2048"))
VLM_JPEG_QUALITY = int(os.environ.get("VLM_JPEG_QUALITY", "90"))

OUTPUT_DIR = os.environ.get("VLM_OUTPUT_DIR", "./output")
TEMP_DIR = os.environ.get("VLM_TEMP_DIR", "./tmp_images")

_client: Client | None = None


def get_client(api_key: str | None = None, host: str | None = None) -> Client:
    """Return a configured Ollama Client, creating one if needed.

    When *api_key* or *host* are supplied they override the module-level
    defaults (useful for the GUI where the user enters a key at runtime).
    Passing new values resets the cached client.
    """
    global _client, OLLAMA_API_KEY, OLLAMA_HOST

    key = api_key or OLLAMA_API_KEY
    h = host or OLLAMA_HOST

    if api_key and api_key != OLLAMA_API_KEY:
        OLLAMA_API_KEY = api_key
        _client = None
    if host and host != OLLAMA_HOST:
        OLLAMA_HOST = host
        _client = None

    if _client is None:
        headers = {}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        _client = Client(host=h, headers=headers)

    return _client
