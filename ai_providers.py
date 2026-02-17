import os
import requests
import json


AI_PROVIDER = os.environ.get('AI_PROVIDER', 'ollama').lower()
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://127.0.0.1:11434').rstrip('/')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'llama3.1:8b')
OLLAMA_TIMEOUT = int(os.environ.get('OLLAMA_TIMEOUT', '60'))


def ollama_generate(prompt, system=None):
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )

        if resp.status_code != 200:
            return None, f"Ollama HTTP {resp.status_code}: {resp.text[:300]}"

        data = resp.json()
        result_text = data.get("response")
        if result_text is None:
            return None, f"Ollama response missing 'response' key. Keys: {list(data.keys())}"

        return result_text, None

    except requests.exceptions.Timeout:
        return None, f"Ollama timeout after {OLLAMA_TIMEOUT}s"
    except requests.exceptions.ConnectionError:
        return None, f"Cannot connect to Ollama at {OLLAMA_BASE_URL}"
    except json.JSONDecodeError as e:
        return None, f"Ollama returned invalid JSON: {str(e)}"
    except Exception as e:
        return None, f"Ollama error: {str(e)}"


def get_provider_info():
    return {
        "provider": AI_PROVIDER,
        "model": OLLAMA_MODEL if AI_PROVIDER == "ollama" else os.environ.get('AI_MODEL', 'gpt-4o-mini'),
        "base_url": OLLAMA_BASE_URL if AI_PROVIDER == "ollama" else "(replit-managed)",
    }
