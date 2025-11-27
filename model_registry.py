import json
from pathlib import Path
from typing import Any, Dict

def load_model_registry() -> Dict[str, Dict[str, Any]]:
    """
    Loads the model registry from the models.json file.
    """
    registry_path = Path(__file__).parent / "models.json"
    with registry_path.open() as f:
        return json.load(f)

MODEL_REGISTRY = load_model_registry()