"""Write the lab's JSON artifacts without leaving partial files."""
import json
from pathlib import Path
from tempfile import NamedTemporaryFile


def write_json(path: str | Path, data: object) -> None:
    destination = Path(path)
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    file = NamedTemporaryFile(dir=destination.parent, delete=False)
    temporary = Path(file.name)
    try:
        with file:
            file.write(payload.encode("utf-8"))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
