import json
import sys

def validate(path):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                assert "text" in obj and isinstance(obj["text"], str)
                assert "label" in obj and isinstance(obj["label"], int)
                assert obj["text"].strip() != ""
            except Exception as e:
                raise ValueError(f"Invalid line {i}: {e}")

    print(f"[OK] {path} is valid")

if __name__ == "__main__":
    validate(sys.argv[1])
