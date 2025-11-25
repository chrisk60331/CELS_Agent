
from toon import encode as tooncode
from strands import tool
import json 
@tool
def file_read(path: str):
    with open(path, 'r') as file_obj:
        raw = json.loads(file_obj.read())
        print(f"raw: {raw}")
        tooncoded = tooncode(raw)
        print(f"tooncoded: {tooncoded}")

        return tooncoded

def file_write():
    pass
