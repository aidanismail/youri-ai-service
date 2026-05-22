import json

input_file = "final_recipes_mongodb_v_final.json"

try:
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("JSON valid!")
    print(type(data))

except json.JSONDecodeError as e:
    print("JSON tidak valid!")
    print(f"Pesan error : {e.msg}")
    print(f"Baris       : {e.lineno}")
    print(f"Kolom       : {e.colno}")
    print(f"Karakter    : {e.pos}")

    print("\nArea sekitar error:\n")

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    start = max(e.lineno - 5, 0)
    end = min(e.lineno + 5, len(lines))

    for i in range(start, end):
        marker = ">>>" if i + 1 == e.lineno else "   "
        print(f"{marker} {i + 1}: {lines[i].rstrip()}")