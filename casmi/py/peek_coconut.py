import sys
path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    header = f.readline()
    print("HEADER:", header)
    for i in range(3):
        print(f"ROW{i}:", f.readline()[:500])
