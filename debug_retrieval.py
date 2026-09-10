import requests

QUESTION = " What were Teekay Corporation's Realized losses in 2019, 2018 and 2017 respectively?"

response = requests.post(
    "http://localhost:8000/search",
    json={
        "query": QUESTION,
        "top_k": 10
    }
)

data = response.json()

print("\n" + "=" * 80)
print("QUESTION")
print("=" * 80)
print(QUESTION)

for result in data.get("candidates", []):
    chunk = result["chunk"]

    print("\n" + "-" * 80)
    print(f"RANK: {result.get('rank')}")
    print(f"SCORE: {result.get('score')}")
    print(f"DOCUMENT: {chunk.get('document_id')}")
    print(f"PAGE: {chunk.get('page')}")
    print(f"SECTION: {chunk.get('section')}")
    print(f"TYPE: {chunk.get('content_type')}")

    content = chunk.get("content")

    if chunk.get("content_type") == "table":
        print("\nTABLE:")

        headers = content.get("headers", [])
        rows = content.get("rows", [])

        print("HEADERS:")
        print(" | ".join(str(x) for x in headers))

        print("\nROWS:")
        for row in rows:
            print(" | ".join(str(x) for x in row))

    else:
        print("\nCONTENT:")
        print(content)

print("\n" + "=" * 80)
