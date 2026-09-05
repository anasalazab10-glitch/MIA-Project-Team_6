from validator import validate_answer


def run_test(name, payload, expected_valid):
    result = validate_answer(payload)
    actual_valid = result[0]

    status = "PASS" if actual_valid == expected_valid else "FAIL"

    print(f"{status}: {name}")
    print(f"      Result: {result}")
    print()


# VALID CASES
print("VALID CASES")
print()


run_test(
    "Valid direct answer",
    {
        "answer_type": "direct",
        "evidence": [
            {
                "document_id": "doc1",
                "page": 1
            }
        ],
        "params": {
            "value": "Alexandria"
        }
    },
    True
)


run_test(
    "Valid calculated answer",
    {
        "answer_type": "calculated",
        "evidence": [
            {
                "document_id": "doc1",
                "page": 2
            }
        ],
        "params": {
            "value": 150.5,
            "formula": "100 + 50.5"
        }
    },
    True
)


run_test(
    "Valid multi-span answer",
    {
        "answer_type": "multi_span",
        "evidence": [
            {
                "document_id": "doc1",
                "page": 3
            },
            {
                "document_id": "doc2",
                "page": 5
            }
        ],
        "params": {
            "values": [
                "Value A",
                "Value B"
            ]
        }
    },
    True
)

run_test(
    "Valid insufficient evidence answer",
    {
        "answer_type": "insufficient_evidence",
        "evidence": [],
        "params": {
            "reason": "The provided documents do not contain enough information."
        }
    },
    True
)


# ANSWER TYPE TESTS

print("ANSWER TYPE TESTS")
print()


run_test(
    "Missing answer_type",
    {
        "evidence": [
            {
                "document_id": "doc1",
                "page": 1
            }
        ],
        "params": {
            "value": "test"
        }
    },
    False
)


run_test(
    "Invalid answer_type",
    {
        "answer_type": "unknown_type",
        "evidence": [
            {
                "document_id": "doc1",
                "page": 1
            }
        ],
        "params": {
            "value": "test"
        }
    },
    False
)


# EVIDENCE TESTS


print("EVIDENCE TESTS")

print()


run_test(
    "Missing evidence",
    {
        "answer_type": "direct",
        "params": {
            "value": "test"
        }
    },
    False
)


run_test(
    "Evidence is not a list",
    {
        "answer_type": "direct",
        "evidence": "doc1",
        "params": {
            "value": "test"
        }
    },
    False
)


run_test(
    "Empty evidence for direct answer",
    {
        "answer_type": "direct",
        "evidence": [],
        "params": {
            "value": "test"
        }
    },
    False
)


run_test(
    "Empty evidence for calculated answer",
    {
        "answer_type": "calculated",
        "evidence": [],
        "params": {
            "value": 100,
            "formula": "50 + 50"
        }
    },
    False
)


run_test(
    "Empty evidence for multi-span answer",
    {
        "answer_type": "multi_span",
        "evidence": [],
        "params": {
            "values": ["A", "B"]
        }
    },
    False
)


run_test(
    "Empty evidence allowed for insufficient evidence",
    {
        "answer_type": "insufficient_evidence",
        "evidence": [],
        "params": {
            "reason": "Not enough information."
        }
    },
    True
)


run_test(
    "Evidence item is not an object",
    {
        "answer_type": "direct",
        "evidence": [
            "invalid citation"
        ],
        "params": {
            "value": "test"
        }
    },
    False
)


run_test(
    "Evidence missing document_id",
    {
        "answer_type": "direct",
        "evidence": [
            {
                "page": 1
            }
        ],
        "params": {
            "value": "test"
        }
    },
    False
)


run_test(
    "Evidence missing page",
    {
        "answer_type": "direct",
        "evidence": [
            {
                "document_id": "doc1"
            }
        ],
        "params": {
            "value": "test"
        }
    },
    False
)


# PARAMS TESTS

print("PARAMS TESTS")

print()


run_test(
    "Missing params",
    {
        "answer_type": "direct",
        "evidence": [
            {
                "document_id": "doc1",
                "page": 1
            }
        ]
    },
    False
)


run_test(
    "Params is not an object",
    {
        "answer_type": "direct",
        "evidence": [
            {
                "document_id": "doc1",
                "page": 1
            }
        ],
        "params": "invalid"
    },
    False
)



# DIRECT ANSWER TESTS

print("DIRECT ANSWER TESTS")

print()


run_test(
    "Direct answer missing value",
    {
        "answer_type": "direct",
        "evidence": [
            {
                "document_id": "doc1",
                "page": 1
            }
        ],
        "params": {}
    },
    False
)



# CALCULATED ANSWER TESTS


print("CALCULATED ANSWER TESTS")

print()


run_test(
    "Calculated missing value",
    {
        "answer_type": "calculated",
        "evidence": [
            {
                "document_id": "doc1",
                "page": 1
            }
        ],
        "params": {
            "formula": "100 + 50"
        }
    },
    False
)


run_test(
    "Calculated missing formula",
    {
        "answer_type": "calculated",
        "evidence": [
            {
                "document_id": "doc1",
                "page": 1
            }
        ],
        "params": {
            "value": 150
        }
    },
    False
)


run_test(
    "Calculated empty formula",
    {
        "answer_type": "calculated",
        "evidence": [
            {
                "document_id": "doc1",
                "page": 1
            }
        ],
        "params": {
            "value": 150,
            "formula": ""
        }
    },
    False
)


run_test(
    "Calculated value is not a number",
    {
        "answer_type": "calculated",
        "evidence": [
            {
                "document_id": "doc1",
                "page": 1
            }
        ],
        "params": {
            "value": "150",
            "formula": "100 + 50"
        }
    },
    False
)


# MULTI-SPAN ANSWER TESTS

print("MULTI-SPAN ANSWER TESTS")

print()


run_test(
    "Multi-span missing values",
    {
        "answer_type": "multi_span",
        "evidence": [
            {
                "document_id": "doc1",
                "page": 1
            }
        ],
        "params": {}
    },
    False
)


run_test(
    "Multi-span values is not a list",
    {
        "answer_type": "multi_span",
        "evidence": [
            {
                "document_id": "doc1",
                "page": 1
            }
        ],
        "params": {
            "values": "not a list"
        }
    },
    False
)


run_test(
    "Multi-span values is empty",
    {
        "answer_type": "multi_span",
        "evidence": [
            {
                "document_id": "doc1",
                "page": 1
            }
        ],
        "params": {
            "values": []
        }
    },
    False
)


# INSUFFICIENT EVIDENCE TESTS

print("INSUFFICIENT EVIDENCE TESTS")

print()


run_test(
    "Insufficient evidence missing reason",
    {
        "answer_type": "insufficient_evidence",
        "evidence": [],
        "params": {}
    },
    False
)


run_test(
    "Insufficient evidence empty reason",
    {
        "answer_type": "insufficient_evidence",
        "evidence": [],
        "params": {
            "reason": ""
        }
    },
    False
)


print("ALL TESTS COMPLETED")
