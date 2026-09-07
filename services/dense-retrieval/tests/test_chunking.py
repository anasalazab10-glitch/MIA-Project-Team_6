
import json
from pathlib import Path

from src.chunking import create_chunks
from src.schemas import ContentType


def test_create_chunks_from_mock_data():
    # Load the existing processed mock data.
    data_path = Path("data/mock_processed.json")

    with data_path.open("r", encoding="utf-8") as file:
        elements = json.load(file)

    chunks = create_chunks(elements)
    assert chunks, "No chunks were created."

    # Every chunk must have an ID.
    assert all(chunk.chunk_id for chunk in chunks)

    # Every chunk must have a document ID.
    assert all(chunk.document_id for chunk in chunks)

    # Every chunk must have at least one page.
    assert all(chunk.page for chunk in chunks)

    # Every chunk must have a content type.
    assert all(chunk.content_type for chunk in chunks)

    # Document checks
    document_ids = {chunk.document_id for chunk in chunks}

    assert "doc1" in document_ids
    assert "doc2" in document_ids

    # Chunk IDs should be unique.
    chunk_ids = [chunk.chunk_id for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))

    # Section-aware checks
    text_chunks = [
        chunk
        for chunk in chunks
        if chunk.content_type == ContentType.TEXT
    ]

    sections = {chunk.section for chunk in text_chunks}

    expected_sections = {
        "Company Overview",
        "Financial Results",
        "Operating Expenses",
        "Business Performance",
        "Revenue by Region",
        "Cash Flow",
        "Outlook",
    }

    assert expected_sections.issubset(sections)

    # Text from a section must not be assigned to another section.
    for chunk in text_chunks:
        assert chunk.section in expected_sections

    # Table-aware checks

    table_chunks = [
        chunk
        for chunk in chunks
        if chunk.content_type == ContentType.TABLE
    ]

    # There are 4 tables in mock_processed.json.
    assert len(table_chunks) == 4

    # Check that every table remains structured.
    for chunk in table_chunks:
        assert chunk.content.headers
        assert chunk.content.rows

    # Check expected table sections.
    table_sections = {
        chunk.section
        for chunk in table_chunks
    }

    assert table_sections == {
        "Financial Results",
        "Operating Expenses",
        "Revenue by Region",
        "Cash Flow",
    }

    # Specific table checks
 
    financial_table = next(
        chunk
        for chunk in table_chunks
        if chunk.section == "Financial Results"
    )

    assert financial_table.content.headers == [
        "Year",
        "Revenue",
        "Operating Income",
        "Net Profit",
    ]

    assert financial_table.content.rows[-1] == [
        "2024",
        "$120M",
        "$19M",
        "$13M",
    ]

    regional_table = next(
        chunk
        for chunk in table_chunks
        if chunk.section == "Revenue by Region"
    )

    assert regional_table.content.headers == [
        "Region",
        "2023 Revenue",
        "2024 Revenue",
        "Growth Rate",
    ]


    # Page checks

    # All mock tables are on a single page.
    for chunk in table_chunks:
        assert len(chunk.page) == 1

    # The page information should be preserved.
    assert financial_table.page == [2]

    # Metadata checks

    for chunk in chunks:
        assert "source_element_ids" in chunk.metadata
        assert "source_pages" in chunk.metadata

    # Every table should preserve its source element.
    for chunk in table_chunks:
        assert len(chunk.metadata["source_element_ids"]) == 1

  
    # Content checks

    company_chunk = next(
        chunk
        for chunk in text_chunks
        if chunk.section == "Company Overview"
    )

    assert "Northbridge Technologies" in company_chunk.content

    cash_flow_chunk = next(
        chunk
        for chunk in text_chunks
        if chunk.section == "Cash Flow"
    )

    assert "$17 million" in cash_flow_chunk.content

    outlook_chunk = next(
        chunk
        for chunk in text_chunks
        if chunk.section == "Outlook"
    )

    assert "$135 million" in outlook_chunk.content

  
    print("\n========== CHUNKING TEST ==========")
    print(f"Input elements : {len(elements)}")
    print(f"Output chunks  : {len(chunks)}")
    print(f"Text chunks    : {len(text_chunks)}")
    print(f"Table chunks   : {len(table_chunks)}")

    for chunk in chunks:
        print("\n----------------------------")
        print(f"ID       : {chunk.chunk_id}")
        print(f"Document : {chunk.document_id}")
        print(f"Pages    : {chunk.page}")
        print(f"Section  : {chunk.section}")
        print(f"Type     : {chunk.content_type}")
        print(f"Content  : {chunk.to_text()[:200]}")

    print("\nAll chunking tests passed!")

if __name__ == "__main__":
    test_create_chunks_from_mock_data()