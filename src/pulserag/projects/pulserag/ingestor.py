"""Loading and Labelling clinical source material.

Three source types feed the index, in decreasing order of authority:
1. Guideline PDFs (data/guidelines/*.pdf)
2. PubMed abstracts
3. Bootstrap documents
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from llama_index.core.schema import Document

from pulserag.core.project import DocumentIngestor

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["PulseRAGIngestor"]
_DRUG_LABEL_HINTS = ("fda", "dailymed", "label")
_PARSING_INSTRUCTION = "Clinical guideline or drug label containing evidence tables, dose recommendations, warnings, and a section hierarchy. Preserve tables and headings."


class PulseRAGIngestor(DocumentIngestor):
    """Builds the clinical corpus from PDFs, PubMed and bootstrap seeds."""

    pubmed_queries: tuple[str, ...] = (
        "type 2 diabetes treatment guideline",
        "hypertension management guideline",
    )
    bootstrap_documents: tuple[dict[str, str], ...] = (
        {
            "title": "Type 2 diabetes first-line therapy overview",
            "text": (
                "Type 2 diabetes management usually begins with lifestyle modification, "
                "glycemic monitoring, and metformin when there are no contraindications. "
                "Guidelines commonly describe individualized escalation to GLP-1 receptor "
                "agonists, SGLT2 inhibitors, or insulin based on comorbidities, kidney "
                "function, cardiovascular risk, and glycemic control."
            ),
        },
        {
            "title": "Type 1 diabetes treatment overview",
            "text": (
                "Type 1 diabetes requires insulin replacement therapy rather than oral "
                "first-line agents. Guideline-based care typically includes basal-bolus "
                "or pump-based insulin delivery, glucose monitoring, hypoglycemia "
                "education, and individualized nutrition planning."
            ),
        },
    )

    # document ingestor
    def load_and_parse(self) -> list[Document]:
        """Load every enabled source and return the combined document list."""
        documents: list[Document] = []
        documents.extend(self._load_guideline_pdf())
        documents.extend(self._load_pubmed_abstracts())
        documents.extend(self._load_bootstrap_documents())
        logger.info("Ingestion complete", extra={"documents": len(documents)})
        return documents

    def enrich_metadata(self, docs: list[Document]) -> list[Document]:
        """Metadata of Documents."""
        for doc in docs:
            metadata: dict[str, Any] = dict(getattr(doc, "metadata", None) or {})
            source_file = str(metadata.get("source_file", "")).lower()

            if metadata.get("source") == "pubmed":
                metadata["source_org"] = "PubMed"
            elif any(hint in source_file for hint in _DRUG_LABEL_HINTS):
                metadata["source_org"] = "FDA"
            else:
                metadata.setdefault("source_org", "WHO")
        return docs

    # source loader
    def _load_guideline_pdf(self) -> list[Document]:
        limit = self.settings.max_guideline_files
        if limit == 0:
            logger.info("Guideline PDF ingestion disabled (MAX_GUIDELINE_FILES=0).")
            return []

        pdf_paths = sorted(self.config.guideline_dir.glob("*.pdf"))[:limit]
        if not pdf_paths:
            logger.info(
                "No guideline PDFs found.",
                extra={"directory": str(self.config.guidelines_dir)},
            )
            return []

        api_key = self.settings.secret(self.settings.llama_cloud_api_key)
        if not api_key:
            raise RuntimeError("LLAMA_CLOUD_API_KEY is not set pdf cannot be parsed.")

        from llama_parse import LlamaParse

        parser = LlamaParse(api_key=api_key, parsing_instruction=_PARSING_INSTRUCTION)

        documents: list[Document] = []
        for pdf_path in pdf_paths:
            logger.info("Parsing guideline PDF", extra={"file": pdf_path.name})
            documents.extend(self._parse_one_pdf(parser, pdf_path))
        logger.info(
            "Parsed guideline PDFs",
            extra={"files": len(pdf_paths), "documents": len(documents)},
        )
        return documents

    def _parse_one_pdf(self, parser: Any, pdf_path: Path) -> list[Document]:
        """Parse a single PDF and provide metadata on its documents."""
        try:
            parsed = parser.load_data(str(pdf_path))
        except Exception:
            logger.exception(
                "Failed to parse guideline PDF", extra={"file": pdf_path.name}
            )
            return []

        documents = []

        for doc in parsed:
            metadata = dict(getattr(doc, "metadata", None) or {})
            metadata.update({"source_file": pdf_path.name, "source": "guideline_pdf"})
            doc.metadata = metadata
            documents.append(doc)
        return documents

    def _load_pubmed_abstracts(self) -> list[Document]:
        """Fetch abstracts for the first standing queries."""
        if not self.settings.pubmed_enabled or self.settings.pubmed_query_limit == 0:
            logger.info("PubMed ingestion disabled.")
            return []

        from llama_index.readers.papers import PubmedReader

        reader = PubmedReader()
        queries = self.pubmed_queries[: self.settings.pubmed_query_limit]
        documents: list[Document] = []

        for query in queries:
            try:
                abstracts = reader.load_data(
                    search_query=query, max_results=self.settings.pubmed_max_results
                )
            except Exception:
                logger.warning(
                    "PubMed query failed; continuing without it.",
                    extra={"query": query},
                    exc_info=True,
                )
                continue

            for doc in abstracts:
                metadata = dict(getattr(doc, "metadata", None) or {})
                metadata.update({"source": "pubmed", "query": query})
                doc.metadata = metadata
                documents.append(doc)
        logger.info(
            "Fetched PubMed abstracts",
            extra={"queries": len(queries), "documents": len(documents)},
        )
        return documents

    def _load_bootstrap_documents(self) -> list[Document]:
        """Return handwritten documents."""
        if not self.settings.include_bootstrap_documents:
            return []
        return [
            Document(
                text=item["text"],
                metadata={
                    "source": "bootstrap",
                    "source_file": "bootstrap_seed",
                    "title": item["title"],
                    "source_org": "Bootstrap",
                },
            )
            for item in self.bootstrap_documents
        ]
