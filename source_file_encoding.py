"""Shared deterministic Python source byte-encoding helpers."""

import codecs
import io
import tokenize


def detect_source_encoding(content: bytes) -> str:
    """Detect and validate the encoding declared by Python source bytes."""

    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(content).readline)
    except (SyntaxError, UnicodeDecodeError) as error:
        raise ValueError(
            f"Python source encoding could not be detected: {error}."
        ) from error
    try:
        codecs.lookup(encoding)
    except LookupError as error:
        raise ValueError(
            f"Python source declares unknown encoding {encoding!r}."
        ) from error
    return encoding


def canonical_encoding(encoding: str) -> str:
    """Return a stable codec name, treating UTF-8 with BOM as UTF-8."""

    canonical = codecs.lookup(encoding).name
    return "utf-8" if canonical == "utf-8-sig" else canonical


def candidate_source_encoding(updated_code: str) -> str:
    """Inspect the encoding declaration in a Unicode candidate source."""

    try:
        candidate_bytes = updated_code.encode("utf-8")
        return detect_source_encoding(candidate_bytes)
    except UnicodeEncodeError as error:
        raise ValueError("Candidate source could not be inspected as UTF-8.") from error


def normalized_newlines(content: str) -> str:
    """Normalize all supported source newline conventions to LF."""

    return content.replace("\r\n", "\n").replace("\r", "\n")


def detect_newline(content: str) -> str:
    """Return the sole newline convention used by source, rejecting mixtures."""

    crlf_count = content.count("\r\n")
    without_crlf = content.replace("\r\n", "")
    lf_count = without_crlf.count("\n")
    cr_count = without_crlf.count("\r")
    styles = sum(count > 0 for count in (crlf_count, lf_count, cr_count))
    if styles > 1:
        raise ValueError(
            "Target uses mixed newline conventions; apply was refused to avoid "
            "rewriting unrelated lines."
        )
    if crlf_count:
        return "\r\n"
    if cr_count:
        return "\r"
    return "\n"


def decode_python_source(content: bytes) -> str:
    """Decode exact Python source bytes using the source's declared encoding."""

    encoding = detect_source_encoding(content)
    try:
        return content.decode(encoding)
    except UnicodeDecodeError as error:
        raise ValueError(
            f"Python source could not be decoded safely as {encoding!r}."
        ) from error


def candidate_bytes_for_modify(
    updated_code: str,
    current_content: bytes,
) -> tuple[bytes, str]:
    """Encode a replacement using the original encoding and newline style."""

    encoding = detect_source_encoding(current_content)
    candidate_encoding = candidate_source_encoding(updated_code)
    if canonical_encoding(candidate_encoding) != canonical_encoding(encoding):
        raise ValueError(
            "Candidate source encoding declaration does not match the original "
            f"target encoding ({candidate_encoding!r} != {encoding!r})."
        )
    try:
        current_text = current_content.decode(encoding)
    except UnicodeDecodeError as error:
        raise ValueError(
            f"Original target could not be decoded safely as {encoding!r}."
        ) from error
    newline = detect_newline(current_text)
    normalized_candidate = normalized_newlines(updated_code)
    candidate_text = normalized_candidate.replace("\n", newline)
    try:
        return candidate_text.encode(encoding), current_text
    except UnicodeEncodeError as error:
        raise ValueError(
            f"Candidate source cannot be encoded safely as {encoding!r}."
        ) from error


def candidate_bytes_for_add(updated_code: str) -> bytes:
    """Encode a newly added Python file as UTF-8 with LF newlines."""

    candidate_encoding = candidate_source_encoding(updated_code)
    if canonical_encoding(candidate_encoding) != "utf-8":
        raise ValueError(
            "New Python source must use UTF-8; its encoding cookie declares "
            f"{candidate_encoding!r}."
        )
    normalized_candidate = normalized_newlines(updated_code)
    try:
        return normalized_candidate.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("New Python source could not be encoded as UTF-8.") from error
