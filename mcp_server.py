"""
NotebookLM MCP Server
=====================
A Model Context Protocol (MCP) server that exposes Google NotebookLM
operations as tools. Wraps the notebooklm-py async Python API.

Usage:
    python mcp_server.py
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("notebooklm-mcp")

# ---------------------------------------------------------------------------
# Lifespan: open / close the NotebookLM client once
# ---------------------------------------------------------------------------
_client = None


async def _get_client():
    """Lazy-initialize the NotebookLM client."""
    global _client
    if _client is None:
        from notebooklm import NotebookLMClient

        _client = await NotebookLMClient.from_storage()
        await _client.__aenter__()
        logger.info("NotebookLM client connected")
    return _client


def _serialize(obj: Any) -> Any:
    """Convert dataclass / datetime objects to JSON-safe dicts."""
    import dataclasses
    from datetime import datetime

    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _serialize(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return str(obj)


# ---------------------------------------------------------------------------
# Create the MCP server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "NotebookLM",
    instructions="MCP server for Google NotebookLM – notebooks, sources, chat, artifacts, research, notes, and sharing.",
)


# ===================================================================
#  NOTEBOOK TOOLS
# ===================================================================

@mcp.tool()
async def list_notebooks() -> str:
    """List all NotebookLM notebooks. Returns JSON array of notebooks with id, title, sources_count, etc."""
    client = await _get_client()
    notebooks = await client.notebooks.list()
    return json.dumps(_serialize(notebooks), indent=2)


@mcp.tool()
async def create_notebook(title: str) -> str:
    """Create a new NotebookLM notebook.

    Args:
        title: The title for the new notebook.
    """
    client = await _get_client()
    nb = await client.notebooks.create(title)
    return json.dumps(_serialize(nb), indent=2)


@mcp.tool()
async def get_notebook(notebook_id: str) -> str:
    """Get details of a specific notebook.

    Args:
        notebook_id: The notebook ID.
    """
    client = await _get_client()
    nb = await client.notebooks.get(notebook_id)
    return json.dumps(_serialize(nb), indent=2)


@mcp.tool()
async def delete_notebook(notebook_id: str) -> str:
    """Delete a notebook.

    Args:
        notebook_id: The notebook ID to delete.
    """
    client = await _get_client()
    result = await client.notebooks.delete(notebook_id)
    return json.dumps({"deleted": result, "notebook_id": notebook_id})


@mcp.tool()
async def rename_notebook(notebook_id: str, new_title: str) -> str:
    """Rename a notebook.

    Args:
        notebook_id: The notebook ID.
        new_title: The new title.
    """
    client = await _get_client()
    nb = await client.notebooks.rename(notebook_id, new_title)
    return json.dumps(_serialize(nb), indent=2)


@mcp.tool()
async def get_notebook_summary(notebook_id: str) -> str:
    """Get an AI-generated summary and suggested topics for a notebook.

    Args:
        notebook_id: The notebook ID.
    """
    client = await _get_client()
    desc = await client.notebooks.get_description(notebook_id)
    return json.dumps(_serialize(desc), indent=2)


# ===================================================================
#  SOURCE TOOLS
# ===================================================================

@mcp.tool()
async def list_sources(notebook_id: str) -> str:
    """List all sources in a notebook.

    Args:
        notebook_id: The notebook ID.
    """
    client = await _get_client()
    sources = await client.sources.list(notebook_id)
    return json.dumps(_serialize(sources), indent=2)


@mcp.tool()
async def add_source_url(notebook_id: str, url: str) -> str:
    """Add a URL (web page) as a source to a notebook.

    Args:
        notebook_id: The notebook ID.
        url: The URL to add.
    """
    client = await _get_client()
    source = await client.sources.add_url(notebook_id, url)
    return json.dumps(_serialize(source), indent=2)


@mcp.tool()
async def add_source_youtube(notebook_id: str, url: str) -> str:
    """Add a YouTube video as a source to a notebook.

    Args:
        notebook_id: The notebook ID.
        url: The YouTube URL.
    """
    client = await _get_client()
    source = await client.sources.add_youtube(notebook_id, url)
    return json.dumps(_serialize(source), indent=2)


@mcp.tool()
async def add_source_text(notebook_id: str, title: str, content: str) -> str:
    """Add pasted text as a source to a notebook.

    Args:
        notebook_id: The notebook ID.
        title: Title for the text source.
        content: The text content.
    """
    client = await _get_client()
    source = await client.sources.add_text(notebook_id, title, content)
    return json.dumps(_serialize(source), indent=2)


@mcp.tool()
async def add_source_file(notebook_id: str, file_path: str) -> str:
    """Upload a file (PDF, text, Markdown, Word, audio, video, images) as a source.

    Args:
        notebook_id: The notebook ID.
        file_path: Local file path to upload.
    """
    client = await _get_client()
    source = await client.sources.add_file(notebook_id, Path(file_path))
    return json.dumps(_serialize(source), indent=2)


@mcp.tool()
async def get_source_fulltext(notebook_id: str, source_id: str) -> str:
    """Get the full indexed text content of a source.

    Args:
        notebook_id: The notebook ID.
        source_id: The source ID.
    """
    client = await _get_client()
    fulltext = await client.sources.get_fulltext(notebook_id, source_id)
    return json.dumps(_serialize(fulltext), indent=2)


@mcp.tool()
async def get_source_guide(notebook_id: str, source_id: str) -> str:
    """Get AI-generated summary and keywords for a source.

    Args:
        notebook_id: The notebook ID.
        source_id: The source ID.
    """
    client = await _get_client()
    guide = await client.sources.get_guide(notebook_id, source_id)
    return json.dumps(_serialize(guide), indent=2)


@mcp.tool()
async def delete_source(notebook_id: str, source_id: str) -> str:
    """Delete a source from a notebook.

    Args:
        notebook_id: The notebook ID.
        source_id: The source ID to delete.
    """
    client = await _get_client()
    result = await client.sources.delete(notebook_id, source_id)
    return json.dumps({"deleted": result, "source_id": source_id})


@mcp.tool()
async def refresh_source(notebook_id: str, source_id: str) -> str:
    """Refresh a URL or Drive source to re-fetch its content.

    Args:
        notebook_id: The notebook ID.
        source_id: The source ID to refresh.
    """
    client = await _get_client()
    result = await client.sources.refresh(notebook_id, source_id)
    return json.dumps({"refreshed": result, "source_id": source_id})


# ===================================================================
#  CHAT TOOLS
# ===================================================================

@mcp.tool()
async def ask_notebook(
    notebook_id: str,
    question: str,
    source_ids: str = "",
    conversation_id: str = "",
) -> str:
    """Ask a question about the notebook's sources.

    Args:
        notebook_id: The notebook ID.
        question: The question to ask.
        source_ids: Optional comma-separated source IDs to limit the query to specific sources.
        conversation_id: Optional conversation ID to continue an existing conversation.
    """
    client = await _get_client()
    src_list = [s.strip() for s in source_ids.split(",") if s.strip()] or None
    conv_id = conversation_id if conversation_id else None
    result = await client.chat.ask(notebook_id, question, source_ids=src_list, conversation_id=conv_id)
    return json.dumps(_serialize(result), indent=2)


@mcp.tool()
async def get_chat_history(notebook_id: str) -> str:
    """Get the chat conversation history for a notebook.

    Args:
        notebook_id: The notebook ID.
    """
    client = await _get_client()
    history = await client.chat.get_history(notebook_id)
    return json.dumps(_serialize(history), indent=2)


# ===================================================================
#  ARTIFACT GENERATION TOOLS
# ===================================================================

@mcp.tool()
async def list_artifacts(notebook_id: str, artifact_type: str = "") -> str:
    """List all generated artifacts in a notebook.

    Args:
        notebook_id: The notebook ID.
        artifact_type: Optional filter – one of: audio, video, report, quiz, flashcards, infographic, slide_deck, data_table. Leave empty for all.
    """
    client = await _get_client()
    type_map = {
        "audio": client.artifacts.list_audio,
        "video": client.artifacts.list_video,
        "report": client.artifacts.list_reports,
        "quiz": client.artifacts.list_quizzes,
        "flashcards": client.artifacts.list_flashcards,
        "infographic": client.artifacts.list_infographics,
        "slide_deck": client.artifacts.list_slide_decks,
        "data_table": client.artifacts.list_data_tables,
    }
    if artifact_type and artifact_type in type_map:
        artifacts = await type_map[artifact_type](notebook_id)
    else:
        artifacts = await client.artifacts.list(notebook_id)
    return json.dumps(_serialize(artifacts), indent=2)


@mcp.tool()
async def generate_audio(
    notebook_id: str,
    instructions: str = "",
    audio_format: str = "deep-dive",
    audio_length: str = "default",
    source_ids: str = "",
) -> str:
    """Generate an audio overview (podcast) from the notebook's sources.

    Args:
        notebook_id: The notebook ID.
        instructions: Optional instructions for the podcast.
        audio_format: Format: deep-dive, brief, critique, or debate.
        audio_length: Length: short, default, or long.
        source_ids: Optional comma-separated source IDs.
    """
    from notebooklm import AudioFormat, AudioLength

    client = await _get_client()
    fmt_map = {"deep-dive": AudioFormat.DEEP_DIVE, "brief": AudioFormat.BRIEF, "critique": AudioFormat.CRITIQUE, "debate": AudioFormat.DEBATE}
    len_map = {"short": AudioLength.SHORT, "default": AudioLength.DEFAULT, "long": AudioLength.LONG}
    src_list = [s.strip() for s in source_ids.split(",") if s.strip()] or None

    status = await client.artifacts.generate_audio(
        notebook_id,
        source_ids=src_list,
        instructions=instructions or None,
        audio_format=fmt_map.get(audio_format, AudioFormat.DEEP_DIVE),
        audio_length=len_map.get(audio_length, AudioLength.DEFAULT),
    )
    return json.dumps(_serialize(status), indent=2)


@mcp.tool()
async def generate_video(
    notebook_id: str,
    instructions: str = "",
    video_format: str = "explainer",
    video_style: str = "auto",
    source_ids: str = "",
) -> str:
    """Generate a video overview from the notebook's sources.

    Args:
        notebook_id: The notebook ID.
        instructions: Optional instructions.
        video_format: Format: explainer or brief.
        video_style: Style: auto, classic, whiteboard, kawaii, anime, watercolor, retro, heritage, paper-craft.
        source_ids: Optional comma-separated source IDs.
    """
    from notebooklm import VideoFormat, VideoStyle

    client = await _get_client()
    fmt_map = {"explainer": VideoFormat.EXPLAINER, "brief": VideoFormat.BRIEF}
    style_map = {
        "auto": VideoStyle.AUTO_SELECT, "classic": VideoStyle.CLASSIC,
        "whiteboard": VideoStyle.WHITEBOARD, "kawaii": VideoStyle.KAWAII,
        "anime": VideoStyle.ANIME, "watercolor": VideoStyle.WATERCOLOR,
        "retro": VideoStyle.RETRO_PRINT, "heritage": VideoStyle.HERITAGE,
        "paper-craft": VideoStyle.PAPER_CRAFT,
    }
    src_list = [s.strip() for s in source_ids.split(",") if s.strip()] or None

    status = await client.artifacts.generate_video(
        notebook_id,
        source_ids=src_list,
        instructions=instructions or None,
        video_format=fmt_map.get(video_format, VideoFormat.EXPLAINER),
        video_style=style_map.get(video_style, VideoStyle.AUTO_SELECT),
    )
    return json.dumps(_serialize(status), indent=2)


@mcp.tool()
async def generate_quiz(
    notebook_id: str,
    instructions: str = "",
    difficulty: str = "medium",
    quantity: str = "standard",
    source_ids: str = "",
) -> str:
    """Generate a quiz from the notebook's sources.

    Args:
        notebook_id: The notebook ID.
        instructions: Optional instructions.
        difficulty: Difficulty: easy, medium, or hard.
        quantity: Quantity: fewer, standard, or more.
        source_ids: Optional comma-separated source IDs.
    """
    from notebooklm import QuizDifficulty, QuizQuantity

    client = await _get_client()
    diff_map = {"easy": QuizDifficulty.EASY, "medium": QuizDifficulty.MEDIUM, "hard": QuizDifficulty.HARD}
    qty_map = {"fewer": QuizQuantity.FEWER, "standard": QuizQuantity.STANDARD, "more": QuizQuantity.MORE}
    src_list = [s.strip() for s in source_ids.split(",") if s.strip()] or None

    status = await client.artifacts.generate_quiz(
        notebook_id,
        source_ids=src_list,
        instructions=instructions or None,
        difficulty=diff_map.get(difficulty, QuizDifficulty.MEDIUM),
        quantity=qty_map.get(quantity, QuizQuantity.STANDARD),
    )
    return json.dumps(_serialize(status), indent=2)


@mcp.tool()
async def generate_flashcards(
    notebook_id: str,
    instructions: str = "",
    difficulty: str = "medium",
    quantity: str = "standard",
    source_ids: str = "",
) -> str:
    """Generate flashcards from the notebook's sources.

    Args:
        notebook_id: The notebook ID.
        instructions: Optional instructions.
        difficulty: Difficulty: easy, medium, or hard.
        quantity: Quantity: fewer, standard, or more.
        source_ids: Optional comma-separated source IDs.
    """
    from notebooklm import QuizDifficulty, QuizQuantity

    client = await _get_client()
    diff_map = {"easy": QuizDifficulty.EASY, "medium": QuizDifficulty.MEDIUM, "hard": QuizDifficulty.HARD}
    qty_map = {"fewer": QuizQuantity.FEWER, "standard": QuizQuantity.STANDARD, "more": QuizQuantity.MORE}
    src_list = [s.strip() for s in source_ids.split(",") if s.strip()] or None

    status = await client.artifacts.generate_flashcards(
        notebook_id,
        source_ids=src_list,
        instructions=instructions or None,
        difficulty=diff_map.get(difficulty, QuizDifficulty.MEDIUM),
        quantity=qty_map.get(quantity, QuizQuantity.STANDARD),
    )
    return json.dumps(_serialize(status), indent=2)


@mcp.tool()
async def generate_report(
    notebook_id: str,
    description: str = "",
    report_format: str = "briefing-doc",
    source_ids: str = "",
) -> str:
    """Generate a text report (briefing doc, study guide, blog post, or custom).

    Args:
        notebook_id: The notebook ID.
        description: Optional description / instructions.
        report_format: Format: briefing-doc, study-guide, blog-post, or custom.
        source_ids: Optional comma-separated source IDs.
    """
    from notebooklm import ReportFormat

    client = await _get_client()
    fmt_map = {
        "briefing-doc": ReportFormat.BRIEFING_DOC, "study-guide": ReportFormat.STUDY_GUIDE,
        "blog-post": ReportFormat.BLOG_POST, "custom": ReportFormat.CUSTOM,
    }
    src_list = [s.strip() for s in source_ids.split(",") if s.strip()] or None

    status = await client.artifacts.generate_report(
        notebook_id,
        source_ids=src_list,
        description=description or None,
        format=fmt_map.get(report_format, ReportFormat.BRIEFING_DOC),
    )
    return json.dumps(_serialize(status), indent=2)


@mcp.tool()
async def generate_slide_deck(
    notebook_id: str,
    description: str = "",
    slide_format: str = "detailed",
    slide_length: str = "default",
    source_ids: str = "",
) -> str:
    """Generate a slide deck presentation.

    Args:
        notebook_id: The notebook ID.
        description: Optional description.
        slide_format: Format: detailed or presenter.
        slide_length: Length: default or short.
        source_ids: Optional comma-separated source IDs.
    """
    from notebooklm import SlideDeckFormat, SlideDeckLength

    client = await _get_client()
    fmt_map = {"detailed": SlideDeckFormat.DETAILED, "presenter": SlideDeckFormat.PRESENTER}
    len_map = {"default": SlideDeckLength.DEFAULT, "short": SlideDeckLength.SHORT}
    src_list = [s.strip() for s in source_ids.split(",") if s.strip()] or None

    status = await client.artifacts.generate_slide_deck(
        notebook_id,
        source_ids=src_list,
        instructions=description or None,
        slide_format=fmt_map.get(slide_format, SlideDeckFormat.DETAILED),
        slide_length=len_map.get(slide_length, SlideDeckLength.DEFAULT),
    )
    return json.dumps(_serialize(status), indent=2)


@mcp.tool()
async def generate_infographic(
    notebook_id: str,
    description: str = "",
    orientation: str = "landscape",
    detail: str = "standard",
    source_ids: str = "",
) -> str:
    """Generate an infographic from the notebook's sources.

    Args:
        notebook_id: The notebook ID.
        description: Optional description.
        orientation: Orientation: landscape, portrait, or square.
        detail: Detail level: concise, standard, or detailed.
        source_ids: Optional comma-separated source IDs.
    """
    from notebooklm import InfographicDetail, InfographicOrientation

    client = await _get_client()
    ori_map = {"landscape": InfographicOrientation.LANDSCAPE, "portrait": InfographicOrientation.PORTRAIT, "square": InfographicOrientation.SQUARE}
    det_map = {"concise": InfographicDetail.CONCISE, "standard": InfographicDetail.STANDARD, "detailed": InfographicDetail.DETAILED}
    src_list = [s.strip() for s in source_ids.split(",") if s.strip()] or None

    status = await client.artifacts.generate_infographic(
        notebook_id,
        source_ids=src_list,
        instructions=description or None,
        orientation=ori_map.get(orientation, InfographicOrientation.LANDSCAPE),
        detail=det_map.get(detail, InfographicDetail.STANDARD),
    )
    return json.dumps(_serialize(status), indent=2)


@mcp.tool()
async def generate_mind_map(notebook_id: str, source_ids: str = "") -> str:
    """Generate a mind map from the notebook's sources.

    Args:
        notebook_id: The notebook ID.
        source_ids: Optional comma-separated source IDs.
    """
    client = await _get_client()
    src_list = [s.strip() for s in source_ids.split(",") if s.strip()] or None
    result = await client.artifacts.generate_mind_map(notebook_id, source_ids=src_list)
    return json.dumps(_serialize(result), indent=2)


@mcp.tool()
async def generate_data_table(
    notebook_id: str, description: str, source_ids: str = ""
) -> str:
    """Generate a data table from the notebook's sources.

    Args:
        notebook_id: The notebook ID.
        description: Description of the table structure (e.g. "compare key concepts").
        source_ids: Optional comma-separated source IDs.
    """
    client = await _get_client()
    src_list = [s.strip() for s in source_ids.split(",") if s.strip()] or None
    status = await client.artifacts.generate_data_table(notebook_id, source_ids=src_list, instructions=description)
    return json.dumps(_serialize(status), indent=2)


# ===================================================================
#  ARTIFACT STATUS & DOWNLOAD TOOLS
# ===================================================================

@mcp.tool()
async def poll_artifact_status(notebook_id: str, task_id: str) -> str:
    """Check the generation status of an artifact.

    Args:
        notebook_id: The notebook ID.
        task_id: The task ID returned by a generate command.
    """
    client = await _get_client()
    status = await client.artifacts.poll_status(notebook_id, task_id)
    return json.dumps(_serialize(status), indent=2)


@mcp.tool()
async def wait_for_artifact(notebook_id: str, task_id: str, timeout: int = 300) -> str:
    """Wait for an artifact to finish generating (blocking).

    Args:
        notebook_id: The notebook ID.
        task_id: The task ID from a generate command.
        timeout: Max seconds to wait (default 300).
    """
    client = await _get_client()
    status = await client.artifacts.wait_for_completion(notebook_id, task_id, timeout=timeout)
    return json.dumps(_serialize(status), indent=2)


@mcp.tool()
async def download_artifact(
    notebook_id: str,
    artifact_type: str,
    output_path: str,
    artifact_id: str = "",
    output_format: str = "",
) -> str:
    """Download a generated artifact to a local file.

    Args:
        notebook_id: The notebook ID.
        artifact_type: Type: audio, video, infographic, slide_deck, report, mind_map, data_table, quiz, or flashcards.
        output_path: Local path for the downloaded file.
        artifact_id: Optional specific artifact ID. If empty, downloads the latest.
        output_format: For quiz/flashcards: json, markdown, or html. For slide_deck: pdf or pptx.
    """
    client = await _get_client()
    aid = artifact_id if artifact_id else None
    download_map = {
        "audio": client.artifacts.download_audio,
        "video": client.artifacts.download_video,
        "infographic": client.artifacts.download_infographic,
        "slide_deck": client.artifacts.download_slide_deck,
        "report": client.artifacts.download_report,
        "mind_map": client.artifacts.download_mind_map,
        "data_table": client.artifacts.download_data_table,
    }
    format_map = {
        "quiz": client.artifacts.download_quiz,
        "flashcards": client.artifacts.download_flashcards,
    }

    if artifact_type in format_map:
        fmt = output_format if output_format else "json"
        path = await format_map[artifact_type](notebook_id, output_path, artifact_id=aid, output_format=fmt)
    elif artifact_type in download_map:
        path = await download_map[artifact_type](notebook_id, output_path, artifact_id=aid)
    else:
        return json.dumps({"error": f"Unknown artifact type: {artifact_type}"})

    return json.dumps({"downloaded": True, "path": str(path)})


# ===================================================================
#  RESEARCH TOOLS
# ===================================================================

@mcp.tool()
async def start_research(
    notebook_id: str,
    query: str,
    source: str = "web",
    mode: str = "fast",
) -> str:
    """Start an AI research session to discover and add sources.

    Args:
        notebook_id: The notebook ID.
        query: The research query.
        source: Source to search: web or drive.
        mode: Research depth: fast or deep.
    """
    client = await _get_client()
    result = await client.research.start(notebook_id, query, source=source, mode=mode)
    return json.dumps(_serialize(result), indent=2)


@mcp.tool()
async def poll_research(notebook_id: str) -> str:
    """Check the status of a running research session.

    Args:
        notebook_id: The notebook ID.
    """
    client = await _get_client()
    status = await client.research.poll(notebook_id)
    return json.dumps(_serialize(status), indent=2)


@mcp.tool()
async def import_research_sources(
    notebook_id: str, task_id: str, max_sources: int = 5
) -> str:
    """Import sources discovered by a research session into the notebook.

    Args:
        notebook_id: The notebook ID.
        task_id: The research task ID.
        max_sources: Maximum number of sources to import (default 5).
    """
    client = await _get_client()
    status = await client.research.poll(notebook_id)
    sources = status.get("sources", [])[:max_sources]
    if not sources:
        return json.dumps({"imported": 0, "message": "No sources found to import"})
    result = await client.research.import_sources(notebook_id, task_id, sources)
    return json.dumps({"imported": len(result), "sources": _serialize(result)}, indent=2)


# ===================================================================
#  NOTE TOOLS
# ===================================================================

@mcp.tool()
async def list_notes(notebook_id: str) -> str:
    """List all notes in a notebook.

    Args:
        notebook_id: The notebook ID.
    """
    client = await _get_client()
    notes = await client.notes.list(notebook_id)
    return json.dumps(_serialize(notes), indent=2)


@mcp.tool()
async def create_note(notebook_id: str, title: str = "New Note", content: str = "") -> str:
    """Create a new note in a notebook.

    Args:
        notebook_id: The notebook ID.
        title: The note title.
        content: The note content.
    """
    client = await _get_client()
    note = await client.notes.create(notebook_id, title=title, content=content)
    return json.dumps(_serialize(note), indent=2)


@mcp.tool()
async def delete_note(notebook_id: str, note_id: str) -> str:
    """Delete a note from a notebook.

    Args:
        notebook_id: The notebook ID.
        note_id: The note ID to delete.
    """
    client = await _get_client()
    result = await client.notes.delete(notebook_id, note_id)
    return json.dumps({"deleted": result, "note_id": note_id})


# ===================================================================
#  SHARING TOOLS
# ===================================================================

@mcp.tool()
async def get_share_status(notebook_id: str) -> str:
    """Get the current sharing status and shared users for a notebook.

    Args:
        notebook_id: The notebook ID.
    """
    client = await _get_client()
    status = await client.sharing.get_status(notebook_id)
    return json.dumps(_serialize(status), indent=2)


@mcp.tool()
async def set_notebook_public(notebook_id: str, public: bool) -> str:
    """Enable or disable public link sharing for a notebook.

    Args:
        notebook_id: The notebook ID.
        public: True to enable, False to disable.
    """
    client = await _get_client()
    status = await client.sharing.set_public(notebook_id, public)
    return json.dumps(_serialize(status), indent=2)


@mcp.tool()
async def share_with_user(
    notebook_id: str,
    email: str,
    permission: str = "viewer",
    welcome_message: str = "",
) -> str:
    """Share a notebook with a specific user.

    Args:
        notebook_id: The notebook ID.
        email: Email address of the user to share with.
        permission: Permission level: viewer or editor.
        welcome_message: Optional welcome message.
    """
    from notebooklm import SharePermission

    client = await _get_client()
    perm_map = {"viewer": SharePermission.VIEWER, "editor": SharePermission.EDITOR}
    status = await client.sharing.add_user(
        notebook_id,
        email,
        perm_map.get(permission, SharePermission.VIEWER),
        welcome_message=welcome_message or None,
    )
    return json.dumps(_serialize(status), indent=2)


# ===================================================================
#  SETTINGS TOOLS
# ===================================================================

@mcp.tool()
async def get_language() -> str:
    """Get the current output language setting for artifact generation."""
    client = await _get_client()
    lang = await client.settings.get_output_language()
    return json.dumps({"language": lang})


@mcp.tool()
async def set_language(language_code: str) -> str:
    """Set the output language for artifact generation (global setting).

    Args:
        language_code: Language code, e.g. en, ja, zh_Hans, es, fr, de.
    """
    client = await _get_client()
    result = await client.settings.set_output_language(language_code)
    return json.dumps({"language": result})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run(transport="stdio")
