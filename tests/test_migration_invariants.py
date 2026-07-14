import re
from pathlib import Path

MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "auction-lens-ai"
    / "supabase"
    / "migrations"
    / "20260714040000_ai_backend_foundations.sql"
)


def migration_sql() -> str:
    return MIGRATION.read_text(encoding="utf-8").casefold()


def test_migration_contains_required_foundations() -> None:
    sql = migration_sql()

    for required in (
        "create extension if not exists vector",
        "create table public.documents",
        "create table public.document_versions",
        "create table public.document_chunks",
        "extensions.vector(1536)",
        "create table public.auction_record_mortgagors",
        "create table public.ingestion_jobs",
        "create table public.source_sync_runs",
        "create table public.conversation_threads",
        "create table public.conversation_messages",
        "create table public.tool_execution_audit",
        "create table public.user_feedback",
        "create table public.evaluation_runs",
        "create table public.evaluation_results",
        "with (security_invoker = true)",
        "security definer",
        "set search_path = pg_catalog, public, extensions",
        "create schema if not exists langgraph",
    ):
        assert required in sql


def test_rid_is_indexed_but_never_unique() -> None:
    sql = migration_sql()

    assert "auction_records_rid_idx" in sql
    assert re.search(r"unique\s+(?:index\s+)?[^\n;]*\brid\b", sql) is None
    assert "auction_records_stable_key_uq" in sql


def test_sensitive_tables_enable_rls_and_functions_have_narrow_grants() -> None:
    sql = migration_sql()

    for table in (
        "documents",
        "document_versions",
        "document_chunks",
        "ingestion_jobs",
        "conversation_threads",
        "conversation_messages",
        "tool_execution_audit",
        "user_feedback",
        "evaluation_runs",
        "evaluation_results",
    ):
        assert f"alter table public.{table} enable row level security" in sql
    assert "revoke all on all tables in schema langgraph from public, anon, authenticated" in sql
    assert "grant execute on function public.search_auction_records" in sql
