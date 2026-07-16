"""Custom E2B template with the agent preinstalled, for fast sandbox cold starts."""

from __future__ import annotations

from e2b import Template, default_build_logger

TEMPLATE_NAME = "flight-recorder"


def build_template(cpu_count: int = 1, memory_mb: int = 1024) -> str:
    template = Template().from_node_image("24").npm_install("@anthropic-ai/claude-code", g=True)
    Template.build(
        template,
        alias=TEMPLATE_NAME,
        cpu_count=cpu_count,
        memory_mb=memory_mb,
        on_build_logs=default_build_logger(),
    )
    return TEMPLATE_NAME
