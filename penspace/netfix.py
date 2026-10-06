"""Repair a container's DNS when the host hands it a resolver that goes nowhere.

A RunPod pod is handed Docker's embedded resolver at 127.0.0.11, which forwards
to whatever the host put in its own /etc/resolv.conf. On at least one EU-RO-1
machine that upstream was 127.0.0.1 — the host's loopback — so every lookup
inside the container failed:

    Failed to resolve 'huggingface.co' ([Errno -3] Temporary failure in name
    resolution)

The symptom is easy to misread. Hugging Face appears in the traceback, so it
looks like a model-download problem, but the backend is equally unreachable and
the worker simply never claims a job. An hour went into that diagnosis once.

This is deliberately a no-op when DNS already works: it resolves a name first
and only rewrites /etc/resolv.conf if that fails. A working host resolver is
left alone, because it may be the only route to a private backend that public
nameservers cannot see.
"""

from __future__ import annotations

import logging
import socket
from pathlib import Path

log = logging.getLogger("penspace.netfix")

RESOLV_CONF = Path("/etc/resolv.conf")
PUBLIC_NAMESERVERS = ("1.1.1.1", "8.8.8.8")


def _resolves(host: str) -> bool:
    try:
        socket.getaddrinfo(host, None)
        return True
    except socket.gaierror:
        return False


def ensure_resolvable(*hosts: str) -> bool:
    """Make `hosts` resolvable, repairing the resolver if it is broken.

    Returns True if every host resolves on return. Never raises: a container
    that cannot write /etc/resolv.conf should fail later with a real error from
    the request it was trying to make, not here.
    """
    hosts = tuple(h for h in hosts if h)
    if not hosts:
        return True

    unresolved = [h for h in hosts if not _resolves(h)]
    if not unresolved:
        return True

    log.warning(
        "DNS cannot resolve %s — rewriting %s with %s",
        ", ".join(unresolved), RESOLV_CONF, ", ".join(PUBLIC_NAMESERVERS),
    )
    try:
        RESOLV_CONF.write_text(
            "".join(f"nameserver {ns}\n" for ns in PUBLIC_NAMESERVERS)
        )
    except OSError as exc:
        log.error("could not rewrite %s (%s) — DNS stays broken", RESOLV_CONF, exc)
        return False

    still_broken = [h for h in hosts if not _resolves(h)]
    if still_broken:
        log.error("still cannot resolve %s after the rewrite", ", ".join(still_broken))
        return False

    log.info("DNS repaired; %s now resolve", ", ".join(unresolved))
    return True
