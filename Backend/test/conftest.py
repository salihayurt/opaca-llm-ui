"""
Shared pytest configuration.

Async tests in this suite are marked with @pytest.mark.anyio, and anyio's
default `anyio_backend` fixture parametrises over backends -- but which ones
depends on the installed version. anyio 4.10 offers both asyncio and trio
whether or not trio is installed, so every async test is collected twice and
the trio variant errors with ModuleNotFoundError; anyio 4.14 offers only the
backends actually present, and the same suite passes. The result is a test
run that fails on one machine and succeeds on another for reasons unrelated
to the code under test.

Pinning asyncio is also the accurate choice rather than a workaround. SAGE
runs on FastAPI and uvicorn, and the clients it depends on -- qdrant-client
among them -- are built on asyncio, so a trio run exercises a combination
that cannot occur in production.

Adding trio as a test dependency would be the alternative. It would double
the runtime of every async test to cover a configuration nothing deploys.
"""

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"