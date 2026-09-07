"""
Sandboxed patch verification.

`generate_fix_node` (app/graph.py) produces a candidate corrected version of
one source file. Before that becomes a pull request, this module applies it
inside a throwaway Docker container, runs the repo's own test suite against
it, and reports whether it passed. Only a green fix reaches `open_pr_node`;
a fix that never goes green is escalated with the failing output attached,
never merged blind (the pre-Phase-1 behavior was to open the PR immediately
with a "this has NOT been tested" disclaimer -- this is what removes that
disclaimer honestly).

Why Docker specifically, and not a subprocess or a venv:
  * The proposed patch is LLM-generated code from an untrusted-by-construction
    source, run against an arbitrary monitored repo. It gets a container with
    a hard memory/CPU cap and a wall-clock timeout -- a bad or hostile patch
    cannot touch the host.
  * "Run the repo's real tests" means running *that repo's* toolchain (its
    Node version, its pytest plugins, its fixtures) -- a container from a
    language-appropriate base image is the only faithful way to do that
    without polluting this service's own environment.

One container does the whole job: clone the repo at `ref`, overlay the
patched file, install dependencies, run the test suite. The container's
final exit code is the verdict. Toolchain detection (npm vs pip) happens
*inside* that script from the cloned repo's own files -- there is no
separate probe container. Everything is torn down in a `finally`.
"""
import io
import logging
import shlex
import tarfile
import time
from dataclasses import dataclass

from opentelemetry import trace

from app.config import get_settings
from app.services.telemetry import get_tracer

logger = logging.getLogger("triage_engine.sandbox")


class SandboxError(RuntimeError):
    """The verification could not be *run* (no Docker daemon, image pull failed, timeout)."""


@dataclass(frozen=True)
class SandboxResult:
    """
    Outcome of one verification attempt.

    `ran` distinguishes "we executed the test suite, here is the result"
    (`ran=True`, `passed` is meaningful) from "we could not get as far as
    running tests" (`ran=False`, `skipped_reason` says why -- an
    unrecognizable repo, a clone failure, the target file absent). The graph
    treats the second case as "verification unavailable -> escalate", never
    as "the fix failed".
    """

    ran: bool
    passed: bool
    exit_code: int | None
    output_tail: str
    duration_s: float
    test_command: str
    skipped_reason: str = ""


# Sentinel exit codes the in-container script uses for conditions that are
# not a test pass/fail -- kept well clear of the 1-125 range real test
# runners use.
_EXIT_TARGET_FILE_MISSING = 90
_EXIT_TOOLCHAIN_UNKNOWN = 91
_EXIT_CLONE_FAILED = 92


def _build_verification_script(*, repo_url: str, ref: str, file_path: str) -> str:
    """
    The one shell script the container runs.

    1. Shallow-clone `repo_url` at `ref` into /work.
    2. Overlay the patched file (piped in separately at /patch/new_content)
       onto /work/`file_path`.
    3. Detect the toolchain from /work's own files and install + test:
         package.json                     -> npm ci || npm install ; npm test
         pyproject.toml / requirements.txt -> pip install ... ; python -m pytest -q
       An operator override (SANDBOX_TEST_COMMAND_OVERRIDE) replaces the
       test command; SANDBOX_IMAGE_OVERRIDE only affects the base image,
       handled by the caller.
    4. Exit with the test runner's own exit code (0 == pass), or one of the
       _EXIT_* sentinels for a non-test condition.

    Section markers are echoed so the output tail kept on the ticket reads
    cleanly for a human.
    """
    settings = get_settings()
    safe_path = shlex.quote(file_path)
    safe_url = shlex.quote(repo_url)
    safe_ref = shlex.quote(ref)
    test_override = settings.sandbox_test_command_override

    if test_override:
        detect_and_run = f"""
echo "=== SANDBOX: install (operator-configured) ==="
pip install -q -e . 2>/dev/null || pip install -q -r requirements.txt 2>/dev/null || npm ci 2>/dev/null || npm install 2>/dev/null || true
echo "=== SANDBOX: running tests -> {test_override} ==="
{test_override}
exit $?
"""
    else:
        detect_and_run = f"""
if [ -f package.json ]; then
  echo "=== SANDBOX: detected Node project ==="
  echo "=== SANDBOX: installing dependencies (npm) ==="
  npm ci 2>/dev/null || npm install
  echo "=== SANDBOX: running tests -> npm test ==="
  npm test --silent
  exit $?
elif [ -f pyproject.toml ] || [ -f requirements.txt ] || [ -f setup.py ]; then
  echo "=== SANDBOX: detected Python project ==="
  echo "=== SANDBOX: installing dependencies (pip) ==="
  pip install -q -r requirements.txt 2>/dev/null || true
  pip install -q -r requirements-dev.txt 2>/dev/null || true
  pip install -q -r test-requirements.txt 2>/dev/null || true
  # Editable install with common test extras; fall back to a plain editable
  # install, then ensure pytest itself is importable regardless.
  pip install -q -e '.[test]' 2>/dev/null || pip install -q -e '.[tests]' 2>/dev/null \
    || pip install -q -e '.[dev]' 2>/dev/null || pip install -q -e . 2>/dev/null || true
  python -c 'import pytest' 2>/dev/null || pip install -q pytest
  echo "=== SANDBOX: running tests -> python -m pytest -q ==="
  python -m pytest -q
  exit $?
else
  echo "=== SANDBOX: could not determine how to install/test this repo ==="
  exit {_EXIT_TOOLCHAIN_UNKNOWN}
fi
"""

    return f"""
set -u
echo "=== SANDBOX: cloning {repo_url} @ {ref} ==="
git clone --depth 1 --branch {safe_ref} {safe_url} /work 2>/dev/null \
  || git clone --depth 1 {safe_url} /work \
  || {{ echo "=== SANDBOX: clone failed ==="; exit {_EXIT_CLONE_FAILED}; }}
cd /work
if [ ! -f {safe_path} ]; then
  echo "=== SANDBOX: target file {safe_path} not found in repo ==="
  exit {_EXIT_TARGET_FILE_MISSING}
fi
echo "=== SANDBOX: applying proposed fix to {safe_path} ==="
mkdir -p "$(dirname {safe_path})"
cp /patch/new_content {safe_path}
{detect_and_run}
"""


def _pick_base_image(file_path: str) -> str:
    """
    Choose the container image. An explicit override wins. Otherwise guess
    from the patched file's extension -- a .js/.ts/.jsx/.tsx fix wants Node,
    everything else defaults to the Python image (which is also fine for a
    repo whose tests are pure-shell). The in-container script does the real
    npm-vs-pip decision from the repo's files; this only picks which
    language runtime is present.
    """
    settings = get_settings()
    if settings.sandbox_image_override:
        return settings.sandbox_image_override
    lowered = file_path.lower()
    if lowered.endswith((".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")):
        return settings.sandbox_image_node
    return settings.sandbox_image_python


def _tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return "...(truncated)...\n" + text[-limit:]


def _put_patch_file(container, new_content: str) -> None:
    """Copy the patched file content into the container at /patch/new_content via the archive API."""
    data = new_content.encode("utf-8")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        info = tarfile.TarInfo(name="new_content")
        info.size = len(data)
        info.mtime = int(time.time())
        tar.addfile(info, io.BytesIO(data))
    stream.seek(0)
    container.put_archive("/patch", stream.read())


def _empty_dir_tar(dirname: str) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        info = tarfile.TarInfo(name=dirname)
        info.type = tarfile.DIRTYPE
        info.mode = 0o777
        info.mtime = int(time.time())
        tar.addfile(info)
    return stream.getvalue()


def _safe_logs(container) -> str:
    try:
        return container.logs(stdout=True, stderr=True).decode("utf-8", "replace")
    except Exception:  # pragma: no cover
        return "(could not read container logs)"


async def verify_patch_in_sandbox(
    *,
    repo_url: str,
    ref: str,
    file_path: str,
    new_content: str,
) -> SandboxResult:
    """
    Run `repo_url`@`ref` with `file_path` replaced by `new_content`, execute
    its test suite in a disposable container, and report the result.

    Never raises for a *test failure* -- that is a normal `SandboxResult(
    ran=True, passed=False, ...)`. Raises `SandboxError` only when the
    verification itself could not be carried out (no Docker daemon
    reachable, image pull failed, timeout). The caller (verify_fix_node)
    maps that to an escalation, not a retry.

    Synchronous docker-SDK calls are offloaded to a worker thread so the
    FastAPI event loop is not blocked for the (potentially minutes-long)
    duration of a container run.
    """
    import anyio

    return await anyio.to_thread.run_sync(
        _verify_patch_blocking, repo_url, ref, file_path, new_content
    )


def _verify_patch_blocking(
    repo_url: str, ref: str, file_path: str, new_content: str
) -> SandboxResult:
    settings = get_settings()
    tracer = get_tracer()
    started = time.perf_counter()

    with tracer.start_as_current_span("sandbox.verify") as span:
        span.set_attribute("sandbox.repo_url", repo_url)
        span.set_attribute("sandbox.ref", ref)
        span.set_attribute("sandbox.file_path", file_path)

        try:
            import docker
            from docker.errors import DockerException, ImageNotFound, NotFound
        except ImportError as exc:  # pragma: no cover - docker is in requirements.txt
            raise SandboxError(
                "The `docker` package is required for sandboxed fix verification "
                "(pip install docker), or set SANDBOX_ENABLED=false."
            ) from exc

        try:
            client = (
                docker.DockerClient(base_url=settings.sandbox_docker_host)
                if settings.sandbox_docker_host
                else docker.from_env()
            )
            client.ping()
        except DockerException as exc:
            span.set_status(trace.Status(trace.StatusCode.ERROR, "no_docker_daemon"))
            raise SandboxError(
                f"Could not reach a Docker daemon for sandbox verification: {exc}. "
                "Start Docker, or set SANDBOX_ENABLED=false to skip verification."
            ) from exc

        image = _pick_base_image(file_path)
        span.set_attribute("sandbox.image", image)
        script = _build_verification_script(repo_url=repo_url, ref=ref, file_path=file_path)

        container = None
        try:
            try:
                client.images.get(image)
            except ImageNotFound:
                logger.info("sandbox: pulling image %s", image)
                client.images.pull(image)

            # git is not in the slim language images by default -- install it
            # as the first step of the script rather than maintaining custom
            # images. `apt-get`/`apk` both covered.
            bootstrapped_script = (
                "(command -v git >/dev/null 2>&1) || "
                "(apt-get update -qq && apt-get install -y -qq git) || "
                "(apk add --no-cache git) || true\n"
            ) + script

            container = client.containers.create(
                image=image,
                command=["sh", "-c", bootstrapped_script],
                working_dir="/",
                network_mode="bridge",  # needed for clone + dependency install
                mem_limit="1g",
                nano_cpus=2_000_000_000,  # 2 CPUs
                detach=True,
            )
            container.put_archive("/", _empty_dir_tar("patch"))
            _put_patch_file(container, new_content)
            container.start()

            try:
                exit_status = container.wait(timeout=settings.sandbox_timeout_seconds)
                exit_code = int(exit_status.get("StatusCode", -1))
            except Exception as exc:  # docker raises requests.ReadTimeout on wait timeout
                span.set_status(trace.Status(trace.StatusCode.ERROR, "timeout"))
                logs = _safe_logs(container)
                raise SandboxError(
                    f"Sandbox verification timed out after {settings.sandbox_timeout_seconds}s. "
                    f"Last output:\n{_tail(logs, settings.sandbox_output_tail_chars)}"
                ) from exc

            logs = _safe_logs(container)
            output_tail = _tail(logs, settings.sandbox_output_tail_chars)
            duration = time.perf_counter() - started

            if exit_code in (_EXIT_TARGET_FILE_MISSING, _EXIT_TOOLCHAIN_UNKNOWN, _EXIT_CLONE_FAILED):
                reason = {
                    _EXIT_TARGET_FILE_MISSING: f"Target file {file_path!r} was not found in the repo at {ref}.",
                    _EXIT_TOOLCHAIN_UNKNOWN: (
                        "Could not determine how to install/test this repo "
                        "(no package.json, requirements.txt, pyproject.toml, or setup.py at its root)."
                    ),
                    _EXIT_CLONE_FAILED: f"Could not clone {repo_url} at {ref}.",
                }[exit_code]
                span.set_attribute("sandbox.ran", False)
                span.set_status(trace.Status(trace.StatusCode.OK))
                return SandboxResult(
                    ran=False, passed=False, exit_code=exit_code,
                    output_tail=output_tail, duration_s=duration,
                    test_command="", skipped_reason=reason,
                )

            passed = exit_code == 0
            # Best-effort: pull the echoed test command out of the log for the ticket.
            test_command = "npm test" if "npm test" in logs else ("python -m pytest -q" if "pytest" in logs else "")
            span.set_attribute("sandbox.ran", True)
            span.set_attribute("sandbox.passed", passed)
            span.set_attribute("sandbox.exit_code", exit_code)
            span.set_attribute("sandbox.duration_s", round(duration, 2))
            span.set_status(trace.Status(trace.StatusCode.OK))
            return SandboxResult(
                ran=True, passed=passed, exit_code=exit_code,
                output_tail=output_tail, duration_s=duration, test_command=test_command,
            )
        except (DockerException, NotFound) as exc:
            span.record_exception(exc)
            span.set_status(trace.Status(trace.StatusCode.ERROR, "docker_error"))
            raise SandboxError(f"Sandbox verification failed to run: {exc}") from exc
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:  # pragma: no cover - cleanup best effort
                    logger.warning("sandbox: failed to remove container %s", container.id, exc_info=True)
