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

Shape of a run:
  1. If the repo's docker-compose.yml declares a database service and
     `settings.sandbox_provision_db` is on, create a private Docker network
     and start that one service (e.g. postgres:16) as a sidecar, wait for
     it to accept connections.
  2. Start the test container on the same network: clone the repo at `ref`,
     overlay the patched file, install dependencies, run the test suite,
     with `DATABASE_URL` (settings.sandbox_db_url_env) pointed at the sidecar.
  3. The test container's exit code is the verdict.
  4. Tear down the container, the sidecar, and the network in a `finally`.

Toolchain detection (npm vs pip) happens *inside* the test script from the
cloned repo's own files -- there is no separate probe container.
"""
import io
import logging
import re
import shlex
import tarfile
import time
import uuid
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
    running tests" (`ran=False`, `skipped_reason` says why). The graph
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


# Sentinel exit codes the in-container script uses for non-test conditions.
_EXIT_TARGET_FILE_MISSING = 90
_EXIT_TOOLCHAIN_UNKNOWN = 91
_EXIT_CLONE_FAILED = 92

# Postgres images the DB sidecar knows how to wait on. Extendable, but
# postgres covers this project's demo-app and the overwhelming majority of
# real backend repos.
_DB_IMAGE_HINTS = ("postgres", "postgis")


@dataclass(frozen=True)
class _DbSidecarSpec:
    image: str
    env: dict[str, str]
    # The URL the test container uses to reach the sidecar (hostname is the
    # sidecar's network alias, not localhost).
    connection_url: str
    ready_cmd: list[str]


def _parse_compose_for_db(compose_text: str) -> _DbSidecarSpec | None:
    """
    Pull a single Postgres service definition out of a docker-compose.yml
    without a YAML dependency -- a deliberately narrow regex scan, since the
    only thing needed is the image tag and the POSTGRES_* env vars.

    Returns None if no recognizable Postgres service is present (the caller
    then runs the tests without a DB, which is correct for a repo that
    doesn't need one).
    """
    # image: postgres:16   (also postgres:16-alpine, postgis/postgis:16-3.4)
    image_match = re.search(
        r"image:\s*['\"]?((?:postgis/)?(?:" + "|".join(_DB_IMAGE_HINTS) + r")[^\s'\"]*)['\"]?",
        compose_text,
    )
    if not image_match:
        return None
    image = image_match.group(1)

    def _env(key: str, default: str) -> str:
        m = re.search(rf"{key}:\s*['\"]?([^\s'\"#]+)['\"]?", compose_text)
        return m.group(1) if m else default

    user = _env("POSTGRES_USER", "postgres")
    password = _env("POSTGRES_PASSWORD", "postgres")
    db = _env("POSTGRES_DB", user)

    return _DbSidecarSpec(
        image=image,
        env={"POSTGRES_USER": user, "POSTGRES_PASSWORD": password, "POSTGRES_DB": db},
        # 'sandbox-db' is the network alias assigned when the sidecar starts.
        connection_url=f"postgresql://{user}:{password}@sandbox-db:5432/{db}",
        ready_cmd=["pg_isready", "-U", user, "-d", db],
    )


def _build_verification_script(*, repo_url: str, ref: str, file_path: str) -> str:
    """
    The shell script the test container runs: clone, overlay the patch,
    detect the toolchain from the repo's own files, install, test. Exit
    with the test runner's own code (0 == pass) or an _EXIT_* sentinel.
    """
    settings = get_settings()
    safe_path = shlex.quote(file_path)
    safe_url = shlex.quote(repo_url)
    safe_ref = shlex.quote(ref)
    test_override = settings.sandbox_test_command_override

    if test_override:
        detect_and_run = f"""
echo "=== SANDBOX: install (operator-configured) ==="
npm ci 2>/dev/null || npm install 2>/dev/null || pip install -q -e . 2>/dev/null || pip install -q -r requirements.txt 2>/dev/null || true
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
    settings = get_settings()
    if settings.sandbox_image_override:
        return settings.sandbox_image_override
    if file_path.lower().endswith((".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")):
        return settings.sandbox_image_node
    return settings.sandbox_image_python


def _tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return "...(truncated)...\n" + text[-limit:]


def _put_patch_file(container, new_content: str) -> None:
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


def _fetch_compose_text(repo_url: str, ref: str) -> str:
    """
    Read the repo's docker-compose.yml over the GitHub raw endpoint without
    cloning -- just enough to decide whether a DB sidecar is needed before
    the test container starts. Returns "" if there is none / it can't be
    fetched (the run then proceeds without a DB).
    """
    m = re.match(r"https://github\.com/([^/]+)/([^/.]+)(?:\.git)?", repo_url)
    if not m:
        return ""
    owner, repo = m.group(1), m.group(2)
    import httpx

    for name in ("docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"):
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{name}"
        try:
            resp = httpx.get(url, timeout=10.0)
            if resp.status_code == 200:
                return resp.text
        except httpx.HTTPError:
            continue
    return ""


async def verify_patch_in_sandbox(
    *, repo_url: str, ref: str, file_path: str, new_content: str
) -> SandboxResult:
    """
    Run `repo_url`@`ref` with `file_path` replaced by `new_content`, execute
    its test suite in a disposable container (with a DB sidecar if the repo
    needs one), and report the result.

    Never raises for a *test failure* -- that is a normal `SandboxResult(
    ran=True, passed=False, ...)`. Raises `SandboxError` only when the
    verification itself could not be carried out.

    Blocking docker-SDK calls run on a worker thread so the FastAPI event
    loop is never blocked for the duration of a container run.
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
        except ImportError as exc:  # pragma: no cover
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

        run_id = uuid.uuid4().hex[:10]
        image = _pick_base_image(file_path)
        span.set_attribute("sandbox.image", image)

        db_spec: _DbSidecarSpec | None = None
        if settings.sandbox_provision_db:
            compose_text = _fetch_compose_text(repo_url, ref)
            db_spec = _parse_compose_for_db(compose_text) if compose_text else None
            span.set_attribute("sandbox.db_sidecar", db_spec is not None)

        network = None
        db_container = None
        test_container = None
        env: dict[str, str] = {}
        try:
            # --- optional DB sidecar -------------------------------------
            if db_spec is not None:
                network = client.networks.create(f"sandbox-net-{run_id}", driver="bridge")
                logger.info("sandbox[%s]: starting DB sidecar %s", run_id, db_spec.image)
                try:
                    client.images.get(db_spec.image)
                except ImageNotFound:
                    client.images.pull(db_spec.image)
                db_container = client.containers.run(
                    image=db_spec.image,
                    environment=db_spec.env,
                    network=network.name,
                    name=f"sandbox-db-{run_id}",
                    hostname="sandbox-db",
                    mem_limit="512m",
                    detach=True,
                )
                network.disconnect(db_container)
                network.connect(db_container, aliases=["sandbox-db"])
                _wait_for_db_ready(db_container, db_spec, settings.sandbox_db_startup_seconds, run_id)
                env[settings.sandbox_db_url_env] = db_spec.connection_url

            # --- test container ----------------------------------------
            script = _build_verification_script(repo_url=repo_url, ref=ref, file_path=file_path)
            bootstrapped = (
                "(command -v git >/dev/null 2>&1) || "
                "(apt-get update -qq && apt-get install -y -qq git) || "
                "(apk add --no-cache git) || true\n"
            ) + script

            try:
                client.images.get(image)
            except ImageNotFound:
                logger.info("sandbox[%s]: pulling image %s", run_id, image)
                client.images.pull(image)

            test_container = client.containers.create(
                image=image,
                command=["sh", "-c", bootstrapped],
                environment={"NODE_ENV": "test", "CI": "true", **env},
                working_dir="/",
                network=network.name if network is not None else "bridge",
                mem_limit="1g",
                nano_cpus=2_000_000_000,
                detach=True,
            )
            test_container.put_archive("/", _empty_dir_tar("patch"))
            _put_patch_file(test_container, new_content)
            test_container.start()

            try:
                exit_status = test_container.wait(timeout=settings.sandbox_timeout_seconds)
                exit_code = int(exit_status.get("StatusCode", -1))
            except Exception as exc:
                span.set_status(trace.Status(trace.StatusCode.ERROR, "timeout"))
                logs = _safe_logs(test_container)
                raise SandboxError(
                    f"Sandbox verification timed out after {settings.sandbox_timeout_seconds}s. "
                    f"Last output:\n{_tail(logs, settings.sandbox_output_tail_chars)}"
                ) from exc

            logs = _safe_logs(test_container)
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
            for c in (test_container, db_container):
                if c is not None:
                    try:
                        c.remove(force=True)
                    except Exception:  # pragma: no cover
                        logger.warning("sandbox[%s]: failed to remove container", run_id, exc_info=True)
            if network is not None:
                try:
                    network.remove()
                except Exception:  # pragma: no cover
                    logger.warning("sandbox[%s]: failed to remove network", run_id, exc_info=True)


def _wait_for_db_ready(db_container, spec: _DbSidecarSpec, timeout_s: float, run_id: str) -> None:
    """Poll the sidecar's readiness command until it succeeds or `timeout_s` elapses."""
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            code, out = db_container.exec_run(spec.ready_cmd)
            last = out.decode("utf-8", "replace") if isinstance(out, bytes) else str(out)
            if code == 0:
                logger.info("sandbox[%s]: DB sidecar ready", run_id)
                return
        except Exception as exc:  # container may not have its exec server up yet
            last = str(exc)
        time.sleep(1.5)
    raise SandboxError(f"DB sidecar did not become ready within {timeout_s}s (last: {last!r})")
