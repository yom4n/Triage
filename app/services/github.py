"""
GitHub integration for the auto-fix loop: read a file from the monitored
repo, and (separately) push a proposed fix as a branch + pull request.

Plain REST calls via `httpx` -- consistent with app/services/llm.py and
app/services/embeddings.py -- rather than a GitHub SDK dependency (PyGithub
et al.), since the whole surface area needed here is four endpoints.

Every call is read-then-write against GitHub's Contents/Git-refs/Pulls
APIs, in the sequence a human would use the UI for the same task:

    1. GET  /repos/{o}/{r}/contents/{path}      -- current file + its sha
    2. GET  /repos/{o}/{r}/git/ref/heads/{base}  -- base branch's commit sha
    3. POST /repos/{o}/{r}/git/refs              -- create the fix branch
    4. PUT  /repos/{o}/{r}/contents/{path}       -- commit the fixed file
    5. POST /repos/{o}/{r}/pulls                 -- open the PR

No step here ever pushes straight to the base branch and no step merges
anything -- the PR is always the terminal artifact, left for a human to
review (see README's "Resilience" / auto-fix sections for why that's a
deliberate choice, not a missing feature).
"""
import base64
import logging

import httpx
from opentelemetry import trace

from app.config import get_settings
from app.services.telemetry import get_tracer

logger = logging.getLogger("triage_engine.github")


class GitHubError(RuntimeError):
    """Raised when a GitHub API call fails or returns something unusable."""


class GitHubFile:
    """A file's current content + the blob sha GitHub needs to accept an update to it."""

    def __init__(self, path: str, content: str, sha: str):
        self.path = path
        self.content = content
        self.sha = sha


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


async def _github_request(method: str, url: str, *, token: str | None, json: dict | None = None) -> httpx.Response:
    """
    Shared request helper: opens a `github.api` span (nested under whichever
    graph node called in -- see app/services/telemetry.py's module
    docstring on context propagation), and translates transport-level
    failures into `GitHubError` uniformly for every endpoint below.

    `token` is optional -- GitHub's read endpoints work unauthenticated
    against public repos (rate-limited to 60 req/hour instead of 5,000,
    which is plenty for triaging one ticket at a time). Write endpoints
    (branch/commit/PR) will simply 404/403 without one; `propose_fix_node`
    checks for a token before attempting those and skips cleanly if unset.
    """
    settings = get_settings()
    tracer = get_tracer()
    with tracer.start_as_current_span(f"github.api.{method.lower()}") as span:
        span.set_attribute("github.url", url)
        span.set_attribute("github.authenticated", token is not None)
        try:
            async with httpx.AsyncClient(timeout=settings.github_timeout_seconds) as client:
                response = await client.request(
                    method, url, headers=_headers(token) if token else {"Accept": "application/vnd.github+json"}, json=json
                )
        except httpx.TimeoutException as exc:
            span.record_exception(exc)
            span.set_status(trace.Status(trace.StatusCode.ERROR, "timeout"))
            raise GitHubError(f"GitHub API request to {url} timed out after {settings.github_timeout_seconds}s") from exc
        except httpx.ConnectError as exc:
            span.record_exception(exc)
            span.set_status(trace.Status(trace.StatusCode.ERROR, "connect_error"))
            raise GitHubError(f"Could not reach GitHub API at {url}") from exc

        span.set_attribute("http.status_code", response.status_code)
        if response.status_code >= 400:
            span.set_status(trace.Status(trace.StatusCode.ERROR, f"http_{response.status_code}"))
            raise GitHubError(
                f"GitHub API {method} {url} failed with {response.status_code}: {response.text[:300]}"
            )
        span.set_status(trace.Status(trace.StatusCode.OK))
        return response


async def get_file_content(owner: str, repo: str, path: str, *, ref: str | None = None) -> GitHubFile:
    """
    Fetch a file's current content + blob sha from the repo -- the sha is
    required by `commit_file_update` below (GitHub's Contents API demands
    it on every update, as an optimistic-concurrency check that the caller
    is editing the version it thinks it is).

    `ref` defaults to the repo's default branch if omitted. No auth
    required for a public repo.
    """
    settings = get_settings()
    url = f"{settings.github_api_base_url}/repos/{owner}/{repo}/contents/{path}"
    if ref:
        url += f"?ref={ref}"

    response = await _github_request("GET", url, token=settings.github_token)
    body = response.json()

    try:
        encoded_content = body["content"]
        sha = body["sha"]
    except (KeyError, TypeError) as exc:
        raise GitHubError(f"Unexpected GitHub contents response shape for {path}: {body!r}") from exc

    # The Contents API always base64-encodes file content (with embedded
    # newlines every 60 chars, which b64decode handles transparently).
    try:
        content = base64.b64decode(encoded_content).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise GitHubError(f"Could not decode {path} as UTF-8 text (binary file?)") from exc

    return GitHubFile(path=path, content=content, sha=sha)


async def find_file_by_basename(owner: str, repo: str, basename: str, *, ref: str) -> str | None:
    """
    Resolve a bare filename to its real repo-relative path by listing the
    full tree and matching on basename. Returns the first match, or None.

    Exists because a browser stack trace only ever names the *served*
    filename (e.g. `index.js`, whatever the dev server/bundler exposed it
    as) -- not its path within the repo (`pages/index.js`) -- so a direct
    `get_file_content(affected_file)` call routinely 404s even when the
    file genuinely exists. Full source-map resolution would solve this
    properly; this basename search is the pragmatic stand-in (see
    README's "deliberately out of scope" list) -- good enough for a repo
    small enough that basenames are unique, which covers this project's
    scope even though it is not a generally sound assumption at real scale.
    """
    settings = get_settings()
    url = f"{settings.github_api_base_url}/repos/{owner}/{repo}/git/trees/{ref}?recursive=1"
    response = await _github_request("GET", url, token=settings.github_token)
    body = response.json()
    for entry in body.get("tree", []):
        if entry.get("type") == "blob" and entry.get("path", "").rsplit("/", 1)[-1] == basename:
            return entry["path"]
    return None


async def get_branch_head_sha(owner: str, repo: str, branch: str) -> str:
    """The commit sha a branch currently points at -- what a new branch is created *from*."""
    settings = get_settings()
    url = f"{settings.github_api_base_url}/repos/{owner}/{repo}/git/ref/heads/{branch}"
    response = await _github_request("GET", url, token=settings.github_token)
    body = response.json()
    try:
        return body["object"]["sha"]
    except (KeyError, TypeError) as exc:
        raise GitHubError(f"Unexpected GitHub ref response shape for {branch}: {body!r}") from exc


async def create_branch(owner: str, repo: str, *, new_branch: str, from_sha: str) -> None:
    """
    Create `new_branch` pointing at `from_sha`. Requires write access
    (`settings.github_token` set with Contents: write on this repo).

    If the branch already exists, GitHub returns 422 -- `propose_fix_node`
    includes the ticket's short UUID in the branch name specifically so
    this collision is not expected in normal operation, but a stale retry
    of the same ticket would still hit it, which surfaces as a normal
    `GitHubError` rather than silently overwriting an existing branch.
    """
    settings = get_settings()
    url = f"{settings.github_api_base_url}/repos/{owner}/{repo}/git/refs"
    await _github_request(
        "POST", url, token=settings.github_token,
        json={"ref": f"refs/heads/{new_branch}", "sha": from_sha},
    )


async def commit_file_update(
    owner: str, repo: str, *, path: str, new_content: str, sha: str, branch: str, message: str
) -> None:
    """Commit `new_content` as the new version of `path` on `branch`. `sha` must be the blob sha being replaced."""
    settings = get_settings()
    url = f"{settings.github_api_base_url}/repos/{owner}/{repo}/contents/{path}"
    encoded = base64.b64encode(new_content.encode("utf-8")).decode("ascii")
    await _github_request(
        "PUT", url, token=settings.github_token,
        json={"message": message, "content": encoded, "sha": sha, "branch": branch},
    )


async def open_pull_request(owner: str, repo: str, *, title: str, body: str, head: str, base: str) -> str:
    """Open a PR from `head` -> `base` and return its HTML URL. Never merges -- see module docstring."""
    settings = get_settings()
    url = f"{settings.github_api_base_url}/repos/{owner}/{repo}/pulls"
    response = await _github_request(
        "POST", url, token=settings.github_token,
        json={"title": title, "body": body, "head": head, "base": base},
    )
    body_json = response.json()
    try:
        return body_json["html_url"]
    except (KeyError, TypeError) as exc:
        raise GitHubError(f"Unexpected GitHub pulls response shape: {body_json!r}") from exc
