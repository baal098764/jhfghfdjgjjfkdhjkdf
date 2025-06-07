import re
import json
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Dict
import httpx
import aiohttp
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# ------------------------------------------------------------------------------  
# CONFIGURATION & FILE PATHS
# ------------------------------------------------------------------------------  
SERVICE_URL = "https://lockify-a1ef.onrender.com"
API_TOKEN = ""  # (Used for premium endpoints.)

# SINGLE FastAPI instance (ensures /static stays mounted)
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
logging.basicConfig(level=logging.INFO)

# File paths for persistence
WHITELIST_FILE = Path("wlist.json")   # (Reserved for future use)
USAGE_FILE     = Path("usage.json")   # (Reserved for future use)
STATS_FILE     = Path("stats.json")   # Stores {"total_links_bypassed": <N>}

# Ensure JSON files exist
for fpath in (WHITELIST_FILE, USAGE_FILE, STATS_FILE):
    if not fpath.exists():
        if fpath is STATS_FILE:
            fpath.write_text(json.dumps({"total_links_bypassed": 0}, indent=2))
        else:
            fpath.write_text("{}")

# ------------------------------------------------------------------------------  
# SUPPORTED WEBSITES / PAGES CONFIGURATION
# ------------------------------------------------------------------------------  
SUPPORTED_WEBSITES: Dict[str, str] = {
    "supported": "supported.html",         # The “Supported Sites” page
    "premium":   "premium.html",           # The “Premium API” info page (HTML)
    "terms":     "terms-of-service.html",  # The “Terms of Service” page (HTML)
}

# ------------------------------------------------------------------------------  
# HELPERS: JSON I/O
# ------------------------------------------------------------------------------  
def load_json(path: Path) -> Dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def save_json(path: Path, data: Dict):
    path.write_text(json.dumps(data, indent=2))

# ------------------------------------------------------------------------------  
# HELPERS: LOCKR.SO INTERACTIONS
# ------------------------------------------------------------------------------  
def parse_locker_id(url: str) -> str:
    """
    Extract the locker ID from a lockr.so URL.
    Raises ValueError if the URL is invalid.
    """
    match = re.search(r"lockr\.so/([A-Za-z0-9]+)", url)
    if match:
        return match.group(1)
    raise ValueError("Invalid lockr.so URL. Expected format: https://lockr.so/<locker_id>")

async def get_view_data(session: aiohttp.ClientSession, locker_id: str):
    """
    Calls https://lockr.so/api/v1/lockers/{locker_id}/view
    and returns (token, user_id, task_ids, task_urls) if successful.
    Otherwise returns (None, None, [], []).
    """
    view_url = f"https://lockr.so/api/v1/lockers/{locker_id}/view"
    headers = {
        "User-Agent":    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:139.0) Gecko/20100101 Firefox/139.0",
        "Accept":        "*/*",
        "Accept-Language":"en-US,en;q=0.5",
        "Accept-Encoding":"gzip, deflate, br, zstd",
        "Referer":       f"https://lockr.so/{locker_id}",
        "Content-Type":  "application/json",
        "DNT":           "1",
        "Connection":    "keep-alive",
        "Sec-Fetch-Dest":"empty",
        "Sec-Fetch-Mode":"cors",
        "Sec-Fetch-Site":"same-origin",
        "Priority":      "u=4",
        "Pragma":        "no-cache",
        "Cache-Control": "no-cache",
        "TE":            "trailers",
    }

    try:
        async with session.get(view_url, headers=headers) as response:
            response.raise_for_status()
            text = await response.text()
            try:
                data = json.loads(text)
            except Exception as e:
                logging.error(f"Error parsing JSON from /view: {e}")
                return None, None, [], []

            view_data = data.get("data", {})
            token = view_data.get("token")
            user_id = view_data.get("user_id")
            tasks = view_data.get("tasks", [])
            task_ids = [task.get("id") for task in tasks if task.get("id")]
            task_urls = [task.get("task_url", "") for task in tasks]
            if token and user_id is not None and task_ids:
                return token, user_id, task_ids, task_urls

            return None, None, [], []
    except Exception as e:
        logging.error(f"Error calling /view: {e}")
        return None, None, [], []

async def validate_task(session: aiohttp.ClientSession, locker_id: str, token: str) -> bool:
    """
    Calls https://lockr.so/api/v1/lockers/{locker_id}/task?token={token}
    Returns True if "success" is True, else False.
    """
    task_url = f"https://lockr.so/api/v1/lockers/{locker_id}/task?token={token}"
    headers = {
        "User-Agent":    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:139.0) Gecko/20100101 Firefox/139.0",
        "Accept":        "*/*",
        "Accept-Language":"en-US,en;q=0.5",
        "Accept-Encoding":"gzip, deflate, br, zstd",
        "Referer":       f"https://lockr.so/{locker_id}",
        "Content-Type":  "application/json",
        "DNT":           "1",
        "Connection":    "keep-alive",
        "Sec-Fetch-Dest":"empty",
        "Sec-Fetch-Mode":"cors",
        "Sec-Fetch-Site":"same-origin",
        "Priority":      "u=4",
        "Pragma":        "no-cache",
        "Cache-Control": "no-cache",
        "TE":            "trailers",
    }

    try:
        async with session.get(task_url, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()
            return data.get("data", {}).get("success", False)
    except Exception as e:
        logging.error(f"Error in validate_task: {e}")
        return False

async def submit_task(session: aiohttp.ClientSession, locker_id: str, token: str, task_id: str) -> bool:
    """
    POST [f"{token}::{task_id}"] to https://lockr.so/{locker_id}, wait ~23s.
    Return True if POST status < 400, False otherwise.
    """
    post_url = f"https://lockr.so/{locker_id}"
    headers = {
        "User-Agent":    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:139.0) Gecko/20100101 Firefox/139.0",
        "Accept":        "text/x-component",
        "Accept-Language":"en-US,en;q=0.5",
        "Accept-Encoding":"gzip, deflate, br, zstd",
        "Referer":       f"https://lockr.so/{locker_id}",
        "Next-Action":            "7f9561be9d96c2dc617a56dca220560e77a0c52138",
        "Next-Router-State-Tree": f"%5B%22%22%2C%7B%22children%22%3A%5B%5B%22lockerId%22%2C%22{locker_id}%22%2C%22d%22%5D%2C%7B%22children%22%3A%5B%22__PAGE__%3F%7B%5C%22ip%5C%22%3A%5C%22193.43.135.42%5C%22%2C%5C%22country%5C%22%3A%5C%22US%5C%22%2C%5C%22referer%5C%22%3A%5C%22xx%5C%22%7D%22%2C%7B%7D%2C%22%2F{locker_id}%22%2C%22refresh%22%5D%7D%5D%7D%2Cnull%2Cnull%2Ctrue%5D",
        "Content-Type":  "text/plain;charset=UTF-8",
        "Origin":        "https://lockr.so",
        "DNT":           "1",
        "Connection":    "keep-alive",
        "Sec-Fetch-Dest":"empty",
        "Sec-Fetch-Mode":"cors",
        "Sec-Fetch-Site":"same-origin",
        "Priority":      "u=0",
        "Pragma":        "no-cache",
        "Cache-Control": "no-cache",
        "TE":            "trailers",
    }
    payload = json.dumps([f"{token}::{task_id}"])
    try:
        async with session.post(post_url, headers=headers, data=payload) as response:
            response.raise_for_status()
            await asyncio.sleep(23)  # wait ~23 seconds before unlocking
            return True
    except Exception as e:
        logging.error(f"Error in submit_task: {e}")
        return False

async def attempt_unlock(session: aiohttp.ClientSession, locker_id: str, token: str) -> Optional[str]:
    """
    GET https://lockr.so/api/v1/lockers/{locker_id}/unlock?token={token}.
    Return the "target" URL if present, else None.
    """
    unlock_url = f"https://lockr.so/api/v1/lockers/{locker_id}/unlock?token={token}"
    headers = {
        "User-Agent":    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:139.0) Gecko/20100101 Firefox/139.0",
        "Accept":        "*/*",
        "Accept-Language":"en-US,en;q=0.5",
        "Accept-Encoding":"gzip, deflate, br, zstd",
        "Referer":       f"https://lockr.so/{locker_id}",
        "Content-Type":  "application/json",
        "DNT":           "1",
        "Connection":    "keep-alive",
        "Sec-Fetch-Dest":"empty",
        "Sec-Fetch-Mode":"cors",
        "Sec-Fetch-Site":"same-origin",
        "Priority":      "u=4",
        "Pragma":        "no-cache",
        "Cache-Control": "no-cache",
        "TE":            "trailers",
    }
    try:
        async with session.get(unlock_url, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()
            return data.get("data", {}).get("target")
    except Exception as e:
        logging.error(f"Error in attempt_unlock: {e}")
        return None

async def resolve_lockr_link(
    session: aiohttp.ClientSession,
    locker_id: str,
    token: str,
    task_ids: List[str]
) -> Optional[str]:
    """
    Iterate through each task_id:
      • If submit_task(...) fails → try attempt_unlock(...) immediately
      • If submit_task(...) succeeds → try validate_task(...); if validation fails → attempt_unlock(...)
    At the end, do one final attempt_unlock(...) if no target was returned yet.
    """
    for task_id in task_ids:
        posted = await submit_task(session, locker_id, token, task_id)
        if not posted:
            unc = await attempt_unlock(session, locker_id, token)
            if unc:
                return unc
            continue

        valid = await validate_task(session, locker_id, token)
        if not valid:
            unc = await attempt_unlock(session, locker_id, token)
            if unc:
                return unc
            continue

    # Final unlock attempt if none of the tasks produced a target earlier
    return await attempt_unlock(session, locker_id, token)

# ------------------------------------------------------------------------------  
# HELPERS: STATISTICS
# ------------------------------------------------------------------------------  
def increment_total_bypassed() -> int:
    """
    Read stats.json, increment "total_links_bypassed" by 1, write back, return new total.
    """
    stats = load_json(STATS_FILE)
    total = stats.get("total_links_bypassed", 0) + 1
    stats["total_links_bypassed"] = total
    save_json(STATS_FILE, stats)
    return total

def get_stats() -> Dict:
    """
    Return the entire stats object from stats.json, e.g. {"total_links_bypassed": 12345}.
    """
    return load_json(STATS_FILE)

# ------------------------------------------------------------------------------  
# FASTAPI MODELS
# ------------------------------------------------------------------------------  
class ResolveRequest(BaseModel):
    url: str

class ResolveResponse(BaseModel):
    success: bool
    target_url: Optional[str] = None
    message: Optional[str] = None

class StatsResponse(BaseModel):
    total_links_bypassed: int

# ------------------------------------------------------------------------------  
# FASTAPI ROUTES + JINJA2 TEMPLATES
# ------------------------------------------------------------------------------  
@app.on_event("startup")
async def schedule_ping_task():
    async def ping_loop():
        async with httpx.AsyncClient(timeout=5) as client:
            while True:
                try:
                    resp = await client.get(f"{SERVICE_URL}/ping")
                    if resp.status_code != 200:
                        print(f"Health ping returned {resp.status_code}")
                except Exception as e:
                    print(f"External ping failed: {e!r}")
                await asyncio.sleep(120)
    asyncio.create_task(ping_loop())

@app.get("/ping")
async def ping():
    return {"status": "alive"}

@app.get("/", response_class=HTMLResponse)
async def bypass_page(request: Request):
    return templates.TemplateResponse("byp.html", {"request": request})

@app.post("/resolve", response_model=ResolveResponse)
async def resolve_link(req: ResolveRequest):
    raw_url = req.url.strip()
    try:
        locker_id = parse_locker_id(raw_url)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid lockr.so URL format.")

    async with aiohttp.ClientSession() as session:
        token, api_user_id, task_ids, _ = await get_view_data(session, locker_id)
        if not token or api_user_id is None or not task_ids:
            return ResolveResponse(
                success=False,
                message=(
                    "⚠️ Failed to fetch view data. Possible reasons:\n"
                    "- Cloudflare or anti-bot blocking the request.\n"
                    "- Lockr.so’s API changed or is temporarily down.\n"
                    "Please try again in a few seconds."
                )
            )

        final_url = await resolve_lockr_link(session, locker_id, token, task_ids)
        if final_url:
            increment_total_bypassed()
            return ResolveResponse(success=True, target_url=final_url)
        else:
            return ResolveResponse(
                success=False,
                message=(
                    "❌ Failed to resolve the link. Possible reasons:\n"
                    "- The server required an ad interaction or additional cookies.\n"
                    "- Tasks could not be validated quickly enough.\n"
                    "Please wait a few seconds and try again."
                )
            )

@app.get("/stats", response_model=StatsResponse)
async def read_stats():
    total = get_stats().get("total_links_bypassed", 0)
    return StatsResponse(total_links_bypassed=total)

@app.get("/supported", response_class=HTMLResponse)
async def render_supported_page(request: Request):
    return templates.TemplateResponse("supported.html", {"request": request})

@app.get("/premium", response_class=HTMLResponse)
async def render_premium_page(request: Request):
    return templates.TemplateResponse("premium.html", {"request": request})

@app.get("/terms", response_class=HTMLResponse)
async def render_terms_page(request: Request):
    return templates.TemplateResponse("terms-of-service.html", {"request": request})

@app.get("/sites/{site_key}", response_class=HTMLResponse)
async def render_supported_site(request: Request, site_key: str):
    template_name = SUPPORTED_WEBSITES.get(site_key)
    if not template_name:
        raise HTTPException(status_code=404, detail=f"Site '{site_key}' not supported.")
    return templates.TemplateResponse(template_name, {"request": request})

@app.get("/api/premium/{site_key}", response_class=HTMLResponse)
async def premium_render(
    request: Request,
    site_key: str,
    x_api_key: Optional[str] = Header(None)
):
    if x_api_key != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid API key.")

    template_name = SUPPORTED_WEBSITES.get(site_key)
    if not template_name:
        raise HTTPException(status_code=404, detail=f"Site '{site_key}' not supported.")

    return templates.TemplateResponse(template_name, {"request": request})

@app.post("/api/premium/render", response_class=HTMLResponse)
async def fetch_and_render_external(
    request: Request,
    target_url: str,
    x_api_key: Optional[str] = Header(None)
):
    if x_api_key != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid API key.")

    if not target_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid URL provided.")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(target_url) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "")
                html_body = await resp.text()

                if "text/html" not in content_type:
                    raise HTTPException(
                        status_code=415,
                        detail="Unsupported Media Type: only HTML pages are allowed for rendering."
                    )
                return HTMLResponse(content=html_body)
        except aiohttp.ClientResponseError as e:
            logging.error(f"Error fetching external URL '{target_url}': {e}")
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch the URL: {e.status} {e.message}"
            )
        except Exception as e:
            logging.error(f"Unexpected error fetching '{target_url}': {e}")
            raise HTTPException(status_code=500, detail="Internal Server Error")
