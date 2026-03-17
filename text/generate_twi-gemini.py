#!/usr/bin/env python3
"""
Automated Twi Text Generator using Gemini API
==============================================
Generates Twi text from English paragraphs using Gemini API with async processing.
Includes validation, infinite retry logic (never skips rows), and resume support.

Usage:
    python generate_twi.py --end-index 200000
    python generate_twi.py AIza... --end-index 200000
    python generate_twi.py --end-index 200000 --workers 20 --rpm 120

Notes:
    - Rows are NEVER skipped — retried indefinitely until success
    - Handles internet drops with exponential backoff
    - Safe to kill and restart at any time (resumes from last saved point)
"""

import asyncio
import argparse
import json
import re
import time
from pathlib import Path
from datetime import datetime

# ════════════════════════════════════════════
#  SET YOUR GEMINI API KEY HERE
# ════════════════════════════════════════════
GEMINI_API_KEY = "YOUR-GEMINI-API-KEY-HERE"
# Get your key from: https://aistudio.google.com/app/apikey
# ════════════════════════════════════════════

# Model
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"

# ─── Rate limits ───────────────────────────────────────────────────────────────
# gemini-2.0-flash-lite free tier: 30 RPM, 1500 RPD
# Pay-as-you-go tier:             1000 RPM
# Set conservatively — code will back off automatically on 429s
MAX_CONCURRENT    = 15          # parallel in-flight requests
REQUESTS_PER_MINUTE = 120        # target RPM (set to ~80% of your actual quota)
RATE_LIMIT_DELAY  = 60 / REQUESTS_PER_MINUTE

# ─── Validation thresholds ────────────────────────────────────────────────────
MIN_CHARS = 2000
MAX_CHARS = 18000

# ─── Retry / backoff (NEVER skips — retries forever) ─────────────────────────
INITIAL_BACKOFF   = 5      # seconds before first retry
MAX_BACKOFF       = 300    # cap at 5 minutes between retries
BACKOFF_FACTOR    = 2      # exponential multiplier
RATE_LIMIT_PAUSE  = 60     # extra pause when a 429 is received

# ─── Prompts ──────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE_1 = """Generate Twi text in 4 parts (each approximately 500 words):
1. Monologue (ɔkasa biako) - passionate, personal perspective
2. Narrative (abakɔsɛm) - journalistic, structured account  
3. Dialogue (nkɔmmɔdie) - conversational debate between 2 people
4. Storyful (anansesɛm) - dramatic, metaphorical storytelling

Use this English paragraph as inspiration:
{paragraph}

Guidelines:
- Mix English words naturally as done in spoken Twi (not all English words have good Twi translations)
- Write in natural, flowing Twi
- Each part should be substantial (~500 words)
- Return ONLY the Twi text, no preamble, no markdown formatting, no section headers"""

PROMPT_TEMPLATE_2 = """Create comprehensive Twi text with 4 distinct sections (each ~500 words total):
- A passionate monologue
- A journalistic narrative  
- A conversational dialogue
- A dramatic story

Based on this topic:
{paragraph}

Important:
- Write naturally in Twi, mixing English words where appropriate
- Make each section substantial and complete
- Total output should be around 2000 words
- Return plain Twi text only, no formatting or labels"""


# ══════════════════════════════════════════════════════════════════════════════
#  Text helpers
# ══════════════════════════════════════════════════════════════════════════════

def remove_consecutive_repetitions(text):
    """Remove consecutive repeated sentences."""
    parts = re.split(r'(\s*[.!?\n]+\s*)', text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        s = parts[i].strip()
        delim = parts[i + 1] if i + 1 < len(parts) else " "
        if s:
            sentences.append((s, delim))
    if parts and parts[-1].strip():
        sentences.append((parts[-1].strip(), ""))

    if not sentences:
        return text, 0

    cleaned = [sentences[0]]
    removed = 0
    i = 1
    while i < len(sentences):
        matched = False
        max_block = (len(sentences) - i) // 2
        for k in range(min(max_block, 10), 0, -1):
            block     = [s for s, _ in sentences[i:i + k]]
            prev_block = [s for s, _ in cleaned[-k:]] if len(cleaned) >= k else None
            if prev_block and [s.lower() for s in block] == [s.lower() for s in prev_block]:
                removed += k
                i += k
                matched = True
                break
        if not matched:
            cleaned.append(sentences[i])
            i += 1

    result = " ".join(s + d for s, d in cleaned).strip()
    return result, removed


def validate_text(text):
    """Validate Twi text. Returns (cleaned, char_count, reps_removed, is_valid, error)."""
    if not text or not text.strip():
        return "", 0, 0, False, "Empty response"

    cleaned, n_removed = remove_consecutive_repetitions(text.strip())
    char_count = len(cleaned)

    if char_count < MIN_CHARS:
        return cleaned, char_count, n_removed, False, f"Too short: {char_count:,} chars (need ≥{MIN_CHARS:,})"
    if char_count > MAX_CHARS:
        return cleaned, char_count, n_removed, False, f"Too long: {char_count:,} chars (need ≤{MAX_CHARS:,})"

    return cleaned, char_count, n_removed, True, None


# ══════════════════════════════════════════════════════════════════════════════
#  Core generator
# ══════════════════════════════════════════════════════════════════════════════

class GeminiGenerator:
    def __init__(self, api_key, output_file="twi_texts.jsonl", progress_file="progress.json"):
        self.api_key       = api_key
        self.output_file   = Path(output_file)
        self.progress_file = Path(progress_file)
        self.completed_indices: set[int] = set()
        self.semaphore     = asyncio.Semaphore(MAX_CONCURRENT)
        self.rate_limiter  = asyncio.Lock()
        self.last_request_time = 0.0
        self._file_lock    = asyncio.Lock()   # guards jsonl writes
        self._load_progress()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load_progress(self):
        if self.output_file.exists():
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        self.completed_indices.add(data['index'])
                    except Exception:
                        pass
        print(f"  Resuming: {len(self.completed_indices):,} rows already done.")

    async def _append_result(self, result: dict):
        async with self._file_lock:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            self.completed_indices.add(result['index'])

    # ── rate limiter ──────────────────────────────────────────────────────────

    async def _rate_limit(self):
        async with self.rate_limiter:
            now = time.monotonic()
            wait = RATE_LIMIT_DELAY - (now - self.last_request_time)
            if wait > 0:
                await asyncio.sleep(wait)
            self.last_request_time = time.monotonic()

    # ── single API call ───────────────────────────────────────────────────────

    async def _call_gemini(self, prompt: str) -> str | None:
        """
        Make one Gemini API call.
        Returns the text on success, None on a non-retriable / content-blocked error,
        raises an exception on network / rate-limit errors so callers can retry.
        """
        import google.generativeai as genai

        await self._rate_limit()

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)

        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config={
                'temperature': 1.0,
                'top_p': 0.95,
                'top_k': 64,
                'max_output_tokens': 8192,
            }
        )

        if response.text:
            return response.text

        # Content was blocked — return None (will retry with alt prompt)
        return None

    # ── infinite retry loop ───────────────────────────────────────────────────

    async def _generate_forever(self, index: int, paragraph: str) -> dict:
        """
        Retries indefinitely until a valid Twi text is produced.
        Uses alternating prompts; backs off on errors; NEVER skips.
        """
        attempt    = 0
        backoff    = INITIAL_BACKOFF
        prompt_idx = 0
        templates  = [PROMPT_TEMPLATE_1, PROMPT_TEMPLATE_2]

        while True:
            attempt += 1
            prompt = templates[prompt_idx % 2].format(paragraph=paragraph)
            prompt_idx += 1

            try:
                response_text = await self._call_gemini(prompt)
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower():
                    wait = RATE_LIMIT_PAUSE + backoff
                    print(f"  [{index:06d}] ⚠ Rate-limited — sleeping {wait}s (attempt {attempt})")
                    await asyncio.sleep(wait)
                    # Don't grow backoff for rate limits — just keep the fixed pause
                else:
                    print(f"  [{index:06d}] ✗ Network/API error (attempt {attempt}): {e} — retry in {backoff}s")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * BACKOFF_FACTOR, MAX_BACKOFF)
                continue

            if response_text is None:
                print(f"  [{index:06d}] ⚠ Blocked/empty response (attempt {attempt}) — retrying in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * BACKOFF_FACTOR, MAX_BACKOFF)
                continue

            cleaned, char_count, reps_removed, is_valid, error = validate_text(response_text)

            if is_valid:
                backoff = INITIAL_BACKOFF   # reset backoff on success
                return {
                    'index':               index,
                    'source_paragraph':    paragraph,
                    'twi_text':            cleaned,
                    'char_count':          char_count,
                    'repetitions_removed': reps_removed,
                    'attempts':            attempt,
                    'model':               GEMINI_MODEL,
                    'timestamp':           datetime.now().isoformat(),
                }
            else:
                print(f"  [{index:06d}] ✗ Validation failed (attempt {attempt}): {error} — retrying in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * BACKOFF_FACTOR, MAX_BACKOFF)

    # ── per-row worker ────────────────────────────────────────────────────────

    async def _process_one(self, index: int, paragraph: str, counter: dict):
        if index in self.completed_indices:
            async with asyncio.Lock():
                counter['skipped'] += 1
            return

        async with self.semaphore:
            result = await self._generate_forever(index, paragraph)
            await self._append_result(result)
            async with asyncio.Lock():
                counter['done'] += 1
            total_done = len(self.completed_indices)
            print(f"  [{index:06d}] ✓ saved  ({result['char_count']:,} chars, "
                  f"attempt {result['attempts']})  |  total saved: {total_done:,}")

    # ── batch driver ──────────────────────────────────────────────────────────

    async def generate_batch(self, paragraphs: list[str], start_index: int = 0):
        pending = [
            (start_index + i, para)
            for i, para in enumerate(paragraphs)
            if (start_index + i) not in self.completed_indices
        ]
        already_done = len(paragraphs) - len(pending)

        print(f"\n{'═'*70}")
        print(f"  Model            : {GEMINI_MODEL}")
        print(f"  Total rows       : {len(paragraphs):,}")
        print(f"  Already complete : {already_done:,}")
        print(f"  To process       : {len(pending):,}")
        print(f"  Concurrency      : {MAX_CONCURRENT} workers")
        print(f"  Rate limit       : {REQUESTS_PER_MINUTE} RPM  ({RATE_LIMIT_DELAY:.2f}s gap)")
        print(f"  Skipping rows    : NEVER  (infinite retry)")
        print(f"{'═'*70}\n")

        counter = {'done': 0, 'skipped': already_done}

        tasks = [
            self._process_one(idx, para, counter)
            for idx, para in pending
        ]

        # Run with a simple progress ticker
        ticker_task = asyncio.create_task(
            self._progress_ticker(counter, len(paragraphs))
        )
        await asyncio.gather(*tasks)
        ticker_task.cancel()

        print(f"\n{'═'*70}")
        print(f"  Batch complete!")
        print(f"  Total saved : {len(self.completed_indices):,}")
        print(f"  Output file : {self.output_file}")
        print(f"{'═'*70}\n")

    async def _progress_ticker(self, counter: dict, total: int):
        """Print a progress line every 30 seconds."""
        start = time.monotonic()
        try:
            while True:
                await asyncio.sleep(30)
                elapsed  = time.monotonic() - start
                done     = len(self.completed_indices)
                rate     = done / (elapsed / 60) if elapsed > 0 else 0
                remaining = total - done
                eta_min  = remaining / rate if rate > 0 else float('inf')
                print(f"\n  ── Progress: {done:,}/{total:,}  "
                      f"({rate:.1f} rows/min)  "
                      f"ETA ≈ {eta_min:.0f} min ──\n")
        except asyncio.CancelledError:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    global MAX_CONCURRENT, REQUESTS_PER_MINUTE, RATE_LIMIT_DELAY

    parser = argparse.ArgumentParser(
        description="Generate Twi text using Gemini API (never skips rows)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("api_key", nargs='?', default=None,
                        help="Gemini API key (optional — uses hardcoded key if omitted)")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Start index in dataset (default: 0)")
    parser.add_argument("--end-index", type=int, default=250000,
                        help="End index in dataset (default: 250000)")
    parser.add_argument("--workers", type=int, default=None,
                        help=f"Max concurrent requests (default: {MAX_CONCURRENT})")
    parser.add_argument("--rpm", type=int, default=None,
                        help=f"Requests per minute (default: {REQUESTS_PER_MINUTE})")
    parser.add_argument("--dataset",
                        default="ghananlpcommunity/twi-english-paragraph-dataset_news",
                        help="HuggingFace dataset repo ID")
    parser.add_argument("--output", default="twi_texts.jsonl",
                        help="Output JSONL file (default: twi_texts.jsonl)")

    args = parser.parse_args()

    if args.workers:
        MAX_CONCURRENT = args.workers
    if args.rpm:
        REQUESTS_PER_MINUTE = args.rpm
        RATE_LIMIT_DELAY    = 60 / REQUESTS_PER_MINUTE

    api_key = args.api_key if args.api_key else GEMINI_API_KEY

    if not api_key or api_key == "AIza_YOUR_API_KEY_HERE":
        print("❌  No API key set.")
        print("    Edit GEMINI_API_KEY in the script, or pass it as the first argument.")
        print("    Get a key at: https://aistudio.google.com/app/apikey")
        return

    if not api_key.startswith("AIza"):
        print("⚠️   Warning: key doesn't start with 'AIza' — double-check it.")
        if input("Continue anyway? (y/N): ").lower() != 'y':
            return

    # ── dependency checks ──────────────────────────────────────────────────────
    try:
        import google.generativeai as genai
        print(f"✓  google-generativeai {genai.__version__}")
    except ImportError:
        print("❌  Missing: pip install google-generativeai")
        return

    # ── load dataset ───────────────────────────────────────────────────────────
    print("\n📂  Loading dataset...")
    try:
        from datasets import load_dataset

        print(f"    Repo: {args.dataset}")
        ds = load_dataset(args.dataset, split="train", trust_remote_code=True)
        df = ds.to_pandas()

        start = args.start_index
        end   = min(args.end_index, len(df))

        if "ENGLISH" not in df.columns:
            print(f"❌  No 'ENGLISH' column. Available: {list(df.columns)}")
            return

        paragraphs = df.iloc[start:end]["ENGLISH"].dropna().tolist()
        print(f"✓  Loaded {len(paragraphs):,} paragraphs (rows {start}–{end - 1})")

    except Exception as e:
        print(f"❌  Dataset load failed: {e}")
        print("    pip install datasets huggingface_hub pandas")
        print("    If private: huggingface-cli login")
        return

    # ── run ────────────────────────────────────────────────────────────────────
    generator = GeminiGenerator(api_key, output_file=args.output)
    await generator.generate_batch(paragraphs, start_index=start)


if __name__ == "__main__":
    asyncio.run(main())
