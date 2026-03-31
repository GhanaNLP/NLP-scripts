import os
import asyncio
import threading
import pandas as pd
from google import genai
from google.genai import types
import random

# ==========================================
# 1. Configuration
# ==========================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "GEMINI_API_KEY_HERE")
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"

INPUT_CSV = "cleaned_twi_dataset.csv"
TWI_COLUMN = "twi"
OUTPUT_FILE = "translated_output.csv"

BATCH_SIZE = 30           # texts per API call
PARALLEL_REQUESTS = 20   # concurrent API calls
CSV_CHUNKSIZE = 20_000    # rows read from disk at a time — keeps memory low

DELIMITER = "---"        # separates translations in the response

client = genai.Client(api_key=GEMINI_API_KEY)

# ==========================================
# 2. Build resume index
# ==========================================
def load_processed_indices() -> set:
    if not os.path.exists(OUTPUT_FILE):
        return set()
    try:
        done = pd.read_csv(OUTPUT_FILE, usecols=["original_index"])
        return set(done["original_index"].dropna().astype(int).tolist())
    except Exception:
        return set()

print("Scanning output file for already-processed rows...")
processed_indices = load_processed_indices()
print(f"  {len(processed_indices)} rows already translated.")

# ==========================================
# 3. Output file — write header if new
# ==========================================
write_lock = threading.Lock()

def init_output_file(sample_df: pd.DataFrame):
    if not os.path.exists(OUTPUT_FILE):
        header = sample_df.iloc[0:0].copy()
        header.insert(0, "original_index", None)
        header["english_translation"] = None
        header.to_csv(OUTPUT_FILE, index=False)
        print("Created output file with header.")

def append_rows(out_df: pd.DataFrame):
    with write_lock:
        out_df.to_csv(OUTPUT_FILE, mode="a", header=False, index=False)

# ==========================================
# 4. Prompt builder
# ==========================================
def build_prompt(texts: list[str]) -> str:
    numbered = "\n\n".join(f"{i+1}.\n{t}" for i, t in enumerate(texts))
    return f"""You are an expert translator specialising in Twi (Akan) to English.
Translate each of the following {len(texts)} Twi texts into natural, fluent English.

Rules:
- Output EXACTLY {len(texts)} translations, in the same order as the input.
- Separate each translation with a blank line followed by "{DELIMITER}" followed by another blank line.
- Do NOT include numbers, labels, or any extra commentary — only the translated text for each item.
- Preserve the meaning and tone of the original.

Example output format for 3 items:
First translation here.

{DELIMITER}

Second translation here.

{DELIMITER}

Third translation here.

Input:
{numbered}"""

# ==========================================
# 5. Response parser
# ==========================================
def parse_response(response_text: str, expected_count: int) -> list[str]:
    parts = [p.strip() for p in response_text.split(DELIMITER)]
    # Strip any empty parts that may appear at start/end
    translations = [p for p in parts if p]

    if len(translations) != expected_count:
        raise ValueError(
            f"Expected {expected_count} translations, got {len(translations)}.\n"
            f"Response snippet: {response_text[:400]}"
        )
    return translations

# ==========================================
# 6. Raw API call (single prompt, no parsing)
# ==========================================
async def _call_api_once(prompt: str) -> str:
    """Single API attempt, no retry, no semaphore. Raises on any failure."""
    loop = asyncio.get_running_loop()

    def _call():
        chunks = []
        for chunk in client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                )
            ],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
            ),
        ):
            if chunk.text:
                chunks.append(chunk.text)
        return "".join(chunks)

    result = await loop.run_in_executor(None, _call)
    if not result.strip():
        raise ValueError("Empty response from API.")
    return result.strip()


async def call_api(prompt: str, semaphore: asyncio.Semaphore) -> str:
    """Acquires semaphore only during the HTTP call itself.
    Retry sleeps happen outside the semaphore so other tasks can proceed."""
    attempt = 0
    last_error = None
    while True:
        attempt += 1
        async with semaphore:   # hold slot only for the actual network call
            try:
                return await _call_api_once(prompt)
            except Exception as e:
                last_error = e
        # Semaphore released before sleeping — other tasks are free to run
        wait = min(2 * (2 ** min(attempt, 6)) + random.uniform(0, 2), 60)
        print(f"  [Retry {attempt}] {last_error} — waiting {wait:.1f}s...")
        await asyncio.sleep(wait)

# ==========================================
# 6b. One-by-one fallback (single text prompt)
# ==========================================
def build_single_prompt(text: str) -> str:
    return f"""You are an expert translator specialising in Twi (Akan) to English.
Translate the following Twi text into natural, fluent English.

Rules:
- Output ONLY the English translation.
- Do NOT add any commentary, explanations, or extra text.
- Preserve the meaning and tone of the original.

Twi:
{text}"""

async def translate_one_by_one(texts: list[str], semaphore: asyncio.Semaphore) -> list[str]:
    """Translate each text individually — used as fallback when batch parsing fails."""
    print(f"  [Fallback] Switching to one-by-one for {len(texts)} texts...")
    tasks = [call_api(build_single_prompt(t), semaphore) for t in texts]
    return await asyncio.gather(*tasks)

# ==========================================
# 7. Batch API call — with one-by-one fallback
# ==========================================
BATCH_PARSE_MAX_RETRIES = 3  # attempts before giving up on batch and going one-by-one

async def translate_batch(texts: list[str], semaphore: asyncio.Semaphore) -> list[str]:
    prompt = build_prompt(texts)
    attempt = 0
    while True:
        attempt += 1
        try:
            full_response = await call_api(prompt, semaphore)
            return parse_response(full_response, len(texts))
        except ValueError as e:
            # Parse error — model didn't follow delimiter instructions
            if attempt >= BATCH_PARSE_MAX_RETRIES:
                print(f"  [Fallback] Batch parse failed {attempt} times: {e}")
                return await translate_one_by_one(texts, semaphore)
            print(f"  [Parse Retry {attempt}/{BATCH_PARSE_MAX_RETRIES}] {e} — retrying batch...")
            await asyncio.sleep(2)

# ==========================================
# 7. Process one batch of rows
# ==========================================
async def process_batch(batch_df: pd.DataFrame, semaphore: asyncio.Semaphore):
    texts = [str(t).strip() for t in batch_df[TWI_COLUMN].tolist()]
    translations = await translate_batch(texts, semaphore)

    out = batch_df.copy()
    out.insert(0, "original_index", batch_df.index.tolist())
    out["english_translation"] = translations
    append_rows(out)

    first, last = batch_df.index[0], batch_df.index[-1]
    print(f"  ✓ Rows {first}–{last} ({len(batch_df)} items) written.")

# ==========================================
# 8. Main — streams CSV in chunks
# ==========================================
async def main():
    semaphore = asyncio.Semaphore(PARALLEL_REQUESTS)

    print("Counting rows in input file...")
    total_rows = sum(len(c) for c in pd.read_csv(INPUT_CSV, usecols=[TWI_COLUMN], chunksize=CSV_CHUNKSIZE))
    print(f"Total rows in input: {total_rows:,}")

    header_written = os.path.exists(OUTPUT_FILE)
    global_row = 0
    rows_to_do = total_rows - len(processed_indices)
    rows_done = 0

    print(f"Rows remaining: {rows_to_do:,}\n")

    for file_chunk in pd.read_csv(INPUT_CSV, chunksize=CSV_CHUNKSIZE):
        file_chunk = file_chunk.reset_index(drop=True)
        file_chunk.index = range(global_row, global_row + len(file_chunk))
        global_row += len(file_chunk)

        if not header_written:
            init_output_file(file_chunk)
            header_written = True

        todo = file_chunk[~file_chunk.index.isin(processed_indices)]

        if todo.empty:
            continue

        # Split into batches of BATCH_SIZE
        indices = todo.index.tolist()
        batches = [indices[i: i + BATCH_SIZE] for i in range(0, len(indices), BATCH_SIZE)]

        tasks = [process_batch(todo.loc[b], semaphore) for b in batches]
        await asyncio.gather(*tasks)

        rows_done += len(todo)
        pct = rows_done / rows_to_do * 100 if rows_to_do else 100
        print(f"  [{rows_done:,}/{rows_to_do:,} — {pct:.1f}%] up to row {global_row:,}\n")

    print(f"\n{'='*55}")
    print(f"All done! Translations saved to: {OUTPUT_FILE}")
    print(f"{'='*55}")

# ==========================================
# 9. Entry point
# ==========================================
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Interrupted] Progress saved row-by-row. Re-run to resume.")
