from rusted_chain import GeminiModel
import time
from statistics import mean
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
""" File to compare the performance of rusted_chain vs langchain-core with Gemini-2.5-Flash model since that is the only free one I could test"""
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

WARMUP = os.getenv("RC_BENCH_WARMUP", "true").lower() == "true"
REPEATS = int(os.getenv("RC_BENCH_REPEATS", "4"))
langChainGemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
rustedChainGemini = GeminiModel(model="gemini-2.5-flash")

def run_chain(chain_callable, prompt, repeats=1):
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        chain_callable(prompt)
        timings.append(time.perf_counter() - start)
    return timings

prompt = "What is a LIAR dataset?"

lang_chain = lambda p: langChainGemini.invoke(p)
rust_chain = lambda p: rustedChainGemini.invoke(p)

# warm-up
if WARMUP:
    print("Performing warm-up calls...")
    rust_chain(prompt)
    lang_chain(prompt)
# TO prevent rate limiting
time.sleep(3)
print("Performing real calls")
lang_times = run_chain(lang_chain, prompt, repeats=REPEATS)
rust_times = run_chain(rust_chain, prompt, repeats=REPEATS)

def summarize(label, ts):
    print(
        f"{label}: "
        f"mean={mean(ts)*1000:.1f} ms "
        f"min={min(ts)*1000:.1f} ms, max={max(ts)*1000:.1f} ms "
        f"(n={len(ts)})"
    )

summarize("rusted_chain", rust_times)
summarize("langchain", lang_times)