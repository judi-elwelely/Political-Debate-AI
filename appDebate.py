"""Political Debate LLM: Government vs Opposition (Groq + Gradio)."""
import os
import time
from typing import Dict, Generator, List

import gradio as gr
from dotenv import load_dotenv
from groq import APIConnectionError, APIStatusError, Groq, RateLimitError

load_dotenv()

MAX_ROUNDS = 10
DEFAULT_ROUNDS = 2
MAX_TOKENS_PER_TURN = 900

# Two different models currently listed in Groq's supported-model documentation.
GOV_MODEL = os.getenv("GOV_MODEL", "openai/gpt-oss-20b")
OPP_MODEL = os.getenv("OPP_MODEL", "qwen/qwen3.8-27b")

SHARED_RULES = """
Use factual, checkable arguments. For statistics, state the figure, year, and source
(World Bank, IMF, UN, WHO, national statistics office, central bank, etc.). Never invent
a number, date, report, or source. If unsure, say that the figure needs verification.
Respond to the opponent's strongest point, then add a distinct argument. Be civil and
criticize policies, not people. Write in the same language as the user's topic where possible.
Keep each turn concise (about 90-130 words). Do not claim to have live internet access.
If the country is not a real country, reply exactly INVALID_COUNTRY.
""".strip()

GOV_SYSTEM = (
    "You represent the GOVERNMENT of {country} in a balanced political debate. Defend its "
    "policies and record with evidence, acknowledge genuine constraints, and answer criticism.\n\n"
    + SHARED_RULES
)
OPP_SYSTEM = (
    "You represent the OPPOSITION in {country} in a balanced political debate. Challenge "
    "government policies and outcomes with evidence, identify shortcomings, and offer credible alternatives.\n\n"
    + SHARED_RULES
)

SIDES = {
    "gov": {"label": "🏛️ Government", "model": GOV_MODEL, "system": GOV_SYSTEM},
    "opp": {"label": "🛡️ Opposition", "model": OPP_MODEL, "system": OPP_SYSTEM},
}
Message = Dict[str, str]


def get_client() -> Groq:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add your Groq API key to the .env file. "
            "Create a key at https://console.groq.com/keys"
        )
    return Groq(api_key=key, timeout=60.0, max_retries=0)


def build_task_prompt(country: str, topic: str, rounds: int) -> str:
    focus = topic or "the economy, cost of living, jobs, healthcare, education and civil liberties"
    return (
        f"Country: {country}\nDebate topic: {focus}\n"
        f"There are {rounds} rounds; each side speaks once per round."
    )


def build_messages(side: str, task: str, transcript: List[dict], final_round: bool) -> List[Message]:
    msgs: List[Message] = [{"role": "user", "content": task}]
    for turn in transcript:
        role = "assistant" if turn["side"] == side else "user"
        if msgs[-1]["role"] == role:
            msgs[-1]["content"] += "\n\n" + turn["text"]
        else:
            msgs.append({"role": role, "content": turn["text"]})
    if not transcript:
        instruction = "Give your opening statement."
    elif final_round:
        instruction = "Answer your opponent and give a short closing statement."
    else:
        instruction = "Reply directly to the opponent's latest points."
    # Ensure the final instruction is always a user message.
    if msgs[-1]["role"] == "user":
        msgs[-1]["content"] += "\n\n" + instruction
    else:
        msgs.append({"role": "user", "content": instruction})
    return msgs


def get_reply(model: str, system: str, messages: List[Message]) -> str:
    client = get_client()
    full_messages = [{"role": "system", "content": system}] + messages
    last_error = None
    for attempt in range(3):
        try:
            result = client.chat.completions.create(
                model=model,
                messages=full_messages,
                max_completion_tokens=MAX_TOKENS_PER_TURN,
                reasoning_effort="low",
                reasoning_format="hidden",
            )
            message = result.choices[0].message
            answer = message.content or ""
            if not answer.strip():
                # Some reasoning models may return no user-visible content.
                raise RuntimeError(
                    f"Model {model} returned no visible text. Try again; if it repeats, change that model."
                )
            return answer.strip()
        except RateLimitError as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                raise
    raise last_error  # defensive; loop always returns or raises


def render(transcript: List[dict]) -> str:
    return "\n\n---\n\n".join(
        f"### {SIDES[t['side']]['label']} — Round {t['round']}\n\n{t['text']}" for t in transcript
    )


def run_debate(country: str, topic: str, rounds: float):
    country = (country or "").strip()
    topic = (topic or "").strip()
    if not country:
        yield "", "⚠️ Please enter a country."
        return
    if len(country) > 60 or len(topic) > 200:
        yield "", "⚠️ Country must be 60 characters or fewer; topic must be 200 characters or fewer."
        return

    rounds = max(1, min(int(rounds), MAX_ROUNDS))
    task = build_task_prompt(country, topic, rounds)
    transcript: List[dict] = []
    try:
        for r in range(1, rounds + 1):
            for side in ("gov", "opp"):
                cfg = SIDES[side]
                status = f"💬 Round {r}/{rounds}: {cfg['label']} is preparing its response…"
                yield render(transcript), status
                messages = build_messages(side, task, transcript, final_round=(r == rounds))
                answer = get_reply(cfg["model"], cfg["system"].format(country=country), messages)
                if answer.startswith("INVALID_COUNTRY"):
                    yield "", f"⚠️ '{country}' does not look like a real country. Try another country."
                    return
                transcript.append({"side": side, "round": r, "text": answer})
                yield render(transcript), status
        yield render(transcript), f"✅ Debate completed: {rounds} round(s)."
    except RuntimeError as exc:
        yield render(transcript), f"⚠️ Setup error: {exc}"
    except RateLimitError:
        yield render(transcript), "⏳ Groq free-tier rate limit reached. Wait a little, then try fewer rounds."
    except APIConnectionError:
        yield render(transcript), "⚠️ Could not connect to Groq. Check your internet connection."
    except APIStatusError as exc:
        detail = getattr(exc, "message", None) or str(exc)
        yield render(transcript), f"⚠️ Groq API error (HTTP {exc.status_code}): {detail[:500]}"
    except Exception as exc:
        yield render(transcript), f"⚠️ Unexpected error: {type(exc).__name__}: {str(exc)[:500]}"


with gr.Blocks(title="Government vs Opposition — LLM Debate") as demo:
    gr.Markdown(
        "# 🗳️ Government vs Opposition — LLM Debate\n"
        "Two different LLMs debate a country: one represents the government and one the opposition. "
        f"Maximum **{MAX_ROUNDS} rounds**."
    )
    with gr.Row():
        country_in = gr.Textbox(label="Country", placeholder="e.g. Egypt, Brazil, Germany", scale=2)
        topic_in = gr.Textbox(label="Debate topic", placeholder="e.g. inflation, healthcare, education", scale=3)
        rounds_in = gr.Slider(1, MAX_ROUNDS, value=DEFAULT_ROUNDS, step=1, label="Rounds (max 10)", scale=2)
    with gr.Row():
        start_btn = gr.Button("Start debate", variant="primary")
        stop_btn = gr.Button("Stop")
    status_out = gr.Markdown()
    debate_out = gr.Markdown()
    gr.Examples(
        examples=[["Egypt", "the economy and cost of living", 2], ["Brazil", "education policy", 2], ["Germany", "energy policy", 2]],
        inputs=[country_in, topic_in, rounds_in],
    )
    gr.Markdown(
        "<sub>Note: The models do not browse live sources. Statistics may be outdated or inaccurate; "
        "verify figures against the cited source. The government/opposition roles are simulations.</sub>"
    )
    event = start_btn.click(run_debate, [country_in, topic_in, rounds_in], [debate_out, status_out])
    stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[event])

if __name__ == "__main__":
    demo.queue().launch()
