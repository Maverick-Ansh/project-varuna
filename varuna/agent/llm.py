"""Local LLM agent (Qwen2.5-Instruct) with a tool-calling loop.

Uses the model's native tool-calling chat template (transformers >= 4.43). Parses
<tool_call>{...}</tool_call> blocks, executes them via varuna.agent.tools.dispatch, feeds
results back, and repeats until the model answers in prose.

Memory: on a 16 GB T4 sharing VRAM with the differentiable simulator, set CFG.llm_4bit=True
(bitsandbytes) — fp16 7B (~15 GB) leaves too little headroom. A100/L4 on Colab Pro is roomier.
"""
from __future__ import annotations

import json
import logging
import re

from ..config import CFG
from .prompts import SYSTEM
from .tools import TOOLS, dispatch

log = logging.getLogger("varuna.agent.llm")

_TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text):
    """Extract [{name, arguments}] from a model turn. Tolerates missing/extra whitespace."""
    calls = []
    for m in _TOOLCALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
            calls.append({"name": obj.get("name"), "arguments": obj.get("arguments", {}) or {}})
        except json.JSONDecodeError:
            log.warning("could not parse tool call: %s", m.group(1)[:120])
    return calls


class Agent:
    """Stateful chat agent. Call `.chat(user_text)` -> assistant reply string."""

    def __init__(self, model_name=None, device=None, four_bit=None, max_new_tokens=512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = model_name or CFG.llm_model
        four_bit = CFG.llm_4bit if four_bit is None else four_bit
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        kw = {}
        if four_bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            kw["device_map"] = "auto"
        else:
            # fp16 on CUDA (incl. T4 — see memory note); float32 on CPU.
            kw["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
            kw["device_map"] = "auto" if torch.cuda.is_available() else None

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
        if not torch.cuda.is_available():
            self.model = self.model.to(device or "cpu")
        self.messages = [{"role": "system", "content": SYSTEM}]
        log.info("loaded %s (4bit=%s)", model_name, four_bit)

    def _generate(self):
        import torch
        prompt = self.tokenizer.apply_chat_template(
            self.messages, tools=TOOLS, add_generation_prompt=True, tokenize=False)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, temperature=None, top_p=None,
                                      pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def chat(self, user_text, max_tool_rounds=6):
        """One user turn. Runs tool calls internally and returns the final prose answer."""
        self.messages.append({"role": "user", "content": user_text})
        for _ in range(max_tool_rounds):
            reply = self._generate()
            calls = parse_tool_calls(reply)
            if not calls:
                self.messages.append({"role": "assistant", "content": reply})
                return reply.strip()
            # record the assistant's tool-call turn, then the results
            self.messages.append({
                "role": "assistant", "content": "",
                "tool_calls": [{"type": "function", "function": c} for c in calls],
            })
            for c in calls:
                result = dispatch(c["name"], c["arguments"])
                log.info("tool %s -> %s", c["name"], str(result)[:200])
                self.messages.append({"role": "tool", "name": c["name"],
                                      "content": json.dumps(result, default=float)})
        # ran out of tool rounds; force a final answer
        final = self._generate()
        self.messages.append({"role": "assistant", "content": final})
        return final.strip()

    def reset(self):
        self.messages = [{"role": "system", "content": SYSTEM}]


_AGENT = None


def chat_once(message, history=None, work=None, four_bit=None):
    """Dashboard fallback entry point for the local Qwen agent (see api/server.py::chat).

    Lazily loads and caches ONE Agent (a 7B model is heavy). `history` is accepted for API parity
    with api/chat_hosted.chat_once but ignored — the cached agent keeps its own conversation. `work`
    is likewise accepted for parity; the agent's tools read the active bundle via CFG. Prefer the
    hosted path (set LLM_API_KEY) for the deployed dashboard; this exists so a local-GPU deploy has
    a working chat instead of the previous dead import.
    """
    global _AGENT
    if _AGENT is None:
        _AGENT = Agent(four_bit=four_bit)
    return _AGENT.chat(message)
