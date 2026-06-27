"""Interactive REPL for the FloodTwin agent."""
from __future__ import annotations

import logging


def repl(four_bit=None):
    """Start a terminal chat loop. Type 'exit' / 'quit' to leave, 'reset' to clear history."""
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    from .llm import Agent

    agent = Agent(four_bit=four_bit)
    print("FloodTwin agent ready. Ask about today's outlook, what-ifs, or interventions.\n"
          "(commands: exit, quit, reset)\n")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in ("exit", "quit"):
            break
        if user.lower() == "reset":
            agent.reset()
            print("(history cleared)")
            continue
        print("\nFloodTwin>", agent.chat(user), "\n")


if __name__ == "__main__":
    repl()
