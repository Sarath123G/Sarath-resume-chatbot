"""Command-line interface.

  python cli.py ingest [--reset] [--path data/]
  python cli.py chat
  python cli.py ask "What are the candidate's key skills?"
"""
import argparse
import sys

from src import config
from src.rag import ResumeRAG


def main():
    ap = argparse.ArgumentParser(description="Resume RAG chatbot")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Index resume(s) from a folder or file")
    p_ing.add_argument("--path", default=str(config.DATA_DIR))
    p_ing.add_argument("--reset", action="store_true", help="Clear the index first")

    sub.add_parser("chat", help="Interactive chat")
    p_ask = sub.add_parser("ask", help="Ask a single question")
    p_ask.add_argument("question")

    args = ap.parse_args()
    rag = ResumeRAG()

    if args.cmd == "ingest":
        from pathlib import Path

        if args.reset:
            rag.store.reset()
        p = Path(args.path)
        report = {p.name: rag.ingest_file(p)} if p.is_file() else rag.ingest_folder(p)
        for name, n in report.items():
            print(f"indexed {n:>3} chunks  <- {name}")
        print(f"total chunks in store: {rag.store.count()}")
        return

    if rag.store.count() == 0:
        sys.exit("Index is empty. Run: python cli.py ingest")

    if args.cmd == "ask":
        stream, _ = rag.ask_stream(args.question)
        for tok in stream:
            print(tok, end="", flush=True)
        print()
        return

    history = []
    print(f"Resume chatbot ({config.LLM_MODEL}). Type 'exit' to quit.\n")
    while True:
        try:
            q = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", ""}:
            if q:
                break
            continue
        stream, _ = rag.ask_stream(q, history)
        print("bot > ", end="", flush=True)
        answer = ""
        for tok in stream:
            answer += tok
            print(tok, end="", flush=True)
        print("\n")
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]


if __name__ == "__main__":
    main()
