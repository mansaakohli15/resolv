"""
Resolv Interactive CLI Entrypoint.
Run inference on a single query or interactive shell.

Usage:
  python run_pipeline.py --message "Where is my order?"
  python run_pipeline.py --interactive
"""
import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import ResolvPipeline


def print_result(res: dict):
    print("\n" + "=" * 70)
    print(f"CUSTOMER QUERY:  {res['customer_message']}")
    print(f"PREDICTED INTENT: {res['intent']} (Confidence: {res['intent_confidence']:.2f}, Margin: {res['intent_margin']:.2f})")
    print(f"DECISION:         [{res['decision']}]")
    print(f"PRIMARY REASON:   {res['reason']}")
    if res.get('reason_codes'):
        print(f"REASON CODES:     {', '.join(res['reason_codes'])}")
    print("-" * 70)
    print(f"SUGGESTED REPLY:  \"{res['suggested_reply']}\"")
    print("-" * 70)
    print("HISTORICAL EVIDENCE RETRIEVED:")
    for idx, ev in enumerate(res['evidence'], start=1):
        print(f"  [{idx}] Case ID: {ev['case_id']} | Intent: {ev['intent']} | Similarity: {ev['similarity_score']:.2f}")
        print(f"      Customer: \"{ev['historical_customer_text'][:100]}...\"")
        print(f"      Brand:    \"{ev['historical_agent_reply'][:100]}...\"")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Resolv Customer Support Automation CLI")
    parser.add_argument("--message", "-m", type=str, help="Customer support message to process")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive CLI session")
    parser.add_argument("--json", action="store_true", help="Output raw JSON format")
    args = parser.parse_args()

    pipeline = ResolvPipeline.load_from_checkpoints()

    if args.message:
        res = pipeline.process_message(args.message)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print_result(res)
    elif args.interactive:
        print("\n=== Resolv Customer Support Automation Console ===")
        print("Type your customer query below (or 'exit' to quit):\n")
        while True:
            try:
                user_msg = input("Customer > ").strip()
                if not user_msg:
                    continue
                if user_msg.lower() in ["exit", "quit", "q"]:
                    break
                res = pipeline.process_message(user_msg)
                print_result(res)
            except (KeyboardInterrupt, EOFError):
                break
    else:
        # Default demo query
        demo_msg = "My delivery was scheduled for yesterday but tracking has not updated at all. Can I get a refund?"
        print(f"No message specified. Running default demo query:\n> \"{demo_msg}\"")
        res = pipeline.process_message(demo_msg)
        print_result(res)


if __name__ == "__main__":
    main()
