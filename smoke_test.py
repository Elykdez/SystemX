"""Exercise a running real server and record end-to-end HTTP latency."""

import argparse
import json
import os
from pathlib import Path
import statistics
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.samples <= 100:
        parser.error("--samples must be between 1 and 100")
    headers = {"Content-Type": "application/json"}
    if os.environ.get("VON_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["VON_API_KEY"]

    def post(payload):
        req = urllib.request.Request(args.url + "/v1/systemone", json.dumps(payload).encode(), headers)
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response)

    questions = {
        "route": {"type": "choice", "instructions": "Which support team should handle this ticket?", "criteria": {
            "billing": "Payment, duplicate charge, invoice or refund problems.",
            "technical": "Software crashes, access problems or broken features."}},
        "refund": {"type": "noul", "instructions": "Did the customer explicitly request a refund?",
                   "criteria": {"true": "The customer asks for money back.", "false": "The customer does not request money back."}},
        "impact": {"type": "score", "instructions": "Rate the customer's financial impact.", "criteria": [
            "No financial loss.", "An incorrect charge needs correction.", "Repeated major financial losses."]}}
    billing = "I was charged twice for one order. Please refund the duplicate charge."
    technical = "The application crashes whenever I open settings. My payments are correct and I am not requesting a refund."
    payload = {"model": "von-option-marker", "state": billing, "questions": questions}
    response = post(payload)
    assert set(response["answers"]) == set(questions)
    for name in ("route", "impact"):
        assert abs(sum(response["answers"][name]["probabilities"].values()) - 1) < .002
    assert 0 <= response["answers"]["refund"]["noul"] <= 1
    assert 0 <= response["answers"]["impact"]["score"] <= 2
    technical_response = post({**payload, "state": technical})
    timings = []
    # Measure a single bounded Choice per HTTP request.
    payload["questions"] = {"route": questions["route"]}
    for _ in range(args.samples):
        start = time.perf_counter()
        single = post(payload)
        timings.append((time.perf_counter() - start) * 1000)
        assert single["answers"]["route"]["choice"] in questions["route"]["criteria"]
    with urllib.request.urlopen(args.url + "/health") as health:
        status = json.load(health)
    report = {"health": status, "samples": len(timings), "http_ms": {
        "median": round(statistics.median(timings), 2), "min": round(min(timings), 2),
        "max": round(max(timings), 2), "p95": round(sorted(timings)[int(.95 * (len(timings) - 1))], 2)},
        "billing_state": response, "technical_state": technical_response,
        "state_sensitive": response["answers"]["route"]["choice"] != technical_response["answers"]["route"]["choice"]}
    Path(__file__).with_name("benchmark-results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
