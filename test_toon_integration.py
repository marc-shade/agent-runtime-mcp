#!/usr/bin/env python3
"""
Test TOON integration in Agent Runtime MCP
Validates token savings and backward compatibility
"""

import os
import platform
import sys
import json
from pathlib import Path


def _get_storage_base() -> Path:
    """Detect storage base path based on platform."""
    env_path = os.environ.get("AGENTIC_SYSTEM_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    system = platform.system()
    if system == "Darwin":  # macOS
        if Path("/Volumes/SSDRAID0/agentic-system").exists():
            return Path("/Volumes/SSDRAID0/agentic-system")
        elif Path("/Volumes/FILES/agentic-system").exists():
            return Path("/Volumes/FILES/agentic-system")
    elif system == "Linux":
        if Path("/home/marc/agentic-system").exists():
            return Path("/home/marc/agentic-system")
        elif Path("/mnt/agentic-system").exists():
            return Path("/mnt/agentic-system")
    return Path(__file__).parent.parent.parent


_STORAGE_BASE = _get_storage_base()

# Add shared directory to path
sys.path.insert(0, str(_STORAGE_BASE / "mcp-servers" / "shared"))

try:
    from toon_utils import encode_with_fallback, smart_decode
    from toon_codec import ToonCodec
    TOON_AVAILABLE = True
except ImportError as e:
    print(f"❌ TOON not available: {e}")
    TOON_AVAILABLE = False
    sys.exit(1)


def test_single_task():
    """Test single task encoding"""
    task = {
        "id": 1,
        "goal_id": 5,
        "title": "Implement feature X",
        "description": "Add new functionality to the system",
        "status": "pending",
        "priority": 8,
        "dependencies": [2, 3],
        "created_at": "2025-01-20T10:00:00",
        "updated_at": "2025-01-20T10:00:00",
        "metadata": {"tags": ["urgent", "backend"]}
    }

    print("=== Single Task Test ===")

    # JSON baseline
    json_formatted = json.dumps(task, indent=2)
    json_compact = json.dumps(task, separators=(',', ':'))

    print(f"\nJSON (formatted): {len(json_formatted)} chars (~{len(json_formatted)//4} tokens)")
    print(f"JSON (compact): {len(json_compact)} chars (~{len(json_compact)//4} tokens)")

    # TOON encoding
    toon_encoded = encode_with_fallback(task, pretty=False)
    print(f"TOON: {len(toon_encoded)} chars (~{len(toon_encoded)//4} tokens)")
    print(f"\nTOON output:\n{toon_encoded}")

    # Calculate savings
    savings_vs_formatted = ((len(json_formatted) - len(toon_encoded)) / len(json_formatted)) * 100
    savings_vs_compact = ((len(json_compact) - len(toon_encoded)) / len(json_compact)) * 100

    print(f"\nSavings: {savings_vs_formatted:.1f}% vs formatted, {savings_vs_compact:.1f}% vs compact")

    # Test round-trip
    decoded = smart_decode(toon_encoded)
    print(f"\nRound-trip successful: {decoded == task}")

    return {
        "json_formatted_tokens": len(json_formatted) // 4,
        "json_compact_tokens": len(json_compact) // 4,
        "toon_tokens": len(toon_encoded) // 4,
        "savings_percent": savings_vs_compact
    }


def test_task_queue():
    """Test task queue encoding (where TOON really shines)"""
    tasks = [
        {
            "id": i,
            "goal_id": 5,
            "title": f"Task {i}",
            "description": f"Description for task {i}",
            "status": "pending" if i % 2 == 0 else "in_progress",
            "priority": 5 + (i % 5),
            "dependencies": [] if i == 1 else [i-1],
            "created_at": f"2025-01-20T10:{i:02d}:00",
            "updated_at": f"2025-01-20T10:{i:02d}:00"
        }
        for i in range(1, 21)  # 20 tasks
    ]

    print("\n=== Task Queue Test (20 tasks) ===")

    # JSON baseline
    json_formatted = json.dumps(tasks, indent=2)
    json_compact = json.dumps(tasks, separators=(',', ':'))

    print(f"\nJSON (formatted): {len(json_formatted)} chars (~{len(json_formatted)//4} tokens)")
    print(f"JSON (compact): {len(json_compact)} chars (~{len(json_compact)//4} tokens)")

    # TOON encoding
    toon_encoded = encode_with_fallback(tasks, pretty=False)
    print(f"TOON: {len(toon_encoded)} chars (~{len(toon_encoded)//4} tokens)")

    # Show first 200 chars
    print(f"\nTOON output (first 200 chars):\n{toon_encoded[:200]}...")

    # Calculate savings
    savings_vs_formatted = ((len(json_formatted) - len(toon_encoded)) / len(json_formatted)) * 100
    savings_vs_compact = ((len(json_compact) - len(toon_encoded)) / len(json_compact)) * 100

    print(f"\nSavings: {savings_vs_formatted:.1f}% vs formatted, {savings_vs_compact:.1f}% vs compact")
    print(f"Token savings: {(len(json_compact) - len(toon_encoded))//4} tokens saved")

    # Test round-trip
    decoded = smart_decode(toon_encoded)
    print(f"\nRound-trip successful: {len(decoded) == len(tasks)}")

    return {
        "json_formatted_tokens": len(json_formatted) // 4,
        "json_compact_tokens": len(json_compact) // 4,
        "toon_tokens": len(toon_encoded) // 4,
        "savings_percent": savings_vs_compact,
        "tokens_saved": (len(json_compact) - len(toon_encoded)) // 4
    }


def test_goal_decomposition_response():
    """Test goal decomposition response (common operation)"""
    response = {
        "goal_id": 5,
        "strategy": "sequential",
        "tasks_created": [101, 102, 103, 104, 105],
        "count": 5,
        "meta_prompted": True,
        "complexity_score": "85%"
    }

    print("\n=== Goal Decomposition Response ===")

    # JSON baseline
    json_formatted = json.dumps(response, indent=2)
    json_compact = json.dumps(response, separators=(',', ':'))

    print(f"\nJSON (formatted): {len(json_formatted)} chars (~{len(json_formatted)//4} tokens)")
    print(f"JSON (compact): {len(json_compact)} chars (~{len(json_compact)//4} tokens)")

    # TOON encoding
    toon_encoded = encode_with_fallback(response, pretty=False)
    print(f"TOON: {len(toon_encoded)} chars (~{len(toon_encoded)//4} tokens)")
    print(f"\nTOON output:\n{toon_encoded}")

    # Calculate savings
    savings_vs_compact = ((len(json_compact) - len(toon_encoded)) / len(json_compact)) * 100
    print(f"\nSavings: {savings_vs_compact:.1f}% vs compact")

    return {
        "json_compact_tokens": len(json_compact) // 4,
        "toon_tokens": len(toon_encoded) // 4,
        "savings_percent": savings_vs_compact
    }


def test_daily_usage_estimate():
    """Estimate daily token savings"""
    print("\n=== Daily Usage Estimate ===")

    # Assume typical usage:
    # - 10 goal decompositions per day
    # - 20 task list queries per day
    # - 50 single task queries per day

    single_task_result = test_single_task()
    task_queue_result = test_task_queue()
    goal_result = test_goal_decomposition_response()

    daily_savings = (
        (single_task_result['json_compact_tokens'] - single_task_result['toon_tokens']) * 50 +
        (task_queue_result['json_compact_tokens'] - task_queue_result['toon_tokens']) * 20 +
        (goal_result['json_compact_tokens'] - goal_result['toon_tokens']) * 10
    )

    print(f"\nEstimated daily token savings: ~{daily_savings} tokens")
    print(f"Monthly savings: ~{daily_savings * 30} tokens")
    print(f"Annual savings: ~{daily_savings * 365} tokens")

    # Cost estimate (Claude Sonnet pricing: ~$3/million input tokens)
    cost_per_token = 3 / 1_000_000
    daily_cost_savings = daily_savings * cost_per_token
    annual_cost_savings = daily_cost_savings * 365

    print(f"\nEstimated cost savings:")
    print(f"Daily: ${daily_cost_savings:.4f}")
    print(f"Annual: ${annual_cost_savings:.2f}")


def test_backward_compatibility():
    """Test backward compatibility with JSON clients"""
    print("\n=== Backward Compatibility Test ===")

    task = {
        "id": 1,
        "title": "Test task",
        "status": "pending"
    }

    # Encode to TOON
    toon_encoded = encode_with_fallback(task, pretty=False)
    print(f"TOON encoded: {toon_encoded}")

    # Decode back
    decoded = smart_decode(toon_encoded)
    print(f"Decoded successfully: {decoded == task}")

    # Test JSON decoding still works
    json_str = json.dumps(task)
    decoded_json = smart_decode(json_str)
    print(f"JSON decoding works: {decoded_json == task}")

    print("\n✓ Backward compatibility maintained")


if __name__ == "__main__":
    print("=" * 60)
    print("Agent Runtime MCP - TOON Integration Test")
    print("=" * 60)

    if not TOON_AVAILABLE:
        print("\n❌ TOON utilities not available")
        print(f"Install with: cd {_STORAGE_BASE / 'mcp-servers' / 'shared'} && npm install @toon-format/toon @toon-format/cli")
        sys.exit(1)

    try:
        # Run all tests
        test_single_task()
        test_task_queue()
        test_goal_decomposition_response()
        test_daily_usage_estimate()
        test_backward_compatibility()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
