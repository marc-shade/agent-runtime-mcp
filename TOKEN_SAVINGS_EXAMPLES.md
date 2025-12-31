# Agent Runtime MCP - Token Savings Examples

## Real Examples from Testing

### Example 1: Simple Task Array (TOON WINS)

**Scenario**: List of basic tasks with 3 fields each

**JSON Compact** (226 chars, ~56 tokens):
```json
[{"id":1,"title":"Task 1","status":"pending"},{"id":2,"title":"Task 2","status":"pending"},{"id":3,"title":"Task 3","status":"pending"},{"id":4,"title":"Task 4","status":"pending"},{"id":5,"title":"Task 5","status":"pending"}]
```

**TOON Compact** (116 chars, ~29 tokens):
```toon
[5]{id,title,status}:
  1,Task 1,pending
  2,Task 2,pending
  3,Task 3,pending
  4,Task 4,pending
  5,Task 5,pending
```

**Result**: 48.7% token savings (27 tokens saved) ✅

---

### Example 2: Complex Task Array (JSON WINS)

**Scenario**: Full task objects with 9 fields, nested arrays, timestamps

**JSON Compact** (4,043 chars, ~1,010 tokens):
```json
[{"id":1,"goal_id":5,"title":"Task 1","description":"Description for task 1","status":"in_progress","priority":6,"dependencies":[],"created_at":"2025-01-20T10:01:00","updated_at":"2025-01-20T10:01:00"},...]
```

**TOON** (4,446 chars, ~1,111 tokens):
```toon
[20]:
  - id: 1
    goal_id: 5
    title: Task 1
    description: Description for task 1
    status: in_progress
    priority: 6
    dependencies[0]:
    created_at: "2025-01-20T10:01:00"
    updated_at: "2025-01-20T10:01:00"
  - id: 2
    ...
```

**Result**: -10.0% MORE tokens (101 tokens wasted) ❌

---

### Example 3: Single Task Object (TOON WINS)

**Scenario**: Single task with metadata

**JSON Formatted** (348 chars, ~87 tokens):
```json
{
  "id": 1,
  "goal_id": 5,
  "title": "Implement feature X",
  "description": "Add new functionality to the system",
  "status": "pending",
  "priority": 8,
  "dependencies": [2, 3],
  "created_at": "2025-01-20T10:00:00",
  "updated_at": "2025-01-20T10:00:00",
  "metadata": {
    "tags": ["urgent", "backend"]
  }
}
```

**JSON Compact** (266 chars, ~66 tokens):
```json
{"id":1,"goal_id":5,"title":"Implement feature X","description":"Add new functionality to the system","status":"pending","priority":8,"dependencies":[2,3],"created_at":"2025-01-20T10:00:00","updated_at":"2025-01-20T10:00:00","metadata":{"tags":["urgent","backend"]}}
```

**TOON** (245 chars, ~61 tokens):
```toon
id: 1
goal_id: 5
title: Implement feature X
description: Add new functionality to the system
status: pending
priority: 8
dependencies[2]: 2,3
created_at: "2025-01-20T10:00:00"
updated_at: "2025-01-20T10:00:00"
metadata:
  tags[2]: urgent,backend
```

**Result**: 7.9% token savings (5 tokens saved) ✅

---

### Example 4: Goal Decomposition Response (TOON WINS)

**Scenario**: Goal decomposition result object

**JSON Compact** (131 chars, ~32 tokens):
```json
{"goal_id":5,"strategy":"sequential","tasks_created":[101,102,103,104,105],"count":5,"meta_prompted":true,"complexity_score":"85%"}
```

**TOON** (120 chars, ~30 tokens):
```toon
goal_id: 5
strategy: sequential
tasks_created[5]: 101,102,103,104,105
count: 5
meta_prompted: true
complexity_score: 85%
```

**Result**: 8.4% token savings (2 tokens saved) ✅

---

## Daily Usage Impact

### Typical Agent Runtime MCP Usage

| Operation | Daily Count | Tokens per Call | Format | Daily Tokens |
|-----------|-------------|-----------------|--------|--------------|
| list_tasks (complex) | 20 | 1,010 (JSON) / 1,111 (TOON) | ❌ TOON worse | +2,020 |
| get_task (single) | 50 | 66 (JSON) / 61 (TOON) | ✅ TOON better | -250 |
| decompose_goal | 10 | 32 (JSON) / 30 (TOON) | ✅ TOON better | -20 |
| **TOTAL** | **80** | - | **Net** | **+1,750/day** |

**Annual Impact**: +638,750 tokens = **-$1.92/year** (WORSE with TOON)

---

## The Key Insight

TOON intelligently chooses encoding strategy:

### Simple Arrays (3-5 fields, no nesting)
→ **Compact Tabular Format**
```toon
[N]{field1,field2,field3}:
  val1,val2,val3
  val1,val2,val3
```
**Result**: 40-50% token savings ✅

### Complex Arrays (8+ fields, nested structures)
→ **YAML-like Format**
```toon
[N]:
  - field1: val1
    field2: val2
    nested:
      - item1
```
**Result**: -10% token loss ❌

### Single Objects
→ **Key-Value Format**
```toon
field1: value1
field2: value2
array[N]: val1,val2
```
**Result**: 5-10% token savings ✅

---

## Optimization Strategy

### Current Reality

Agent Runtime MCP primarily returns **complex task arrays** with:
- 9+ fields per task
- Nested dependency arrays
- JSON-formatted metadata
- ISO timestamp strings

This triggers TOON's YAML mode → **10% token LOSS**.

### Solution: Selective TOON

```python
def _encode_response(data: Any, pretty: bool = False) -> str:
    """Use TOON only when beneficial"""
    if TOON_AVAILABLE and _would_benefit_from_toon(data):
        return encode_with_fallback(data, pretty=pretty)
    else:
        return json.dumps(data, separators=(',', ':'))

def _would_benefit_from_toon(data: Any) -> bool:
    """Check if data structure benefits from TOON encoding"""
    # Arrays with 5 or fewer fields → TOON wins
    if isinstance(data, list) and data and isinstance(data[0], dict):
        field_count = len(data[0].keys())
        has_nested = any(isinstance(v, (list, dict)) for v in data[0].values())
        return field_count <= 5 and not has_nested

    # Single objects → TOON wins slightly
    elif isinstance(data, dict):
        return True

    # Primitives → no difference
    else:
        return False
```

### Expected Impact with Selective TOON

| Operation | Daily Count | Current | With Selective | Savings |
|-----------|-------------|---------|----------------|---------|
| list_tasks | 20 | +2,020 tokens | 0 tokens (use JSON) | ✅ -2,020 |
| get_task | 50 | -250 tokens | -250 tokens (use TOON) | ✅ -250 |
| decompose_goal | 10 | -20 tokens | -20 tokens (use TOON) | ✅ -20 |
| **TOTAL** | **80** | **+1,750** | **-270** | **✅ -2,020** |

**Annual Impact**: -98,550 tokens = **+$0.30/year savings** (BETTER than both!)

---

## Recommendation

✅ **Implement Selective TOON** (20 lines of code)

- Use TOON for: single objects, simple arrays (≤5 fields)
- Use JSON for: complex arrays, nested structures
- Result: Best of both worlds

**Implementation time**: 10 minutes
**Token savings**: 2,020 tokens/day
**Cost savings**: $0.30/year + avoids $1.92 loss = **$2.22/year benefit**

---

## Code Example

### Full Implementation

```python
def _encode_response(data: Any, pretty: bool = False) -> str:
    """
    Encode response using optimal format.
    Uses TOON for simple structures, JSON for complex ones.
    """
    if TOON_AVAILABLE and _should_use_toon(data):
        return encode_with_fallback(data, pretty=pretty)
    else:
        if pretty:
            return json.dumps(data, indent=2)
        else:
            return json.dumps(data, separators=(',', ':'))


def _should_use_toon(data: Any) -> bool:
    """
    Determine if TOON encoding would benefit this data structure.

    TOON wins for:
    - Simple arrays (≤5 fields, no nesting): 48% savings
    - Single objects: 8% savings

    JSON wins for:
    - Complex arrays (>5 fields or nested): -10% loss
    """
    if isinstance(data, list):
        if not data or not isinstance(data[0], dict):
            return True  # Primitive arrays, doesn't matter much

        # Check complexity
        first_item = data[0]
        field_count = len(first_item)

        # Too many fields → JSON wins
        if field_count > 5:
            return False

        # Check for nesting
        for value in first_item.values():
            if isinstance(value, (list, dict)):
                return False  # Nested structures → JSON wins

        return True  # Simple array → TOON wins

    elif isinstance(data, dict):
        return True  # Single objects → TOON wins slightly

    else:
        return False  # Primitives → doesn't matter


# Usage in tool responses:
return [TextContent(
    type="text",
    text=_encode_response(tasks, pretty=False)  # Auto-selects optimal format
)]
```

---

## Testing

```bash
# Test the optimization
cd /Volumes/SSDRAID0/agentic-system/mcp-servers/agent-runtime-mcp
python3 -c "
from server import _should_use_toon

# Simple array → TOON
simple = [{'id': 1, 'title': 'Task', 'status': 'done'}]
print(f'Simple array should use TOON: {_should_use_toon(simple)}')

# Complex array → JSON
complex = [{'id': 1, 'goal_id': 5, 'title': 'Task', 'description': 'Desc',
            'status': 'done', 'priority': 5, 'dependencies': [],
            'created_at': '2025-01-20', 'updated_at': '2025-01-20'}]
print(f'Complex array should use JSON: {not _should_use_toon(complex)}')
"
```

---

## Summary

| Scenario | TOON Format | Token Impact | Recommendation |
|----------|-------------|--------------|----------------|
| Simple arrays (≤5 fields) | Tabular | ✅ +48% savings | Use TOON |
| Complex arrays (>5 fields) | YAML | ❌ -10% loss | Use JSON |
| Single objects | Key-value | ✅ +8% savings | Use TOON |
| Mixed usage (current) | YAML | ❌ -10% net | Implement selective |

**Final Answer**: Implement selective TOON encoding for optimal token efficiency.
