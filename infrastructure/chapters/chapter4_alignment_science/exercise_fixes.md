# Exercise Fixes Tracking

## Fix 1: Remove Exercise 4 (batched activation extraction)
- Status: PENDING
- Action: Delete exercise markdown, code cell, and HIDE block

## Fix 2: Move Exercise 10 (load/parse transcripts) to utils.py
- Status: PENDING
- Action: Move `load_transcript` to utils.py, remove exercise markdown/code, add import

## Fix 3: Shrink Exercise 17 (compute safe range threshold)
- Status: PENDING
- Action: Keep exercise but only mark out hook+projection code (not the full function)

## Fix 4: Remove Exercise 22 (score responses with autorater)
- Status: PENDING
- Action: Delete exercise, keep the function as provided code

## Fix 5: Create `generate_responses_api` utility + refactor all API calls
- Status: PENDING
- Action: Create parallelized `generate_responses_api(messages_list)` function,
  refactor all `openrouter_client.chat.completions.create` calls to use it,
  simplify Exercise 2 (generate responses for all personas)
