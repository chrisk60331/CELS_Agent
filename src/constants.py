
tests = [
    ("pick any number between 1 and 100", "../data/42.txt"),
    ("summarize file data/food_facts.json", "../data/food_facts.txt"),
    ("summarize file data/history.json", "../data/history.txt"),
    ("summarize file data/prize.json", "../data/prize_summary.txt"),
    ("summarize file data/laureate.json", "../data/laureate.txt"),
]

task, SUMMARY_FILE = tests[2] 

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
