import sys
sys.path.insert(0, r'd:\Code generator agent')
from agents.interpreter import _prompt_uses_files

cases = [
    ('real time hand movement detection using camera', False),
    ('read all csv files in a folder and plot data', True),
    ('control screen brightness with hand gestures', False),
    ('parse log files and generate a report', True),
    ('generate a prime number sieve', False),
    ('rename all files in directory', True),
]

all_ok = True
for prompt, expected in cases:
    result = _prompt_uses_files(prompt)
    status = '[OK]  ' if result == expected else '[FAIL]'
    if result != expected:
        all_ok = False
    print(f'{status} expected={expected} got={result} | "{prompt}"')

sys.exit(0 if all_ok else 1)
