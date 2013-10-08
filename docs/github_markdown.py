"""
Fix up pandoc's code syntax highlighting, which github doesn't understand
"""

import re
import sys

fn = sys.argv[1]
data = open("out.md").read()
data = re.sub(r"~~~~ {.sourceCode .(.*?)}", r"```\1", data)
data = data.replace("~~~~", "```")
open(fn, "w").write(data)
