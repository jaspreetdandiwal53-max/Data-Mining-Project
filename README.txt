MARKET BASKET ANALYSIS - PURE PYTHON VERSION

This version does NOT require pandas, mlxtend, numpy, or any extra library.
It uses only Python's built-in csv and itertools modules.

Files:
- market_basket_analysis.py: implements Apriori, FP-Growth, and association rules
- transactions.csv: sample shopping transactions

How to run in VS Code:
1. Keep both files in the same folder.
2. Open VS Code terminal in that folder.
3. Run: python market_basket_analysis.py

If VS Code says the file cannot be found, use File > Open Folder and open the folder
containing these two files.

You can change MIN_SUPPORT and MIN_CONFIDENCE near the top of the Python file.
