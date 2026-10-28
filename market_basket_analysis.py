"""
Market Basket Analysis using Apriori and FP-Growth
Pure Python version: no pandas, mlxtend, numpy, or other third-party libraries.

Files needed in the same folder:
    market_basket_analysis.py
    transactions.csv

Run:
    python market_basket_analysis.py
"""
import csv
from itertools import combinations

MIN_SUPPORT = 0.25
MIN_CONFIDENCE = 0.60
DATA_FILE = "transactions.csv"


def load_transactions(filename):
    """Read comma-separated product lists from the CSV file."""
    transactions = []
    with open(filename, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            text = row.get("Items", "")
            items = [item.strip() for item in text.split(",") if item.strip()]
            if items:
                transactions.append(set(items))
    return transactions


def support_count(itemset, transactions):
    """Count how many transactions contain every item in itemset."""
    return sum(1 for transaction in transactions if itemset.issubset(transaction))


def apriori(transactions, min_support):
    """Find frequent itemsets using the Apriori candidate-generation method."""
    total = len(transactions)
    if total == 0:
        return {}

    all_items = sorted(set().union(*transactions))
    frequent = {}
    current = {(item,) for item in all_items}

    k = 1
    while current:
        counts = {}
        for candidate in current:
            itemset = set(candidate)
            count = support_count(itemset, transactions)
            support = count / total
            if support >= min_support:
                key = tuple(sorted(candidate))
                counts[key] = support
        if not counts:
            break

        for itemset, support in counts.items():
            frequent[itemset] = support

        k += 1
        previous = set(counts.keys())
        candidates = set()
        for a in previous:
            for b in previous:
                joined = tuple(sorted(set(a) | set(b)))
                if len(joined) == k and len(joined) == len(set(a) | set(b)):
                    # Apriori property: all (k-1)-subsets must be frequent.
                    subsets = combinations(joined, k - 1)
                    if all(tuple(subset) in previous for subset in subsets):
                        candidates.add(joined)
        current = candidates

    return frequent


class FPNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.link = None


def build_fp_tree(transactions, min_count):
    """Build an FP-tree and header links; return (root, header)."""
    item_counts = {}
    for transaction in transactions:
        for item in transaction:
            item_counts[item] = item_counts.get(item, 0) + 1

    item_counts = {item: count for item, count in item_counts.items()
                   if count >= min_count}
    if not item_counts:
        return None, {}

    header = {item: [count, None] for item, count in item_counts.items()}
    root = FPNode(None, 0, None)

    for transaction in transactions:
        filtered = [item for item in transaction if item in item_counts]
        filtered.sort(key=lambda item: (-item_counts[item], item))
        current = root
        for item in filtered:
            if item in current.children:
                child = current.children[item]
                child.count += 1
            else:
                child = FPNode(item, 1, current)
                current.children[item] = child
                if header[item][1] is None:
                    header[item][1] = child
                else:
                    node = header[item][1]
                    while node.link is not None:
                        node = node.link
                    node.link = child
            current = child

    return root, header


def conditional_pattern_base(item, header_entry):
    """Collect prefix paths ending in the given item, with their counts."""
    paths = []
    node = header_entry[1]
    while node is not None:
        path = []
        parent = node.parent
        while parent is not None and parent.item is not None:
            path.append(parent.item)
            parent = parent.parent
        if path:
            paths.append((path, node.count))
        node = node.link
    return paths


def mine_fp_tree(header, min_count, suffix, patterns):
    """Recursively mine an FP-tree and save frequent patterns with counts."""
    # Least frequent items are mined first.
    for item in sorted(header, key=lambda x: (header[x][0], x)):
        new_pattern = tuple(sorted(suffix + (item,)))
        patterns[new_pattern] = header[item][0]

        base = conditional_pattern_base(item, header[item])
        conditional_transactions = []
        for path, count in base:
            for _ in range(count):
                conditional_transactions.append(set(path))

        if not conditional_transactions:
            continue

        _, conditional_header = build_fp_tree(conditional_transactions, min_count)
        if conditional_header:
            mine_fp_tree(conditional_header, min_count, new_pattern, patterns)


def fp_growth(transactions, min_support):
    """Find frequent itemsets using FP-Growth and return their support."""
    total = len(transactions)
    if total == 0:
        return {}
    min_count = max(1, int(min_support * total + 0.999999))
    _, header = build_fp_tree(transactions, min_count)
    patterns = {}
    if header:
        mine_fp_tree(header, min_count, tuple(), patterns)
    return {itemset: count / total for itemset, count in patterns.items()}


def generate_rules(frequent_itemsets, transactions, min_confidence):
    """Generate association rules from frequent itemsets."""
    total = len(transactions)
    rules = []
    for itemset, itemset_support in frequent_itemsets.items():
        if len(itemset) < 2:
            continue
        items = set(itemset)
        for size in range(1, len(itemset)):
            for left_tuple in combinations(sorted(items), size):
                left = set(left_tuple)
                right = items - left
                left_count = support_count(left, transactions)
                right_count = support_count(right, transactions)
                if left_count == 0 or right_count == 0:
                    continue
                confidence = support_count(items, transactions) / left_count
                if confidence >= min_confidence:
                    right_support = right_count / total
                    lift = confidence / right_support if right_support else 0
                    rules.append((tuple(sorted(left)), tuple(sorted(right)),
                                  itemset_support, confidence, lift))
    rules.sort(key=lambda rule: (-rule[3], -rule[4], rule[0], rule[1]))
    return rules


def display_itemsets(title, itemsets):
    print(f"\n--- {title}: FREQUENT ITEMSETS ---")
    if not itemsets:
        print("No frequent itemsets found. Try lowering MIN_SUPPORT.")
        return
    print(f"{'Itemset':45} {'Support':>10}")
    print("-" * 58)
    for itemset, support in sorted(itemsets.items(),
                                   key=lambda x: (len(x[0]), x[0])):
        print(f"{', '.join(itemset):45} {support:>9.1%}")


def display_rules(title, rules):
    print(f"\n--- {title}: ASSOCIATION RULES ---")
    if not rules:
        print("No rules found. Try lowering MIN_CONFIDENCE.")
        return
    print(f"{'Rule':43} {'Support':>9} {'Confidence':>11} {'Lift':>7}")
    print("-" * 75)
    for left, right, support, confidence, lift in rules:
        rule_text = f"{', '.join(left)} -> {', '.join(right)}"
        print(f"{rule_text:43} {support:>8.1%} {confidence:>10.1%} {lift:>7.2f}")


def main():
    transactions = load_transactions(DATA_FILE)
    if not transactions:
        print("No transactions found. Check transactions.csv.")
        return

    print("=" * 60)
    print(" MARKET BASKET ANALYSIS: APRIORI AND FP-GROWTH")
    print("=" * 60)
    print(f"Transactions loaded: {len(transactions)}")
    print(f"Products found: {', '.join(sorted(set().union(*transactions)))}")
    print(f"Minimum support: {MIN_SUPPORT:.0%}")
    print(f"Minimum confidence: {MIN_CONFIDENCE:.0%}")

    apriori_sets = apriori(transactions, MIN_SUPPORT)
    fp_sets = fp_growth(transactions, MIN_SUPPORT)

    display_itemsets("APRIORI", apriori_sets)
    display_rules("APRIORI", generate_rules(apriori_sets, transactions,
                                             MIN_CONFIDENCE))
    display_itemsets("FP-GROWTH", fp_sets)
    display_rules("FP-GROWTH", generate_rules(fp_sets, transactions,
                                               MIN_CONFIDENCE))

    print("\n--- COMPARISON ---")
    if {k for k in apriori_sets} == {k for k in fp_sets}:
        print("Both algorithms found the same frequent itemsets.")
    else:
        print("The itemsets differ; check the implementation and thresholds.")
    print("Apriori checks candidate itemsets. FP-Growth builds and mines an FP-tree.")
    print("\nTip: Change MIN_SUPPORT or MIN_CONFIDENCE at the top to explore results.")


if __name__ == "__main__":
    main()
