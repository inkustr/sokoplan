#!/usr/bin/env python3
"""Compare GNN vs Hungarian heuristics on the same benchmark."""

import csv
from collections import defaultdict

def load_csv(path):
    results = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row['level_id']:
                continue
            level_id = row['level_id']
            nodes = int(row['nodes'])
            runtime = float(row['runtime'])
            success = row['success'] == 'True'
            solution_len = int(row['solution_len']) if row['solution_len'] != '-1' else -1
            results[level_id] = (nodes, runtime, success, solution_len)
    return results

gnn_results = load_csv('results/evaluate/batch_gnn_speed.sorted.csv')
hungarian_results = load_csv('results/evaluate/train.sorted.csv')

print("=" * 100)
print("GNN vs HUNGARIAN COMPARISON")
print("=" * 100)
print()

gnn_levels = set(gnn_results.keys())
hungarian_levels = set(hungarian_results.keys())
common_levels = gnn_levels & hungarian_levels

if len(common_levels) != len(gnn_levels) or len(common_levels) != len(hungarian_levels):
    print(f"Warning: Different level sets!")
    print(f"GNN: {len(gnn_levels)} levels")
    print(f"Hungarian: {len(hungarian_levels)} levels")
    print(f"Common: {len(common_levels)} levels")
    print()

gnn_total_time = sum(runtime for _, runtime, _, _ in gnn_results.values())
hungarian_total_time = sum(runtime for _, runtime, _, _ in hungarian_results.values())

gnn_successful = sum(1 for _, _, success, _ in gnn_results.values() if success)
hungarian_successful = sum(1 for _, _, success, _ in hungarian_results.values() if success)

gnn_total_nodes = sum(nodes for nodes, _, _, _ in gnn_results.values())
hungarian_total_nodes = sum(nodes for nodes, _, _, _ in hungarian_results.values())

print("AGGREGATE STATS:")
print("-" * 100)
print(f"{'Method':<25} {'Total Time':<15} {'Successful':<15} {'Total Nodes':<15} {'Avg Time/Level':<20}")
print(f"{'GNN [speed+dl]':<25} {gnn_total_time:>10.2f}s    {gnn_successful:>5}/{len(gnn_results):<6}  {gnn_total_nodes:>12,}   {gnn_total_time/len(gnn_results):>10.4f}s")
print(f"{'Hungarian+dl':<25} {hungarian_total_time:>10.2f}s    {hungarian_successful:>5}/{len(hungarian_results):<6}  {hungarian_total_nodes:>12,}   {hungarian_total_time/len(hungarian_results):>10.4f}s")
print()

time_ratio = gnn_total_time / hungarian_total_time
nodes_ratio = gnn_total_nodes / hungarian_total_nodes

if time_ratio < 1.0:
    print(f"GNN is {1/time_ratio:.2f}x FASTER than Hungarian overall!")
else:
    print(f"GNN is {time_ratio:.2f}x SLOWER than Hungarian overall")

print(f"GNN explores {nodes_ratio:.2f}x {'fewer' if nodes_ratio < 1 else 'more'} nodes than Hungarian")
print()

gnn_per_node = (gnn_total_time * 1000) / gnn_total_nodes
hungarian_per_node = (hungarian_total_time * 1000) / hungarian_total_nodes

print("PER-NODE EVALUATION TIME:")
print("-" * 100)
print(f"{'GNN [speed+dl]':<25} {gnn_per_node:>10.4f}ms per node")
print(f"{'Hungarian+dl':<25} {hungarian_per_node:>10.4f}ms per node")
print(f"Per-node overhead: GNN is {gnn_per_node/hungarian_per_node:.2f}x {'slower' if gnn_per_node > hungarian_per_node else 'faster'} per node")
print()

print("=" * 100)
print("LEVELS WHERE GNN BEATS HUNGARIAN (Faster Runtime)")
print()

gnn_wins = []
hungarian_wins = []
ties = []

for level_id in common_levels:
    gnn_nodes, gnn_time, gnn_success, gnn_sol = gnn_results[level_id]
    hun_nodes, hun_time, hun_success, hun_sol = hungarian_results[level_id]
    
    if gnn_success and hun_success:
        speedup = hun_time / gnn_time
        node_reduction = gnn_nodes / hun_nodes
        if gnn_time < hun_time * 0.95:  # 5% threshold to avoid noise
            gnn_wins.append((level_id, gnn_nodes, gnn_time, hun_nodes, hun_time, speedup, node_reduction))
        elif hun_time < gnn_time * 0.95:
            hungarian_wins.append((level_id, gnn_nodes, gnn_time, hun_nodes, hun_time, speedup, node_reduction))
        else:
            ties.append(level_id)

gnn_wins.sort(key=lambda x: x[5], reverse=True)
hungarian_wins.sort(key=lambda x: 1/x[5], reverse=True)

print(f"GNN wins on {len(gnn_wins)} out of {len(common_levels)} levels ({100*len(gnn_wins)/len(common_levels):.1f}%)\n")
print(f"Hungarian wins on {len(hungarian_wins)} out of {len(common_levels)} levels ({100*len(hungarian_wins)/len(common_levels):.1f}%)\n")
print()

node_wins = [(level_id, gnn_n, hun_n, gnn_n/hun_n, gnn_t, hun_t) 
             for level_id, gnn_n, gnn_t, hun_n, hun_t, _, node_ratio in gnn_wins + hungarian_wins
             if node_ratio < 0.5]
node_wins.sort(key=lambda x: x[3])

if node_wins:
    print(f"Found {len(node_wins)} levels where GNN explores <50% of Hungarian's nodes:\n")
    print(f"{'Level':<50} {'GNN Nodes':<12} {'Hun Nodes':<12} {'Reduction':<12} {'GNN wins?':<10}")
    print("-" * 100)
    for level_id, gnn_n, hun_n, ratio, gnn_t, hun_t in node_wins[:15]:
        level_name = level_id.split('/')[-1]
        wins = "Yes" if gnn_t < hun_t else "No"
        print(f"{level_name:<50} {gnn_n:>10}   {hun_n:>10}   {ratio:>10.2%}   {wins}")
else:
    print("No levels where GNN explores significantly fewer nodes.")

print()


print("=" * 100)
print("SUCCESS RATE COMPARISON")
print("=" * 100)
print()

gnn_only_success = []
hungarian_only_success = []
both_failed = []

for level_id in common_levels:
    gnn_success = gnn_results[level_id][2]
    hun_success = hungarian_results[level_id][2]
    
    if gnn_success and not hun_success:
        gnn_only_success.append(level_id)
    elif hun_success and not gnn_success:
        hungarian_only_success.append(level_id)
    elif not gnn_success and not hun_success:
        both_failed.append(level_id)

print(f"Both succeeded:        {len(common_levels) - len(gnn_only_success) - len(hungarian_only_success) - len(both_failed)} levels")
print(f"Only GNN succeeded:    {len(gnn_only_success)} levels")
print(f"Only Hungarian succeeded: {len(hungarian_only_success)} levels")
print(f"Both failed:           {len(both_failed)} levels")

