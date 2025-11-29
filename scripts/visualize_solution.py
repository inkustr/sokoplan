#!/usr/bin/env python3
"""Reconstruct and visualize a solution from labels.jsonl."""

import argparse
import json
from sokoban_core.state import State


def state_to_ascii(s: State) -> str:
    """Convert State to ASCII visualization."""
    player_mask = 1 << s.player
    lines = []
    for y in range(s.height):
        line = []
        for x in range(s.width):
            pos = 1 << (y * s.width + x)
            
            if s.walls & pos:
                line.append('#')
            elif s.boxes & pos and s.goals & pos:
                line.append('*')
            elif s.boxes & pos:
                line.append('$')
            elif s.goals & pos and player_mask == pos:
                line.append('+')
            elif s.goals & pos:
                line.append('.')
            elif player_mask == pos:
                line.append('@')
            else:
                line.append(' ')
        lines.append(''.join(line))
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Visualize solution from labels')
    parser.add_argument('--labels', required=True, help='Path to labels.jsonl file')
    parser.add_argument('--level_id', required=True, help='Level ID to visualize')
    parser.add_argument('--out', required=True, help='Output text file')
    args = parser.parse_args()

    
    records = []
    total_lines = 0
    with open(args.labels, 'r') as f:
        for line in f:
            total_lines += 1
            rec = json.loads(line)
            if rec.get('level_id') == args.level_id:
                records.append(rec)
    
    if not records:
        print(f"No records found for level: {args.level_id}")
        return
    
    records.sort(key=lambda r: r['y'], reverse=True)
    
    states = []
    for i, rec in enumerate(records):
        state = State(
            width=rec['width'],
            height=rec['height'],
            walls=rec['walls'],
            goals=rec['goals'],
            boxes=rec['boxes'],
            player=rec['player'],
            board_mask=rec['board_mask']
        )
        states.append((state, rec['y']))
    
    with open(args.out, 'w') as f:
        f.write(f"Total solution length: {states[0][1]} moves\n")
        f.write(f"Sampled states: {len(states)}\n")
        f.write(f"Boxes: {bin(states[0][0].boxes).count('1')}\n")
        f.write(f"Level size: {states[0][0].width}x{states[0][0].height}\n")
        f.write("\n")
        
        for i, (state, cost_to_go) in enumerate(states):
            move_num = states[0][1] - cost_to_go
            boxes_on_goals = bin(state.boxes & state.goals).count('1')
            total_boxes = bin(state.boxes).count('1')
            
            f.write(f"Move {move_num}/{states[0][1]} (cost-to-go: {cost_to_go}, boxes on goals: {boxes_on_goals}/{total_boxes})\n")
            f.write(state_to_ascii(state))
            f.write("\n\n")
    

if __name__ == '__main__':
    main()

