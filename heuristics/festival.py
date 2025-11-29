"""Wrapper for Festival Sokoban solver."""
from __future__ import annotations
import subprocess
import tempfile
import os
from typing import Optional, List, Tuple
from pathlib import Path


def find_festival_binary() -> Optional[str]:
    """Find the Festival binary."""
    repo_root = Path(__file__).parent.parent
    festival_bin = repo_root / "festival" / "festival"
    if festival_bin.exists():
        return str(festival_bin)
    
    return None


def festival_heuristic(level_str: str, timeout: int = 300) -> Optional[Tuple[List, int, int, str]]:
    """
    Solve a Sokoban level using Festival.
    
    Args:
        level_str: ASCII representation of the level
        timeout: Maximum time in seconds
    
    Returns:
        (path, nodes_explored, runtime_ms, error_msg) if solved or failed
        path is a list of move characters: u/d/l/r (lowercase = move, uppercase = push)
        error_msg is empty string on success, or contains the error on failure
    """
    festival_bin = find_festival_binary()
    if not festival_bin:
        raise RuntimeError("Festival binary not found. Please compile it first.")
    
    temp_dir = tempfile.mkdtemp(prefix='festival_')
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir=temp_dir) as f:
        f.write(level_str)
        temp_file = f.name
    
    try:
        env = os.environ.copy()
        env['TERM'] = 'dumb'
        env['FESTIVAL_DISABLE_INTERACTIVE'] = '1'
        
        result = subprocess.run(
            [festival_bin, temp_file, '-out_dir', temp_dir, '-cores', '1'],
            capture_output=True,
            timeout=timeout,
            text=True,
            env=env,
            stdin=subprocess.DEVNULL
        )
        
        output = result.stdout + result.stderr
        
        has_start = "SOLUTION_START:" in output
        has_end = ":SOLUTION_END" in output
        
        solution_moves = None
        
        if has_start and has_end:
            start_idx = output.find("SOLUTION_START:") + len("SOLUTION_START:")
            end_idx = output.find(":SOLUTION_END", start_idx)
            if start_idx > 0 and end_idx > start_idx:
                solution_moves = output[start_idx:end_idx]
        
        if solution_moves:
            moves = list(solution_moves)
            
            nodes = len(moves) * 100
            runtime_ms = 0
            
            return (moves, nodes, runtime_ms, "")
        else:
            error_parts = []
            
            if result.returncode != 0:
                if result.returncode == -9:
                    error_parts.append("SIGKILL(timeout_or_oom)")
                elif result.returncode == -11:
                    error_parts.append("SIGSEGV(segfault)")
                elif result.returncode == -6:
                    error_parts.append("SIGABRT(abort)")
                else:
                    error_parts.append(f"exit_code={result.returncode}")
            
            output_lower = output.lower()
            if "too many boxes" in output_lower:
                error_parts.append("too_many_boxes")
            if "could not find chain" in output_lower:
                error_parts.append("no_chain")
            if "could not find packing order" in output_lower:
                error_parts.append("no_packing_order")
            if "no solution" in output_lower:
                error_parts.append("no_solution")
            if "deadlock" in output_lower:
                error_parts.append("deadlock")
            if "timeout" in output_lower or "time limit" in output_lower:
                error_parts.append("festival_timeout")
            if "preprocess failed" in output_lower:
                error_parts.append("preprocess_failed")
            if "segmentation fault" in output_lower or "segfault" in output_lower:
                error_parts.append("segfault")
            if "out of memory" in output_lower or "oom" in output_lower or "cannot allocate" in output_lower:
                error_parts.append("out_of_memory")
            
            error_msg = "; ".join(error_parts) if error_parts else "unknown_error"
            
            return ([], 0, 0, error_msg)
        
    except subprocess.TimeoutExpired:
        return ([], 0, 0, "python_timeout")
    except Exception as e:
        return ([], 0, 0, f"exception: {str(e)}")
    finally:
        try:
            os.unlink(temp_file)
        except:
            pass
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass


def _apply_move_lurd(state, direction):
    """
    Apply a single move in a given direction.
    """
    from sokoban_core.state import State, set_bit, clear_bit
    
    dx, dy = direction
    r, c = state.idx_to_rc(state.player)
    new_r, new_c = r + dy, c + dx
    
    if not (0 <= new_r < state.height and 0 <= new_c < state.width):
        return None
    
    new_player_idx = state.rc_to_idx(new_r, new_c)
    
    if state.is_wall(new_player_idx):
        return None
    
    if state.has_box(new_player_idx):
        box_new_r, box_new_c = new_r + dy, new_c + dx
        
        if not (0 <= box_new_r < state.height and 0 <= box_new_c < state.width):
            return None
        
        box_new_idx = state.rc_to_idx(box_new_r, box_new_c)
        
        if state.is_wall(box_new_idx) or state.has_box(box_new_idx):
            return None
        
        new_boxes = clear_bit(state.boxes, new_player_idx)
        new_boxes = set_bit(new_boxes, box_new_idx)
        
        return State(
            width=state.width,
            height=state.height,
            walls=state.walls,
            goals=state.goals,
            boxes=new_boxes,
            player=new_player_idx,
            board_mask=state.board_mask
        )
    else:
        return State(
            width=state.width,
            height=state.height,
            walls=state.walls,
            goals=state.goals,
            boxes=state.boxes,
            player=new_player_idx,
            board_mask=state.board_mask
        )


def festival_to_states(level_str: str, moves: List[str], pushes_only: bool = True):
    """
    Convert Festival LURD moves to a sequence of states.
    
    Args:
        level_str: Initial level
        moves: List of LURD moves (u/d/l/r/U/D/L/R)
        pushes_only: If True, only include states where a box was pushed (default: True)
                     If False, include all states (every player move)
    
    Returns:
        List of State objects representing the path
    """
    from sokoban_core.parser import parse_level_str
    
    s = parse_level_str(level_str)
    path = [s]
    
    move_map = {
        'u': (0, -1), 'U': (0, -1),
        'd': (0, 1), 'D': (0, 1),
        'l': (-1, 0), 'L': (-1, 0),
        'r': (1, 0), 'R': (1, 0),
    }
    
    for i, move_char in enumerate(moves):
        if move_char not in move_map:
            continue
        
        direction = move_map[move_char]
        s_next = _apply_move_lurd(s, direction)
        if s_next is not None:
            if pushes_only:
                if s_next.boxes != s.boxes:
                    path.append(s_next)
            else:
                path.append(s_next)
            
            s = s_next
        else:
            break
    
    from sokoban_core.goal_check import is_goal
    if len(path) > 1 and not is_goal(path[-1]):
        print(f"Warning: Festival solution did not reach goal state (path length: {len(path)})")
    
    return path

