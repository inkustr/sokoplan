from sokoban_core.parser import parse_level_str
from gnn.graphs import grid_to_graph

def test_grid_to_graph_smoke():
    lvl = """
#####
#.@ #
# $ #
# . #
#####
"""
    s = parse_level_str(lvl)
    data, idx2nid = grid_to_graph(s)
    assert data.x.shape[1] == 7
    assert data.edge_index.shape[0] == 2
    assert data.edge_attr.shape[0] == data.edge_index.shape[1]
    assert data.edge_attr.shape[1] == 4
    assert len(idx2nid) > 0