def read_graph(filename):
    with open(filename, 'r') as f:
        n = int(f.readline().strip())
        edges = []
        for line in f:
            if line.strip():
                u, v = map(int, line.strip().split())
                edges.append((u, v))
    return n, edges

def construct_signed_graph(n, edges):
    # We'll maintain a mapping from special named nodes to IDs.
    # s=0, l=1, m=2, r=3, t=4, h1=5, h2=6
    node_count = 7  # we have these nodes so far: s=0,l=1,m=2,r=3,t=4,h1=5,h2=6

    # We'll store edges in a list as (from, to, sign)
    signed_edges = []

    # Helper: add a positive edge
    def add_pos(a,b):
        signed_edges.append((a,b,1))
    # Helper: add a negative edge
    def add_neg(a,b):
        signed_edges.append((a,b,-1))

    # 1. Create the path s-l-m-r-t with positive edges
    add_pos(0,1) # s->l
    add_pos(1,2) # l->m
    add_pos(2,3) # m->r
    add_pos(3,4) # r->t

    # s neg connected to m,r,t
    add_neg(0,2)
    add_neg(0,3)
    add_neg(0,4)

    # t neg connected to l,m
    add_neg(4,1)
    add_neg(4,2)

    # 2. Introduce h1,h2:
    # h1=5, h2=6
    add_pos(5,6) # h1->h2
    add_pos(0,5) # h1->s

    # h1 neg connected to r,t
    add_neg(5,3)
    add_neg(5,4)

    # h2 neg connected to s,l,r,t
    add_neg(6,0)
    add_neg(6,1)
    add_neg(6,3)
    add_neg(6,4)

    vertex_map = {}
    prev_h = 6  # start from h2

    # Create quadruple of nodes for each original vertex
    for orig_v in range(n):
        x_u = node_count; node_count += 1
        x = node_count; node_count += 1
        x_l = node_count; node_count += 1
        x_h = node_count; node_count += 1
        vertex_map[orig_v] = (x_u, x, x_l, x_h)

    for orig_v in range(n):
        (x_u, x, x_l, x_h) = vertex_map[orig_v]

        # pos edges
        add_pos(prev_h, x_u)
        add_pos(x_u, x)
        add_pos(x_l, x)
        add_pos(x_l, x_h)

        # neg edges to s and t
        add_neg(0, x_u)
        add_neg(0, x)
        add_neg(0, x_l)
        add_neg(0, x_h)
        add_neg(4, x_u)
        add_neg(4, x)
        add_neg(4, x_l)
        add_neg(4, x_h)

        # neg connect x to m
        add_neg(x, 2)

        # neg connect x_h to l and r
        add_neg(x_h, 1)
        add_neg(x_h, 3)

        prev_h = x_h

    final_vertex = node_count
    node_count += 1

    add_pos(prev_h, final_vertex)
    add_pos(final_vertex, 4)
    add_neg(final_vertex, 0)
    add_neg(final_vertex, 1)

    # Pairwise neg connect all v nodes
    all_vs = [vertex_map[i][1] for i in range(n)]
    for i in range(len(all_vs)):
        for j in range(i+1, len(all_vs)):
            add_neg(all_vs[i], all_vs[j])

    # For each directed edge (x,y) in the instance:
    # Introduce neg edges {y, x_u} and {y, x_l}
    for (xx,yy) in edges:
        x_u, x_v, x_l, x_h = vertex_map[xx]
        y_u, y_v, y_l, y_h = vertex_map[yy]

        add_neg(y_v, x_u)
        add_neg(y_v, x_l)

    return signed_edges, node_count

def main():
    # Example usage:
    n, edges = read_graph("data/directed_graph.txt")
    signed_edges, node_count = construct_signed_graph(n, edges)

    # Separate positive and negative edges
    pos_edges = [(f,t) for (f,t,s) in signed_edges if s == 1]
    neg_edges = [(f,t) for (f,t,s) in signed_edges if s == -1]

    # Determine max_node
    max_node = max(max(f for f,t,s in signed_edges), max(t for f,t,s in signed_edges))

    # Create a custom order:
    # First from 5 up to max_node
    # Then from 5 down to 0
    order_map = {}
    current_index = 0

    # Go upwards from 5 to max_node
    for v in range(5, max_node+1):
        order_map[v] = current_index
        current_index += 1
    # Then go down from 5 to 0
    for v in range(5, -1, -1):
        # If v already assigned, skip (for v=5)
        if v not in order_map:
            order_map[v] = current_index
            current_index += 1

    # For any nodes not covered (if any), assign them after current_index
    # (Should not normally happen if the construction always fits in the pattern)
    for v in range(max_node+1):
        if v not in order_map:
            order_map[v] = current_index
            current_index += 1

    # Sort positive edges by custom order
    # Sort by order of fromNode, then by order of toNode
    def pos_sort_key(e):
        f, t = e
        return (order_map.get(f, 999999), order_map.get(t, 999999))

    pos_edges.sort(key=pos_sort_key)

    # Print output
    print("#FromNodeId\tToNodeId\tSign")
    # Print positive edges first
    for (f,t) in pos_edges:
        print(f"{f}\t{t}\t1")
    # Then print negative edges
    for (f,t) in neg_edges:
        print(f"{f}\t{t}\t-1")

    # Also write to a file
    with open("data/graph6.txt", "w") as outf:
        outf.write("#FromNodeId\tToNodeId\tSign\n")
        for (f,t) in pos_edges:
            outf.write(f"{f}\t{t}\t1\n")
        for (f,t) in neg_edges:
            outf.write(f"{f}\t{t}\t-1\n")

if __name__ == "__main__":
    main()
