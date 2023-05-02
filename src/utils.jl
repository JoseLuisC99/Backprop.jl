import ShowGraphviz
import .Jacobi: Tensor as Tensor

function explicit_graph(t::Tensor)::Tuple{Set{Tensor}, Set{Tuple{Tensor, Tensor}}}
    nodes = Set{Tensor}()
    edges = Set{Tuple{Tensor, Tensor}}()
    next = Vector{Tensor}()

    push!(next, t)
    while !isempty(next)
        v = popfirst!(next)
        for t in v.ctx
            push!(next, t)
            push!(edges, (t, v))
        end
        push!(nodes, v)
    end

    return nodes, edges
end

function plot_tensor(t::Tensor; rankdir::String = "TB")::ShowGraphviz.DOT
    nodes, edges = explicit_graph(t)
    id_node = Dict([
        (v, "n" * lpad(i, 2, "0")) for (i, v) in enumerate(nodes)
    ])

    g_str = """
    digraph tensor {
    fontname="Helvetica,Arial,sans-serif"
    node [fontname="Helvetica,Arial,sans-serif"]
    edge [fontname="Helvetica,Arial,sans-serif"]
    rankdir=$rankdir;
    node [shape=record];
    """
    t_id = 0
    for node in nodes
        g_str *= """
        $(id_node[node]) [label="$(node.name == "" ? "t$(t_id = t_id + 1)" : node.name)\\n|{shape:|req_grad:}|{{$(size(node))}|{$(node.autograd.requires_grad)}}"];
        """
    end
    for (from, to) in edges
        g_str *= """
        $(id_node[from]) -> $(id_node[to]);
        """
    end
    g_str *= "}"

    return ShowGraphviz.DOT(g_str)
end