import mesh_tensorflow as mtf

graph = mtf.Graph()
mesh = mtf.Mesh(graph, "my_mesh")

batch             = mtf.Dimension('batch', 1)
seq_len           = mtf.Dimension('sequence', 1024)
dim               = mtf.Dimension('dim', 512)
dim_head          = mtf.Dimension('dim_head', 12)
dim_features_head = mtf.Dimension('dim_features_head', 64)