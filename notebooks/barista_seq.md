---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.3
  kernelspec:
    display_name: Julia 1.3.1
    language: julia
    name: julia-1.3
---

```julia
import Baysor

using barista_seq_mouse.csvtances
using Distributed
using DataFrames
using DataFramesMeta
using LinearAlgebra
using NearestNeighbors
using ProgressMeter
using Statistics
using StatsBase

import Colors
import CSV
import Images
import Plots
import Random
import NPZ
import MAT
```

## Read data

```julia
DATA_PATH = "../data/";
EXCHANGE_PATH = expanduser("~/mh/tmp_exchange/");

X_COL = :x;
Y_COL = :y;
GENE_COL = :gene;

GENE_COMPOSITION_NEIGHBORHOOD = 30;

MIN_MOLECULES_PER_GENE = 10;
MIN_MOLECULES_PER_CELL = 5;

NEW_COMPONENT_FRAC = 0.3;
NEW_COMPONENT_WEIGHT = 0.2;
CENTER_COMPONENT_WEIGHT = 0.6;
```

```julia
@time df_spatial, gene_names = Baysor.load_df("$DATA_PATH/barista_seq_mouse.csv"; 
    x_col=X_COL, y_col=Y_COL, gene_col=GENE_COL, min_molecules_per_gene=MIN_MOLECULES_PER_GENE);
df_spatial[!, :mol_id] .= 1:size(df_spatial, 1);

size(df_spatial)
```

```julia
@time neighb_cm = Baysor.neighborhood_count_matrix(df_spatial, GENE_COMPOSITION_NEIGHBORHOOD);
@time gene_color_transform = Baysor.gene_composition_transformation(neighb_cm);
```

## Run on a single frame to find parameters


### Select single frame

```julia
hcat([f(df_spatial.x) for f in [minimum, maximum]], [f(df_spatial.y) for f in [minimum, maximum]])
```

```julia
cur_df = @where(df_spatial, :x .> 0, :x .< 500, :y .> 0, :y .< 500) |> deepcopy;
size(cur_df)
```

```julia
@time cur_neighb_cm = Baysor.neighborhood_count_matrix(cur_df, GENE_COMPOSITION_NEIGHBORHOOD, length(gene_names));
@time cur_gene_colors = Baysor.gene_composition_colors(cur_neighb_cm, gene_color_transform);
```

```julia
Baysor.plot_cell_borders_polygons(cur_df; color=cur_gene_colors, size=(750, 750))
```

### Annotate and visualize data

```julia
# @time raw_neighb_cm = Baysor.neighborhood_count_matrix(cur_df, 30, length(gene_names), normalize=false);
```

```julia
# NPZ.npzwrite("$EXCHANGE_PATH/cur_neighb_cm_df.npy", raw_neighb_cm);
# CSV.write("$EXCHANGE_PATH/cur_neighb_cm_genes.csv", DataFrame(:genes => gene_names));
```

```julia
# annot_df = CSV.read("$EXCHANGE_PATH/cur_neighb_cm_annot.csv") |> DataFrame;
# annot_df[!, 1:3] .= String.(annot_df[!, 1:3]);
# first(annot_df, 3)
```

```julia
# Plots.plot(
#     Baysor.plot_cell_borders_polygons(cur_df, paper_polygons, cur_centers.centers; annotation=annot_df.l2),
#     Baysor.plot_cell_borders_polygons(cur_df, paper_polygons, cur_centers.centers; color=cur_gene_colors),
#     layout=2, size=(750*2, 750)
# )
```

### Initialize BM data

```julia
Baysor.append_confidence!(cur_df; nn_id=30, border_quantiles=(0.2, 0.6));

conf_colors = Baysor.map_to_colors(cur_df.confidence);
Baysor.plot_colorbar(conf_colors)
```

```julia
Baysor.plot_cell_borders_polygons(cur_df, polygons; color=conf_colors[:colors], size=(500, 500))
```

```julia
SCALE = 4;
SCALE_STD = "30%";
MIN_MOLECULES_PER_CELL = 20;

# @time bm_data = Baysor.initial_distribution_arr(cur_df, cur_centers; min_molecules_per_cell=MIN_MOLECULES_PER_CELL, scale=SCALE,
#     new_component_weight=NEW_COMPONENT_WEIGHT, center_component_weight=CENTER_COMPONENT_WEIGHT, n_frames=1, scale_std="50%")[1];

@time bm_data = Baysor.initial_distribution_arr(cur_df; min_molecules_per_cell=MIN_MOLECULES_PER_CELL, scale=SCALE,
    new_component_weight=NEW_COMPONENT_WEIGHT, n_frames=1, scale_std=SCALE_STD)[1];
```

```julia
# Baysor.append_dapi_brightness!(bm_data.x, dapi_arr);
# @time Baysor.adjust_field_weights_by_dapi!(bm_data, dapi_arr);
```

### Run

```julia
@time Baysor.bmm!(bm_data; n_iters=300, min_molecules_per_cell=MIN_MOLECULES_PER_CELL, 
#     split_period=100, n_expression_clusters=10, min_cluster_size=10, n_clustering_pcs=30, clustering_distance=Distances.CosineDist(),
    verbose=true, new_component_frac=NEW_COMPONENT_FRAC, log_step=50);
```

```julia
Baysor.plot_num_of_cells_per_iterarion(bm_data.tracer)
```

```julia
@time polygons = Baysor.boundary_polygons(bm_data, grid_step=1.0, dens_threshold=1e-7);
```

```julia
Baysor.plot_cell_borders_polygons(cur_df, polygons; color=cur_gene_colors)
```

```julia
# Plots.plot(
#     Baysor.plot_cell_borders_polygons(cur_df, polygons, cur_centers.centers; annotation=annot_df.l2),
#     Baysor.plot_cell_borders_polygons(cur_df, polygons, cur_centers.centers; color=cur_gene_colors),
#     layout=2, size=(750*2, 750)
# )
```

<!-- #region heading_collapsed=true -->
### Mixing scoring
<!-- #endregion -->

```julia hidden=true
# clust_per_mol = annot_df.l1;

# c_assignment = bm_data.assignment
# c_pap_assignment = bm_data.x.paper_cell
# c_rand_assignment = Random.shuffle(c_pap_assignment)

# cluster_mixing_score = Baysor.score_doublets_by_local_clusters(c_assignment, clust_per_mol, na_value=0.0)
# cluster_mixing_score_paper = Baysor.score_doublets_by_local_clusters(c_pap_assignment, clust_per_mol, na_value=0.0)
# cluster_mixing_score_rand = Baysor.score_doublets_by_local_clusters(c_rand_assignment, clust_per_mol, na_value=0.0)

# c_n_mols_per_cell = Baysor.count_array(c_assignment .+ 1)[2:end]
# c_n_mols_per_cell_paper = Baysor.count_array(c_pap_assignment .+ 1)[2:end]
# c_n_mols_per_cell_rand = Baysor.count_array(c_rand_assignment .+ 1)[2:end]

# p0, p1 = Baysor.plot_mixing_scores(cluster_mixing_score, c_n_mols_per_cell, cluster_mixing_score_paper, c_n_mols_per_cell_paper, 
#     cluster_mixing_score_rand, c_n_mols_per_cell_rand, format=:png, min_mols_per_cell=50);

# display(p0)
# display(p1)
```

```julia hidden=true
# t_xvals = 0:800
# t_scores = [[mean(ms[(n .>= i) .& (n .< (i + 50))] .* n[(n .>= i) .& (n .< (i + 50))]) for i in t_xvals] for (ms, n) in 
#         zip([cluster_mixing_score, cluster_mixing_score_paper, cluster_mixing_score_rand],
#             [c_n_mols_per_cell, c_n_mols_per_cell_paper, c_n_mols_per_cell_rand])]

# p = Plots.plot(xlabel="Size window", ylabel="Mean number of 'wrong' molecules")

# for (n,sc) in zip(["Baysor", "Paper", "Random"], t_scores)
#     p = Plots.plot!(t_xvals, sc, label=n)
# end
# p
```

```julia hidden=true
# t_xvals = 0:800
# t_scores = [[mean(ms[(n .>= i) .& (n .< (i + 50))]) for i in t_xvals] for (ms, n) in 
#         zip([cluster_mixing_score, cluster_mixing_score_paper, cluster_mixing_score_rand],
#             [c_n_mols_per_cell, c_n_mols_per_cell_paper, c_n_mols_per_cell_rand])]

# p = Plots.plot(xlabel="Size window", ylabel="Mixing score")

# for (n,sc) in zip(["Baysor", "Paper", "Random"], t_scores)
#     p = Plots.plot!(t_xvals, sc, label=n)
# end
# p
```

```julia hidden=true
# mixing_fractions = Baysor.score_doublets(bm_data; n_expression_clusters=10, n_pcs=30, min_cluster_size=10, 
#     min_molecules_per_cell=MIN_MOLECULES_PER_CELL, distance=Distances.CosineDist(), na_value=0.0)[2];

# doublet_mixing_colors = Baysor.map_to_colors(get.(Ref(mixing_fractions), bm_data.assignment, 0.0))
# Baysor.plot_colorbar(doublet_mixing_colors)
```

```julia hidden=true
# cluster_mixing_score = Baysor.score_doublets_by_local_clusters(c_assignment, annot_df.l1, na_value=0.0);
# cluster_mixing_colors = Baysor.map_to_colors(get.(Ref(cluster_mixing_score), bm_data.assignment, 0.0))
# Baysor.plot_colorbar(cluster_mixing_colors)
```

```julia hidden=true
# Baysor.plot_cell_borders_polygons(cur_df, polygons, cur_centers.centers; color=doublet_mixing_colors[:colors])
```

```julia hidden=true
# Plots.plot(
#     Baysor.plot_cell_borders_polygons(cur_df, polygons, cur_centers.centers; color=doublet_mixing_colors[:colors], title="Doublet mixing scores"),
#     Baysor.plot_cell_borders_polygons(cur_df, polygons, cur_centers.centers; color=cluster_mixing_colors[:colors], title="Cluster mixing scores"),
#     layout=2, size=(750 * 2, 750)
# )
```

```julia hidden=true
# Plots.plot(
#     Baysor.plot_cell_borders_polygons(cur_df, polygons, cur_centers.centers; annotation=annot_df.l2),
#     Baysor.plot_cell_borders_polygons(cur_df, polygons, cur_centers.centers; color=cur_gene_colors),
#     layout=2, size=(750*2, 750)
# )
```

## Run on whole dataset


### Initialize BM data

```julia
Baysor.append_confidence!(df_spatial; nn_id=30, border_quantiles=(0.2, 0.6));
```

```julia
@time bm_data_arr = Baysor.initial_distribution_arr(df_spatial; scale=SCALE, min_molecules_per_cell=MIN_MOLECULES_PER_CELL,
    new_component_weight=NEW_COMPONENT_WEIGHT, n_frames=8, scale_std=SCALE_STD, gene_num=maximum(df_spatial.gene));
```

```julia
# @showprogress for bm in bm_data_arr
#     Baysor.append_dapi_brightness!(bm.x, dapi_arr);
#     Baysor.adjust_field_weights_by_dapi!(bm, dapi_arr);
# end
```

### Parallel run

```julia
@time bm_data_arr_res = deepcopy(bm_data_arr);
```

```julia
@time bm_data_merged = Baysor.run_bmm_parallel!(bm_data_arr_res, 350; min_molecules_per_cell=MIN_MOLECULES_PER_CELL, 
#     split_period=100, n_expression_clusters=10, min_cluster_size=10, n_clustering_pcs=30, clustering_distance=Distances.CosineDist(),
    new_component_frac=NEW_COMPONENT_FRAC, assignment_history_depth=100);
```

```julia
Baysor.plot_num_of_cells_per_iterarion(bm_data_merged.tracer)
```

```julia
Baysor.drop_unused_components!(bm_data_merged; min_n_samples=20, force=true);
```

```julia
@time segmentation_df = Baysor.get_segmentation_df(bm_data_merged, gene_names);
@time cell_stat_df = Baysor.get_cell_stat_df(bm_data_merged, segmentation_df);
```

```julia
# t_bm = deepcopy(bm_data_arr_res[19]);
```

```julia
t_df = @where(Baysor.get_segmentation_df(bm_data_merged; use_assignment_history=false), :x .> 500, :x .< 1000, :y .> 500, :y .< 1000) |> deepcopy;
size(t_df)
```

```julia
@time t_neighb_cm = Baysor.neighborhood_count_matrix(t_df, GENE_COMPOSITION_NEIGHBORHOOD, length(gene_names));
@time t_gene_colors = Baysor.gene_composition_colors(t_neighb_cm, gene_color_transform);
```

```julia
@time t_polygons = Baysor.boundary_polygons(t_df, t_df.cell, grid_step=1.0, 
    dens_threshold=1e-7, bandwidth=0.5, min_molecules_per_cell=5);
```

```julia
Baysor.plot_cell_borders_polygons(t_df, t_polygons; color=cur_gene_colors)
```

```julia
out_stat_names = names(cell_stat_df)[.!in.(names(cell_stat_df), Ref([:has_center, :x_prior, :y_prior]))];

CSV.write("../output/barista_seq/cm.tsv", Baysor.convert_segmentation_df_to_cm(segmentation_df), delim="\t");
CSV.write("../output/barista_seq/cell_stat.csv", cell_stat_df[:, out_stat_names]);
CSV.write("../output/barista_seq/segmentation.csv", segmentation_df[:, names(segmentation_df) .!= :mol_id]);
```
