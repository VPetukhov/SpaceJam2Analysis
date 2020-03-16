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

using Distances
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

X_COL = :x_um;
Y_COL = :y_um;
GENE_COL = :target;

GENE_COMPOSITION_NEIGHBORHOOD = 20;

MIN_MOLECULES_PER_GENE = 10;
MIN_MOLECULES_PER_CELL = 5;

NEW_COMPONENT_FRAC = 0.3;
NEW_COMPONENT_WEIGHT = 0.2;
CENTER_COMPONENT_WEIGHT = 0.6;
```

```julia
@time df_spatial, gene_names = Baysor.load_df("$DATA_PATH/DARTFISH_DecodedSpots_Mm_20190513.csv"; 
    x_col=X_COL, y_col=Y_COL, gene_col=GENE_COL, min_molecules_per_gene=MIN_MOLECULES_PER_GENE);
df_spatial[!, :mol_id] .= 1:size(df_spatial, 1);

size(df_spatial)
```

```julia
@time neighb_cm = Baysor.neighborhood_count_matrix(df_spatial, GENE_COMPOSITION_NEIGHBORHOOD);
@time gene_color_transform = Baysor.gene_composition_transformation(neighb_cm);
```

```julia
# @time dapi_arr = MAT.matread("$DATA_PATH/Base_1_stitched-1.tif.mat")["I"];
```

```julia
# @time max_brightness = quantile(dapi_arr[dapi_arr .> 0.1], 0.99);

# dapi_arr[dapi_arr .> max_brightness] .= max_brightness;
# dapi_arr = dapi_arr ./ max_brightness;
```

```julia
# @time seg_mask = (dapi_arr .> Images.otsu_threshold(dapi_arr));
# Plots.heatmap(seg_mask[5000:10000, 5000:10000], format=:png)
```

```julia
# @time cell_centers = Baysor.extract_centers_from_mask(seg_mask, min_segment_size=20);
# size(cell_centers.centers, 1), cell_centers.scale_estimate
```

Append confidence based on DAPI

```julia
# Baysor.append_confidence!(df_spatial, seg_mask, nn_id = 10);
```

## Run


### Select single frame

```julia
hcat([f(df_spatial.x) for f in [minimum, maximum]], [f(df_spatial.y) for f in [minimum, maximum]])
```

```julia
@time gene_colors = Baysor.gene_composition_colors(neighb_cm, gene_color_transform);
```

```julia
# Plots.heatmap(dapi_arr[Int(minimum(cur_df.y)):Int(maximum(cur_df.y)), Int(minimum(cur_df.x)):Int(maximum(cur_df.x))], format=:png, size=(600, 600), c=:ice)
# Baysor.plot_cell_borders_polygons(cur_df; color=cur_gene_colors, point_size=2, offset=(-minimum(cur_df.x), -minimum(cur_df.y)), 
#     alpha=0.2, polygon_line_color=Colors.colorant"red")
```

```julia
Baysor.plot_cell_borders_polygons(df_spatial; color=gene_colors, size=(1600, 400))
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

```julia
Baysor.append_confidence!(df_spatial; nn_id=10, border_quantiles=(0.2, 0.8));

conf_colors = Baysor.map_to_colors(df_spatial.confidence);
Baysor.plot_colorbar(conf_colors)
```

```julia
Baysor.plot_cell_borders_polygons(df_spatial; color=conf_colors[:colors], size=(1600, 400))
```

### Run

```julia
SCALE = 50;

@time bm_data = Baysor.initial_distribution_arr(df_spatial; min_molecules_per_cell=MIN_MOLECULES_PER_CELL, scale=SCALE,
    new_component_weight=NEW_COMPONENT_WEIGHT, n_frames=1, scale_std="40%")[1];
```

```julia
@time Baysor.bmm!(bm_data; n_iters=350, min_molecules_per_cell=MIN_MOLECULES_PER_CELL, assignment_history_depth=100,
#     split_period=100, n_expression_clusters=10, min_cluster_size=10, n_clustering_pcs=30, clustering_distance=Distances.CosineDist(),
    verbose=true, new_component_frac=NEW_COMPONENT_FRAC, log_step=50);
```

```julia
Baysor.plot_num_of_cells_per_iterarion(bm_data.tracer)
```

```julia
bm_data.assignment = Baysor.estimate_assignment_by_history(bm_data)[1];
Baysor.maximize!(bm_data, MIN_MOLECULES_PER_CELL)
```

```julia
@time polygons = Baysor.boundary_polygons(bm_data, grid_step=5.0, bandwidth=5.0, dens_threshold=1e-7);
```

```julia
Baysor.plot_cell_borders_polygons(df_spatial, polygons; color=gene_colors, size=(1600, 400))
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

## Save

```julia
Baysor.drop_unused_components!(bm_data; min_n_samples=3, force=true);
```

```julia
@time segmentation_df = Baysor.get_segmentation_df(bm_data, gene_names);
@time cell_stat_df = Baysor.get_cell_stat_df(bm_data, segmentation_df);
```

```julia
out_stat_names = names(cell_stat_df)[.!in.(names(cell_stat_df), Ref([:has_center, :x_prior, :y_prior]))];

CSV.write("../output/dartfish_mouse/cm.tsv", Baysor.convert_segmentation_df_to_cm(segmentation_df), delim="\t")
CSV.write("../output/dartfish_mouse/cell_stat.csv", cell_stat_df[:, out_stat_names]);
CSV.write("../output/dartfish_mouse/segmentation.csv", segmentation_df[:, names(segmentation_df) .!= :mol_id]);
```
