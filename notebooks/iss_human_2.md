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
GENE_COL = :gene;

GENE_COMPOSITION_NEIGHBORHOOD = 5;

MIN_MOLECULES_PER_GENE = 10;
MIN_MOLECULES_PER_CELL = 5;

NEW_COMPONENT_FRAC = 0.3;
NEW_COMPONENT_WEIGHT = 0.2;
CENTER_COMPONENT_WEIGHT = 0.6;
```

```julia
# @time df_spatial, gene_names = Baysor.load_df("$DATA_PATH/merfish_coords_perprocessed.csv"; 
@time df_spatial, gene_names = Baysor.load_df("$DATA_PATH/ISS_1_spot_table.csv"; 
    x_col=X_COL, y_col=Y_COL, gene_col=GENE_COL, min_molecules_per_gene=MIN_MOLECULES_PER_GENE);
df_spatial[!, :mol_id] .= 1:size(df_spatial, 1);

size(df_spatial)
```

```julia
df_spatial.x .= round.(Int, df_spatial.x ./ 0.1625);
df_spatial.y .= round.(Int, df_spatial.y ./ 0.1625);
```

```julia
@time neighb_cm = Baysor.neighborhood_count_matrix(df_spatial, GENE_COMPOSITION_NEIGHBORHOOD);
@time gene_color_transform = Baysor.gene_composition_transformation(neighb_cm);
```

```julia
@time dapi_arr = MAT.matread("$DATA_PATH/Base_1_stitched-1.tif.mat")["I"];
```

```julia
@time max_brightness = quantile(dapi_arr[dapi_arr .> 0.1], 0.99);

dapi_arr[dapi_arr .> max_brightness] .= max_brightness;
dapi_arr = dapi_arr ./ max_brightness;
```

```julia
@time seg_mask = (dapi_arr .> Images.otsu_threshold(dapi_arr));
Plots.heatmap(seg_mask[5000:10000, 5000:10000], format=:png)
```

```julia
@time cell_centers = Baysor.extract_centers_from_mask(seg_mask, min_segment_size=20);
size(cell_centers.centers, 1), cell_centers.scale_estimate
```

Append confidence based on DAPI

```julia
# Baysor.append_confidence!(df_spatial, seg_mask, nn_id = 10);
```

## Run on a single frame to find parameters


### Select single frame

```julia
hcat([f(df_spatial.x) for f in [minimum, maximum]], [f(df_spatial.y) for f in [minimum, maximum]])
```

```julia
cur_df = @where(df_spatial, :x .> 10000, :x .< 13000, :y .> 10000, :y .< 13000) |> deepcopy;
cur_centers = Baysor.subset_by_coords(cell_centers, cur_df);
size(cur_df)
```

```julia
@time cur_neighb_cm = Baysor.neighborhood_count_matrix(cur_df, GENE_COMPOSITION_NEIGHBORHOOD, length(gene_names));
@time cur_gene_colors = Baysor.gene_composition_colors(cur_neighb_cm, gene_color_transform);
```

```julia
Plots.heatmap(dapi_arr[Int(minimum(cur_df.y)):Int(maximum(cur_df.y)), Int(minimum(cur_df.x)):Int(maximum(cur_df.x))], format=:png, size=(600, 600), c=:ice)
Baysor.plot_cell_borders_polygons!(cur_df, cur_centers.centers; color=cur_gene_colors, point_size=2, offset=(-minimum(cur_df.x), -minimum(cur_df.y)), 
    alpha=0.2, polygon_line_color=Colors.colorant"red")
```

```julia
Baysor.plot_cell_borders_polygons!(cur_df; color=cur_gene_colors, size=(750, 750))
```

### Annotate and visualize data

```julia
1
```

```julia
@time raw_neighb_cm = Baysor.neighborhood_count_matrix(cur_df, 30, length(gene_names), normalize=false);
```

```julia
NPZ.npzwrite("$EXCHANGE_PATH/cur_neighb_cm_df.npy", raw_neighb_cm);
CSV.write("$EXCHANGE_PATH/cur_neighb_cm_genes.csv", DataFrame(:genes => gene_names));
```

```julia
annot_df = CSV.read("$EXCHANGE_PATH/cur_neighb_cm_annot.csv") |> DataFrame;
annot_df[!, 1:3] .= String.(annot_df[!, 1:3]);
first(annot_df, 3)
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
# SCALE = 10 / 0.1625;
# SCALE = 100;
SCALE = 75;

# @time bm_data = Baysor.initial_distribution_arr(cur_df, cur_centers; min_molecules_per_cell=MIN_MOLECULES_PER_CELL, scale=SCALE,
#     new_component_weight=NEW_COMPONENT_WEIGHT, center_component_weight=CENTER_COMPONENT_WEIGHT, n_frames=1, scale_std="50%")[1];

@time bm_data = Baysor.initial_distribution_arr(cur_df; min_molecules_per_cell=MIN_MOLECULES_PER_CELL, scale=SCALE,
    new_component_weight=NEW_COMPONENT_WEIGHT, n_frames=1, scale_std="40%")[1];
```

```julia
# Baysor.append_dapi_brightness!(bm_data.x, dapi_arr);
# @time Baysor.adjust_field_weights_by_dapi!(bm_data, dapi_arr);
```

### Run

```julia
@time Baysor.bmm!(bm_data; n_iters=500, min_molecules_per_cell=MIN_MOLECULES_PER_CELL, 
#     split_period=100, n_expression_clusters=10, min_cluster_size=10, n_clustering_pcs=30, clustering_distance=Distances.CosineDist(),
    verbose=true, new_component_frac=NEW_COMPONENT_FRAC, log_step=50);
```

```julia
Baysor.plot_num_of_cells_per_iterarion(bm_data.tracer)
```

```julia
@time polygons = Baysor.boundary_polygons(bm_data, grid_step=20.0, dens_threshold=1e-7);
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
@time bm_data_arr = Baysor.initial_distribution_arr(df_spatial; scale=SCALE, min_molecules_per_cell=MIN_MOLECULES_PER_CELL,
    new_component_weight=NEW_COMPONENT_WEIGHT, n_frames=20, scale_std="40%");
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
bm_data_merged = bm_data_arr_res;
```

```julia
@time bm_data_merged = Baysor.run_bmm_parallel!(bm_data_arr_res, 500; min_molecules_per_cell=MIN_MOLECULES_PER_CELL, 
#     split_period=100, n_expression_clusters=10, min_cluster_size=10, n_clustering_pcs=30, clustering_distance=Distances.CosineDist(),
    new_component_frac=NEW_COMPONENT_FRAC);
```

```julia
Baysor.plot_num_of_cells_per_iterarion(bm_data_merged.tracer)
```

```julia
Baysor.drop_unused_components!(bm_data_merged; min_n_samples=3, force=true);
```

```julia
@time segmentation_df = Baysor.get_segmentation_df(bm_data_merged, gene_names);
@time cell_stat_df = Baysor.get_cell_stat_df(bm_data_merged);
```

```julia
# t_bm = deepcopy(bm_data_arr_res[19]);
```

```julia
t_df = @where(segmentation_df, :x .> 25000, :x .< 30000, :y .> 30000, :y .< 35000) |> deepcopy;
# t_df = @where(segmentation_df, :x .> 10000, :x .< 13000, :y .> 10000, :y .< 13000) |> deepcopy;
# t_df = @where(Baysor.get_segmentation_df(t_bm), :x .> 10000, :x .< 13000, :y .> 10000, :y .< 13000);

size(t_df)
```

```julia
@time t_neighb_cm = Baysor.neighborhood_count_matrix(t_df, GENE_COMPOSITION_NEIGHBORHOOD, length(gene_names));
@time t_gene_colors = Baysor.gene_composition_colors(t_neighb_cm, gene_color_transform);
```

```julia
@time polygons = Baysor.boundary_polygons(t_df, t_df.cell, grid_step=5.0, 
    dens_threshold=1e-7, bandwidth=10.0, min_molecules_per_cell=5);
```

```julia
seg_df_per_cell = Baysor.split(segmentation_df, segmentation_df.cell .+ 1; max_factor=length(bm_data_merged.components)+1)[2:end];
```

```julia
Baysor.plot_cell_borders_polygons(t_df, polygons; color=cur_gene_colors)
```

```julia
out_stat_names
```

```julia
out_stat_names = names(cell_stat_df)[.!in.(names(cell_stat_df), Ref([:has_center, :x_prior, :y_prior]))];

CSV.write("../output/iss_human_2/cm.tsv", Baysor.convert_segmentation_df_to_cm(segmentation_df), delim="\t")
CSV.write("../output/iss_human_2/cell_stat.csv", cell_stat_df[:, out_stat_names]);
CSV.write("../output/iss_human_2/segmentation.csv", segmentation_df[:, names(segmentation_df) .!= :mol_id]);
```
