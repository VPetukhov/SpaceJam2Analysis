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

X_COL = :pos1;
Y_COL = :pos2;
GENE_COL = :name;

GENE_COMPOSITION_NEIGHBORHOOD = 5;

MIN_MOLECULES_PER_GENE = 10;
MIN_MOLECULES_PER_CELL = 5;

NEW_COMPONENT_FRAC = 0.3;
NEW_COMPONENT_WEIGHT = 0.2;
CENTER_COMPONENT_WEIGHT = 0.6;
```

```julia
# @time df_spatial, gene_names = Baysor.load_df("$DATA_PATH/merfish_coords_perprocessed.csv"; 
@time df_spatial, gene_names = Baysor.load_df("$DATA_PATH/iss_human_hca10_2_spots.csv"; 
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
cur_df = @where(df_spatial, :x .> 15000, :x .< 20000, :y .> 15000, :y .< 20000) |> deepcopy;
size(cur_df)
```

```julia
Baysor.append_confidence!(cur_df; nn_id=25, border_quantiles=(0.2, 0.7));

conf_colors = Baysor.map_to_colors(cur_df.confidence);
Baysor.plot_colorbar(conf_colors)
```

```julia
Baysor.plot_cell_borders_polygons(cur_df; color=conf_colors[:colors], size=(500, 500))
```

```julia
@time cur_neighb_cm = Baysor.neighborhood_count_matrix(cur_df, GENE_COMPOSITION_NEIGHBORHOOD, length(gene_names));
@time cur_gene_colors = Baysor.gene_composition_colors(cur_neighb_cm, gene_color_transform);
```

```julia
Baysor.plot_cell_borders_polygons(cur_df; color=cur_gene_colors, size=(750, 750))
```

```julia
Baysor.append_confidence!(df_spatial; nn_id=25, border_quantiles=(0.2, 0.7));
```

### Initialize BM data

```julia
# SCALE = 10 / 0.1625;
# SCALE = 100;
SCALE = 75;
SCALE_STD = "40";

# @time bm_data = Baysor.initial_distribution_arr(cur_df, cur_centers; min_molecules_per_cell=MIN_MOLECULES_PER_CELL, scale=SCALE,
#     new_component_weight=NEW_COMPONENT_WEIGHT, center_component_weight=CENTER_COMPONENT_WEIGHT, n_frames=1, scale_std="50%")[1];

@time bm_data = Baysor.initial_distribution_arr(cur_df; min_molecules_per_cell=MIN_MOLECULES_PER_CELL, scale=SCALE,
    new_component_weight=NEW_COMPONENT_WEIGHT, n_frames=1, scale_std=SCALE_STD)[1];
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
@time polygons = Baysor.boundary_polygons(bm_data, grid_step=10.0, bandwidth=10.0, dens_threshold=1e-7);
```

```julia
Baysor.plot_cell_borders_polygons(cur_df, polygons; color=cur_gene_colors)
```

## Run on whole dataset


### Initialize BM data

```julia
@time bm_data_arr = Baysor.initial_distribution_arr(df_spatial; scale=SCALE, min_molecules_per_cell=MIN_MOLECULES_PER_CELL,
    new_component_weight=NEW_COMPONENT_WEIGHT, n_frames=20, scale_std=SCALE_STD);
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
Baysor.drop_unused_components!(bm_data_merged; min_n_samples=3, force=true);
```

```julia
@time segmentation_df = Baysor.get_segmentation_df(bm_data_merged, gene_names);
@time cell_stat_df = Baysor.get_cell_stat_df(bm_data_merged, segmentation_df);
```

```julia
t_df = @where(Baysor.get_segmentation_df(bm_data_merged, use_assignment_history=false), :x .> 20000, :x .< 30000, :y .> 25000, :y .< 35000) |> deepcopy;
size(t_df)
```

```julia
@time t_neighb_cm = Baysor.neighborhood_count_matrix(t_df, GENE_COMPOSITION_NEIGHBORHOOD, length(gene_names));
@time t_gene_colors = Baysor.gene_composition_colors(t_neighb_cm, gene_color_transform);
```

```julia
@time polygons = Baysor.boundary_polygons(t_df, t_df.cell, grid_step=10.0, 
    dens_threshold=1e-7, bandwidth=10.0, min_molecules_per_cell=5);
```

```julia
Baysor.plot_cell_borders_polygons(t_df, polygons; color=cur_gene_colors)
```

```julia
out_stat_names = names(cell_stat_df)[.!in.(names(cell_stat_df), Ref([:has_center, :x_prior, :y_prior]))];

CSV.write("../output/iss_human_hca10_2/cm.tsv", Baysor.convert_segmentation_df_to_cm(segmentation_df), delim="\t")
CSV.write("../output/iss_human_hca10_2/cell_stat.csv", cell_stat_df[:, out_stat_names]);
CSV.write("../output/iss_human_hca10_2/segmentation.csv", segmentation_df[:, names(segmentation_df) .!= :mol_id]);
```
