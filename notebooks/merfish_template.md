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
using Statistics
using StatsBase

import Colors
import CSV
import Images
import Plots
import Random
import NPZ
```

## Read data

```julia
DATA_PATH = "../../CellSegmentation/data/merfish_moffit/";
EXCHANGE_PATH = expanduser("~/mh/tmp_exchange/");

X_COL = :x;
Y_COL = :y;
GENE_COL = :gene;

GENE_COMPOSITION_NEIGHBORHOOD = 20;

MIN_MOLECULES_PER_GENE = 10;
MIN_MOLECULES_PER_CELL = 30;

NEW_COMPONENT_FRAC = 0.3;
NEW_COMPONENT_WEIGHT = 0.2;
CENTER_COMPONENT_WEIGHT = 5.0;
```

```julia
# @time df_spatial, gene_names = Baysor.load_df("$DATA_PATH/merfish_coords_perprocessed.csv"; 
@time df_spatial, gene_names = Baysor.load_df("$DATA_PATH/merfish_coords_adj.csv"; 
    x_col=X_COL, y_col=Y_COL, gene_col=GENE_COL, min_molecules_per_gene=MIN_MOLECULES_PER_GENE);
df_spatial[!, :paper_cell], paper_cell_names = Baysor.encode_genes(df_spatial[!, :cell]);
df_spatial[!, :paper_cell] .-= 1;
df_spatial[!, :mol_id] .= 1:size(df_spatial, 1);

paper_cell_centers = Baysor.load_centers("$DATA_PATH/merfish_centers_adj.csv");
size(df_spatial)
```

```julia
@time neighb_cm = Baysor.neighborhood_count_matrix(df_spatial, GENE_COMPOSITION_NEIGHBORHOOD);
@time gene_color_transform = Baysor.gene_composition_transformation(neighb_cm);
```

```julia
@time dapi = Images.load("$DATA_PATH/stains/dapi_merged.tiff");
dapi_arr = Float64.(dapi);

@time seg_mask = (dapi_arr .> Images.otsu_threshold(dapi_arr));
Plots.heatmap(seg_mask[2000:9000, 1:7000], format=:png)
```

Append confidence based on DAPI

```julia
Baysor.append_confidence!(df_spatial, seg_mask, nn_id = 15);
```

## Run on a single frame to find parameters


### Select single frame

```julia
cur_df = @where(deepcopy(df_spatial), :x_raw .> -3500, :x_raw .< -3000, :y_raw .> -2500, :y_raw .< -2000);
cur_centers = Baysor.subset_by_coords(paper_cell_centers, cur_df);

@show size(cur_df)

@time cur_neighb_cm = Baysor.neighborhood_count_matrix(cur_df, GENE_COMPOSITION_NEIGHBORHOOD, length(gene_names));
@time cur_gene_colors = Baysor.gene_composition_colors(cur_neighb_cm, gene_color_transform);

@time paper_polygons = Baysor.boundary_polygons(cur_df, cur_df.paper_cell, grid_step=10.0, dens_threshold=1e-7);
```

```julia
@time paper_polygons = Baysor.boundary_polygons(cur_df, cur_df.paper_cell, grid_step=10.0, dens_threshold=1e-7);
```

### Annotate and visualize data

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
Plots.plot(
    Baysor.plot_cell_borders_polygons(cur_df, paper_polygons, cur_centers.centers; annotation=annot_df.l2),
    Baysor.plot_cell_borders_polygons(cur_df, paper_polygons, cur_centers.centers; color=cur_gene_colors),
    layout=2, size=(750*2, 750)
)
```

### Initialize BM data

```julia
@time bm_data = Baysor.initial_distribution_arr(cur_df, cur_centers; min_molecules_per_cell=MIN_MOLECULES_PER_CELL,
    new_component_weight=NEW_COMPONENT_WEIGHT, n_frames=1, scale_std="50%")[1];
```

Adjust edge weights using DAPI:

```julia
Plots.heatmap(dapi[2000:5000, 1:3000], format=:png, size=(300, 300))
```

```julia
Baysor.append_dapi_brightness!(bm_data.x, dapi_arr);
@time Baysor.adjust_field_weights_by_dapi!(bm_data, dapi_arr);
```

### Run

```julia
@time Baysor.bmm!(bm_data; n_iters=500, min_molecules_per_cell=MIN_MOLECULES_PER_CELL, 
    split_period=100, n_expression_clusters=10, min_cluster_size=10, n_clustering_pcs=30, clustering_distance=Distances.CosineDist(),
    verbose=true, new_component_frac=NEW_COMPONENT_FRAC, log_step=50);
```

```julia
Baysor.plot_num_of_cells_per_iterarion(bm_data.tracer)
```

```julia
@time polygons = Baysor.boundary_polygons(bm_data, grid_step=10.0, dens_threshold=1e-7);
```

```julia
Plots.plot(
    Baysor.plot_cell_borders_polygons(cur_df, polygons, cur_centers.centers; annotation=annot_df.l2),
    Baysor.plot_cell_borders_polygons(cur_df, polygons, cur_centers.centers; color=cur_gene_colors),
    layout=2, size=(750*2, 750)
)
```

<!-- #region heading_collapsed=true -->
### Mixing scoring
<!-- #endregion -->

```julia hidden=true
clust_per_mol = annot_df.l1;

c_assignment = bm_data.assignment
c_pap_assignment = bm_data.x.paper_cell
c_rand_assignment = Random.shuffle(c_pap_assignment)

cluster_mixing_score = Baysor.score_doublets_by_local_clusters(c_assignment, clust_per_mol, na_value=0.0)
cluster_mixing_score_paper = Baysor.score_doublets_by_local_clusters(c_pap_assignment, clust_per_mol, na_value=0.0)
cluster_mixing_score_rand = Baysor.score_doublets_by_local_clusters(c_rand_assignment, clust_per_mol, na_value=0.0)

c_n_mols_per_cell = Baysor.count_array(c_assignment .+ 1)[2:end]
c_n_mols_per_cell_paper = Baysor.count_array(c_pap_assignment .+ 1)[2:end]
c_n_mols_per_cell_rand = Baysor.count_array(c_rand_assignment .+ 1)[2:end]

p0, p1 = Baysor.plot_mixing_scores(cluster_mixing_score, c_n_mols_per_cell, cluster_mixing_score_paper, c_n_mols_per_cell_paper, 
    cluster_mixing_score_rand, c_n_mols_per_cell_rand, format=:png, min_mols_per_cell=50);

display(p0)
display(p1)
```

```julia hidden=true
t_xvals = 0:800
t_scores = [[mean(ms[(n .>= i) .& (n .< (i + 50))] .* n[(n .>= i) .& (n .< (i + 50))]) for i in t_xvals] for (ms, n) in 
        zip([cluster_mixing_score, cluster_mixing_score_paper, cluster_mixing_score_rand],
            [c_n_mols_per_cell, c_n_mols_per_cell_paper, c_n_mols_per_cell_rand])]

p = Plots.plot(xlabel="Size window", ylabel="Mean number of 'wrong' molecules")

for (n,sc) in zip(["Baysor", "Paper", "Random"], t_scores)
    p = Plots.plot!(t_xvals, sc, label=n)
end
p
```

```julia hidden=true
t_xvals = 0:800
t_scores = [[mean(ms[(n .>= i) .& (n .< (i + 50))]) for i in t_xvals] for (ms, n) in 
        zip([cluster_mixing_score, cluster_mixing_score_paper, cluster_mixing_score_rand],
            [c_n_mols_per_cell, c_n_mols_per_cell_paper, c_n_mols_per_cell_rand])]

p = Plots.plot(xlabel="Size window", ylabel="Mixing score")

for (n,sc) in zip(["Baysor", "Paper", "Random"], t_scores)
    p = Plots.plot!(t_xvals, sc, label=n)
end
p
```

```julia hidden=true
mixing_fractions = Baysor.score_doublets(bm_data; n_expression_clusters=10, n_pcs=30, min_cluster_size=10, 
    min_molecules_per_cell=MIN_MOLECULES_PER_CELL, distance=Distances.CosineDist(), na_value=0.0)[2];

doublet_mixing_colors = Baysor.map_to_colors(get.(Ref(mixing_fractions), bm_data.assignment, 0.0))
Baysor.plot_colorbar(doublet_mixing_colors)
```

```julia hidden=true
cluster_mixing_score = Baysor.score_doublets_by_local_clusters(c_assignment, annot_df.l1, na_value=0.0);
cluster_mixing_colors = Baysor.map_to_colors(get.(Ref(cluster_mixing_score), bm_data.assignment, 0.0))
Baysor.plot_colorbar(cluster_mixing_colors)
```

```julia hidden=true
# Baysor.plot_cell_borders_polygons(cur_df, polygons, cur_centers.centers; color=doublet_mixing_colors[:colors])
```

```julia hidden=true
Plots.plot(
    Baysor.plot_cell_borders_polygons(cur_df, polygons, cur_centers.centers; color=doublet_mixing_colors[:colors], title="Doublet mixing scores"),
    Baysor.plot_cell_borders_polygons(cur_df, polygons, cur_centers.centers; color=cluster_mixing_colors[:colors], title="Cluster mixing scores"),
    layout=2, size=(750 * 2, 750)
)
```

```julia hidden=true
Plots.plot(
    Baysor.plot_cell_borders_polygons(cur_df, polygons, cur_centers.centers; annotation=annot_df.l2),
    Baysor.plot_cell_borders_polygons(cur_df, polygons, cur_centers.centers; color=cur_gene_colors),
    layout=2, size=(750*2, 750)
)
```

<!-- #region heading_collapsed=true -->
### Comparison with paper
<!-- #endregion -->

```julia hidden=true
baysor_assignment = bm_data.assignment;
paper_assignment = cur_df.paper_cell;
```

```julia hidden=true
cluster_mixing_scores_baysor = Baysor.score_doublets_by_local_clusters(baysor_assignment, annot_df.l1, na_value=0.0);
cluster_mixing_scores_paper = Baysor.score_doublets_by_local_clusters(paper_assignment, annot_df.l1, na_value=0.0);

n_mols_per_cell_baysor = Baysor.count_array(baysor_assignment .+ 1)[2:end];
n_mols_per_cell_paper = Baysor.count_array(paper_assignment .+ 1)[2:end];
```

```julia hidden=true
Plots.histogram(cluster_mixing_scores_baysor[n_mols_per_cell_baysor .>= MIN_MOLECULES_PER_CELL], bins=100, normalize=true, label="Baysor")
Plots.histogram!(cluster_mixing_scores_paper[n_mols_per_cell_paper .>= MIN_MOLECULES_PER_CELL], bins=100, alpha=0.7, normalize=true, label="Paper")
```

```julia hidden=true
Plots.histogram(n_mols_per_cell_baysor[n_mols_per_cell_baysor .>= MIN_MOLECULES_PER_CELL], bins=50, label="Baysor")
Plots.histogram!(n_mols_per_cell_paper[n_mols_per_cell_paper .>= MIN_MOLECULES_PER_CELL], bins=50, alpha=0.7, label="Paper")
```

```julia hidden=true
Baysor.assignment_summary_df(:Baysor => baysor_assignment, :Paper => paper_assignment; min_molecules_per_cell=MIN_MOLECULES_PER_CELL)
```

## Run on whole dataset


### Initialize BM data

```julia
@time bm_data_arr = Baysor.initial_distribution_arr(df_spatial, paper_cell_centers; min_molecules_per_cell=MIN_MOLECULES_PER_CELL,
    new_component_weight=NEW_COMPONENT_WEIGHT, n_frames=20, scale_std="50%");
```

### Parallel run

```julia
@time bm_data_arr_res = Baysor.run_bmm_parallel(bm_data_arr, 500; min_molecules_per_cell=MIN_MOLECULES_PER_CELL, 
    split_period=100, n_expression_clusters=10, min_cluster_size=10, n_clustering_pcs=30, clustering_distance=Distances.CosineDist(),
    new_component_frac=NEW_COMPONENT_FRAC)
```

```julia

```
