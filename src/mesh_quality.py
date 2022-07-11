import pyvista
import json
import sys
import os
import statistics


criteria = [
    'area',
    'aspect_beta',
    'aspect_frobenius',
    'aspect_gamma',
    'aspect_ratio',
    'collapse_ratio',
    'condition',
    'diagonal',
    'dimension',
    'distortion',
    'jacobian',
    'max_angle',
    'max_aspect_frobenius',
    'max_edge_ratio',
    'med_aspect_frobenius',
    'min_angle',
    'oddy',
    'radius_ratio',
    'relative_size_squared',
    'scaled_jacobian',
    'shape',
    'shape_and_size',
    'shear',
    'shear_and_size',
    'skew',
    'stretch',
    'taper',
    'volume',
    'warpage',
]

meshes_dir = "./Data/"
meshes = next(os.walk(meshes_dir), (None, None, []))[2]  # [] if no file



print(f"meshes {meshes}")
for criterion in criteria:
  all_meshes_quality_stats = []
  
  for mesh_file in meshes:
    print(f"mesh_file {mesh_file}")
    meshe_quality_stats = {}
    pvmesh = pyvista.read(os.path.join(meshes_dir,mesh_file) )
    cqual = pvmesh.compute_cell_quality(criterion)
    scores = cqual.cell_data["CellQuality"]
    meshe_quality_stats["name"] = mesh_file.split('/')[-1]
    meshe_quality_stats["average"] = statistics.mean(scores)
    meshe_quality_stats["variance"] = statistics.variance(scores)
    quantiles = statistics.quantiles(scores, n=10)
    meshe_quality_stats["first_quantile"] = quantiles[0]
    meshe_quality_stats["last_quantile"] = quantiles[-1]

    all_meshes_quality_stats.append(meshe_quality_stats)

  with open(f"output/quality_criteria/{criterion}.json", "w") as outfile:
    json.dump(all_meshes_quality_stats, outfile, indent=4)
