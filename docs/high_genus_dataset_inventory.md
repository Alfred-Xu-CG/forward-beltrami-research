# Official high-genus dataset inventory

Date: 2026-08-28

## Provenance

- Publication: *Inter-Surface Maps via Constant-Curvature Metrics*.
- Official project page: <https://www.graphics.rwth-aachen.de/publication/03309/>
- Downloaded archive:
  <https://www.graphics.rwth-aachen.de/media/papers/309/s2020-inter-surface-maps-data.zip>
- Local archive size: `73,023,114` bytes.
- SHA-256:
  `4d4e7d4c0ecb33c1be334219de1c6cf46faeec15453d8d2db170911bfbf3c9f2`.
- Local extraction root:
  `external_data/s2020-intersurfacemaps-data` (ignored by Git).

No license file was found inside the archive.  The data are used locally for
research validation and are not redistributed by this repository.

## Primary pairs

| case | archive directory | genus | source V/F | target V/F | common overlay V/F | OBJ quantization repair |
|---|---|---:|---:|---:|---:|---:|
| `official_genus3` | `Fig11_genus/3` | 3 | 941 / 1,890 | 651 / 1,310 | 8,947 / 17,902 | merge 5 vertices; drop 10 zero-area duplicate faces |
| `official_genus5` | `Fig11_genus/5` | 5 | 2,946 / 5,908 | 1,901 / 3,818 | 23,238 / 46,492 | merge 15; drop 30; one `2.5e-7` source nudge |
| `official_pretzel_genus3` | `Fig12_texture/pretzel` | 3 | 1,966 / 3,940 | 4,558 / 9,124 | 29,920 / 59,848 | merge 6; drop 12 |

The paired `M_on_A.obj`/`M_on_B.obj` meshes have identical connectivity and
different embeddings on the source and target.  They encode the published
piecewise-linear common-refinement map.  Six-decimal OBJ export creates a
small number of coincident vertices and exactly zero-area duplicate triangles;
the loader performs only joint, topology-checked quantization repair.  Maximum
projection residuals back to the original meshes are `4.53e-7`, `6.49e-7`, and
`6.64e-7`, respectively.

The archive contains no genus-2 real pair.  The exact-genus-2 primary case is a
generated smoothed implicit double torus with 1,450 vertices and 2,904 faces;
it is clearly labeled generated in every metrics file.
