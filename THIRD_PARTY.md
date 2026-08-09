# Third-party components

LatSearch is released under the MIT License. This repository also contains or
adapts components from the following projects; their original licenses and
citations continue to apply to those components.

| Component | Use in this repository | Upstream license |
| --- | --- | --- |
| [Wan2.1](https://github.com/Wan-Video/Wan2.1) | Video diffusion backbone in `third_party/WanVideoModel/` | [Apache-2.0](licenses/Wan2.1-APACHE-2.0.txt) |
| [VideoReward / VideoAlign](https://github.com/KwaiVGI/VideoAlign) | Video-level reward model in `third_party/VideoReward/` | [MIT](licenses/VideoAlign-MIT.txt) |
| [FreeInit](https://github.com/TianxingWu/FreeInit) | Evaluation baseline | [MIT](licenses/FreeInit-MIT.txt) |
| [FreqPrior](https://github.com/fudan-zvg/FreqPrior) | Evaluation baseline reproduced from the paper | No license declared upstream at the time of release |
| [EvoSearch](https://github.com/tinnerhrhe/EvoSearch-codes) | Evaluation baseline | [Apache-2.0](licenses/EvoSearch-APACHE-2.0.txt) |

Please cite the corresponding papers when using a baseline. The implementations
under `third_party/WanVideoModel/wan/` were integrated with Wan2.1 for the controlled
comparison reported in the LatSearch paper and may differ from later upstream
versions.
