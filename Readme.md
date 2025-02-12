# Emb-FT-Code

## Description
This project implements task-specific embedding adaptation using Constrastive Learning. It allows for a label-aware dimensionality reduction in binary classification tasks which helps increase performance in the downstream task.
This is the implementation of the method discussed in the paper [Efficient Domain Adaptation of Multimodal Embeddings using Constrastive Learning](https://arxiv.org/abs/2502.02048).

## File Structure

```
- `mmimdb/`: Directory containing preprocessed data and generated embeddings.
- `main_mmimdb.py`: Main script to load the dataset, apply projections, and run classification experiments using various machine learning models. You should start here for a full experiment with actual embeddings with the MM-IMDB dataset.
- `contrastive_projector.py`: Contains the implementation of the `ContrastiveProjector` class, which uses contrastive learning to project embeddings into a lower-dimensional space in a label-aware manner. The file also contains sample code on how to use the class independently in an sklearn-compatible way.
- `mmimdb_emb_generate.py`: Script to generate image and text embeddings from the MM-IMDb dataset using pre-trained models like ViT and BERT.
```

## Getting Started

### Prerequisites
- Python 3.x
- Pytorch


### Usage
1. Run on MM-IMDB:
    ```bash
    python main_mmimdb.py
    ```
2. Run on dummy scikit-learn data (as a simple projection):
    ```bash
    python contrastive_projector.py
    ```


## Citation
If you use this repository, please cite the following repository:

```
@misc{margaritis2025efficientdomainadaptationmultimodal,
      title={Efficient Domain Adaptation of Multimodal Embeddings using Constrastive Learning}, 
      author={Georgios Margaritis and Periklis Petridis and Dimitris J. Bertsimas},
      year={2025},
      eprint={2502.02048},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.02048}, 
}
```
