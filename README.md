# AMP-design
AI-mediated strategy for Development of Antimicrobial Peptide for  Anti-infection
<div align=center><img src=workflow.png></div>

## Preperation


## AMP generation

### Training
```
python soft-prompt-tune.py
```

### Generation
```
python generation.py
```

## AMP discrimination

### Training
```
python MCL-AMP.py
```

### Inference
```
python MCL-AMP-prediction.py
```

### Citation

If this work or code is helpful to your research, please kindly cite our paper:

```bibtex
@article{liu2026deep,
  title={Deep learning-driven integrated pipeline for de novo design and synthesis of antimicrobial peptides},
  author={Liu, Jiahui and Chen, Y. and Tang, J. and others},
  journal={npj Drug Discovery},
  volume={3},
  number={1},
  pages={15},
  year={2026},
  publisher={Nature Publishing Group},
  doi={10.1038/s44386-026-00045-6},
  url={[https://www.nature.com/articles/s44386-026-00045-6](https://www.nature.com/articles/s44386-026-00045-6)}
}