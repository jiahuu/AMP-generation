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