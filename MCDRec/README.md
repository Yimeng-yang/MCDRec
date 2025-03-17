# Multimodal Conditioned Diffusion Model for Recommendation

## Environment

- cuda 10.2
- python 3.8.10
- pytorch 1.12.0
- numpy 1.21.2

## Usage

### Data

The experimental data are in './data' folder, including Amazon-Baby and Amazon-Sports.

### Training

#### MCDRec_BM3

```
cd ./src
python main.py -m bm3_mcdrec -d baby
```

#### MCDRec_FREEDOM

```
cd ./src
python main.py -m freedom_mcdrec -d baby
```

#### MRD_BM3

```
cd ./src
python main.py -m bm3_mrd -d baby
```

#### DGD_BM3

```
cd ./src
python main.py -m bm3_dgd -d baby
```
