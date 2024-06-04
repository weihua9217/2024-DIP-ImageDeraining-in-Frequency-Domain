# DIP2024 Final Project: Image Deraining with Multi-angle Rain Streak Extraction and Frequency Domain Processing

## Dataset
Place the image follow the format in the 'dataset' folder.

## Requirement

```
pip install -r requirement.txt
```

## Original Method 

```
python generate_ldgp.py --dataset Rain100L

python generate_sdr_rain100l.py --ldgp_data_path './dataset/Rain100L/ldgp' --sdr_result_path './dataset/Rain100L/sdr'
```

Note: 
   - After running generate_ldgp.py, the predict rain streak will be saved in dataset/Rain100L/ldgp. 
   - After running generate_sdr_rain100l.py, the derained result will be save in dataset/Rain100L/sdr.

## Multi-angle Rain Streak Extraction

```
python multiangle_ldgp.py --dataset Rain100L

python generate_sdr_rain100l.py --ldgp_data_path './dataset/Rain100L/ldgp_mul' --sdr_result_path './dataset/Rain100L/sdr_mul'
```

- Note: 
   - After running multiangle_ldgp.py, the predict rain streak will be saved in dataset/Rain100L/ldgp_mul.
   - After running generate_sdr_rain100l.py, the derained result will be saved in dataset/Rain100L/sdr_mul.


## Image deraining in frequency domain

```
python ldgp_freq.py --dataset Rain100L
```
- Note: 
   - After ldgp_freq.py, the derained result will be saved in freq_result.