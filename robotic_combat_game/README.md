# CED

## Code Environment
- **Python Version**: 3.10  
- **CUDA Version**: 12.2
- **OS Version**: Ubuntu 22.04

### Main Installed Libraries

| Library         | Version  |
|-----------------|----------|
| numpy           | 1.24.0   |
| networkx        | 3.3      |
| ray             | 2.43.0   |
| scikit-image    | 0.25.2   |
| scikit-learn    | 1.6.1    |
| tensorboard     | 2.19.0   |
| torch (PyTorch) | 1.13.1   |
| matplotlib      | 3.10.1   |

## How to run
### Generate buffer data

```
python generate_buffer.py
```

### Run CED code
```
python driver_ced.py
```

### Run OSP code
```
python driver_osp.py
```

### Run BC code
```
python driver_bc.py
```

### Test your model
```
python test_driver.py
```