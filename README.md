

# Enabling Knowledge Refinement upon New Concepts in Abductive Learning

This is the repository for holding the sample code of [Enabling Knowledge Refinement upon New Concepts in Abductive Learning](https://www.lamda.nju.edu.cn/publication/aaai23ablnc.pdf)  in AAAI 2023.

This code is only tested in Linux environment.

## Environment Dependency

- Ubuntu 20.04
- Python 3.8
- Cuda 11.3
- PyTorch
- CuPy
- clingo
- tqdm
- imblearn
- pytod
- scikit-learn
- ILASP (https://doc.ilasp.com/installation.html)

To create the above environment with [Anaconda](https://www.anaconda.com/products/distribution), you can run the following command (cudatoolkit=11.3 for new GPUs / new drivers, cudatoolkit=10.1 for old GPUs):

(cudatoolkit=11.3)

```bash
conda create -n ablnc python=3.8 -y
conda activate ablnc
conda install pytorch=1.12 torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install cupy-cuda11x clingo tqdm matplotlib imblearn pytod scikit-learn
Download and install ILASP according to https://doc.ilasp.com/installation.html and copy './ILASP' to current path
```

(cudatoolkit=10.1)

```bash
conda create -n ablnc python=3.8 -y
conda activate ablnc
conda install pytorch=1.7 torchvision torchaudio cudatoolkit=10.1 -c pytorch
pip install cupy-cuda101 clingo tqdm matplotlib imblearn pytod scikit-learn
Download and install ILASP according to https://doc.ilasp.com/installation.html and copy './ILASP' to current path
```

## Running Code

To reproduce the experiment results, we can simply run the following code:

- Less-Than with New Digits

  ```
  python main.py --task=less_than
  ```

- Chess with New Pieces

  ```
  python main.py --task=chess
  ```

- Multiples of Three

  ```
  python main.py --task=multiples_of_three
  ```

To view or change the hyperparameters, please refer to the *arg_init()* function in the code.

## Reference

```
@inproceedings{ablnc2023huang,
  title={Enabling Knowledge Refinement upon New Concepts in Abuctive Learning},
  author={Huang, Yu-Xuan and Dai, Wang-Zhou and Jiang, Yuan and Zhou, Zhi-Hua},
  booktitle={Proceedings of the 37th AAAI Conference on Artificial Intelligence (AAAI'23)},
  //pages={},
  year={2023}
}
```

