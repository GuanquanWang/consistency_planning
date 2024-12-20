from setuptools import setup, find_packages

setup(
  name = 'diffuser',
  packages = find_packages(),
  install_requires=[
        "blobfile>=1.0.5",
        "numpy",
        "torch",
        "torchvision",
        "torchaudio",
        "tqdm",
        "scipy",
        "pandas",
        "piq",
        "joblib==0.14.0",
        "albumentations==0.4.3",
        "lmdb",
        "clip @ git+https://github.com/openai/CLIP.git",
        "mpi4py",
        "pillow",
        "mujoco-py",
        "d4rl @ git+https://github.com/Farama-Foundation/d4rl@f2a05c0d66722499bf8031b094d9af3aea7c372b",
        "matplotlib",
        "typed-argument-parser",
        "gitpython",
        "scikit-video==1.1.11",
        "einops",
    ],
)
