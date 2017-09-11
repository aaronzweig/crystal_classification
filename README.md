## Installation
Ideally do all this in a Conda virtual environment

For Python Nauty Wrapper:
```bash
cd pynauty-0.6.0
tar xvzf nauty26r9.tar.gz
cd nauty26r9
make
cd ..
ln -s nauty26r9 nauty
make user-ins
make pynauty
sudo pip install .
```

For RDKit:
```bash
conda install -c rdkit rdkit
```

Which then requires you install Tensorflow like this:
```bash
pip install --upgrade tensorflow
```

For GCN and required packages, go to the top folder:
```bash
python setup.py install
```

Then to run,
```bash
cd gcn
python train_experimental.py
```
