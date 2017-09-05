## Installation
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

For GCN and required packages, go to the top folder:

```bash
python setup.py install
```
