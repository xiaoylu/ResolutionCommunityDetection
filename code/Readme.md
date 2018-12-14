Instructions
===

Note that, this version is built on the top of a customized NetworkX.
You need to install networkx from source code first. 

Please download this repo branch instead of the NetworkX master branch:

[https://github.com/xiaoylu/networkx/tree/issue3153](https://github.com/xiaoylu/networkx/tree/issue3153)

Make sure uninstall networkx if you have installed it.  

Then, at the home folder, type 
```
pip install -e .
```
to install by source.

Now, execute the algorithm on LFR networks
```
python LFR.py
```

Execute the algorithm on American college football networks:
```
python generalized_modularity.py 
```

