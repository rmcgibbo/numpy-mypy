# numpy-mypy

This is a very preliminary work-in-progress attempt to introduce mypy type annotations for numpy.

Clone and `cd` into this repo, and try something like

```
$ cat simple-example.py
import numpy as np
a = np.zeros((2,2,2))
reveal_type(a)

$ mypy simple-example.py
simple-example.py:3: error: Revealed type is 'numpy.ndarray[builtins.float, numpy.ThreeD]'
```
