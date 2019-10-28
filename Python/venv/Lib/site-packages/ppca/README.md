ppca
======

The `ppca` packages implements different inference methods for Probabilistic Principal Component Analysis described by Christopher Bishop.

Python implementation followed the way from the book `A First Course in Machine Learning` by *Simon Rogers* and *Mark Girolami* from Chapter 7.5 to 7.7

`ppca.py`: probabilistic PCA for continuous values (Simon's book Chapter 7.5), update tau, X and W when doing EM.

`probit_ppca.py`: probit ppca for binary values (Simon's book Chapter 7.7), apply probit function, update Q, bias, X and W when doing EM.

Also, borrowed some code from:
https://github.com/cangermueller/ppca
