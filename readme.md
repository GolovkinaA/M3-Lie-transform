# M3 Project R-PNN 

**R-PNN**: **R**ecurrent **P**olynomial **N**eural **N**etwork for dynamic system simulation and learning

## Introduction
R-PNN is a software written in Python and developed under the M3 project №75206008 at Saint Petersburg State University.

The project aims to develop a set of libraries for constructing and learning recurrent polynomial neural networks (R-PNN).

By design, R-PNN has a strong connection with ordinary differential equations within the mechanism of differential algebra.
Thanks to it, it provides interpretable predictions even when trained on small datasets.

The project consists of three logical parts:

* ODE identify:
extracting governing differential equations in polynomial form from time-series data

* R-PNN construct:
converting known/identified system of ODEs to initial weights for R-PNN of a certain order

* R-PNN learn:
learning R-PNN of a certain order from time-series data

The project is in the middle of its realization and not all the modules are in their final version by now. For the moment, the repository contains the source code for the identification and construction parts with the restriction on system dimension (the realization supports only 2 independent variables in the time-series data).


## Usage

### Running R-PNN
The source code for the identification part is placed in the ode_lie.py file while the source code for the R-PNN construction part is placed in the 
sym_func.py file.

These blocks work separately, each file contains its own ''__ main __'' section.


### Examples
There are two examples implemented in the code.
The time-series for the identification block is generated artificially by numerical solving the ODEs describing electromagnetic deflector's dynamics and Van-der-Pol oscillator's dynamics.

In the R-PNN construction block, the same equations are used for the initial weights generation for the R-PNN. The user can choose the order of R-PNN to be constructed.

## Future Developments
* Improvement in the code and user interface to utilize the software
* Realization of R-PNN with shared weights learning from the additional bunch of data
* Possibility to transfer information from one block to another
* Increase in order and dimension of the system
* Adding documentation to the functions in the code
* Implementation of R-PNN construct block for time-dependent and time-delay data
* Realization of R-PNN learning from the additional bunch of data

## License
The Apache 2.0 License is used for this software.