# DLMF
Implemention of DLMF with C/C++.

The main code of DLMF is in language.cpp, and the tow .hpp files(namely language.hpp and common.hpp) provide some tool functions.

Compile using "make".

Before running command which will be described in the following paragraph, you will need to export liblbfgs to your LD_LIBRARY_PATH to run the code (export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/liblbfgs-1.10/lib/.libs/).

Run using ./train and specifying an input file (e.g. the Artz.votes.gz file provided). Input files should be a list of quadruples of the form (userID, itemID, rating, time) followed by the number of words for that review, followed by the words themselves (see Arts.votes.gz). For example, a possible command may like this:

./train test.txt

where test.txt is a file which meets the above requirements. To run this command correctly, you may need pay attention to the relative position of input file to make sure the command finds this file.

For additional datasets see snap.stanford.edu/data

Questions and comments to cathylilin@gmail.com
