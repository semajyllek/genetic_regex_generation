## REGEX Generation

This project was an attempt to reimplement the algorithm used by Bartoli, et. al:

```
C Bartoli, Davanzo, De Lorenzo, Mauri, Medvet, Sorio, Automatic Generation of Regular Expressions from Examples with Genetic Programming
```

Their implementation is in Java and uses a clever way of distributing the genetic search to find the best candidate for each generation. 

This was an attempt to replicate the genetic algorithm for automatic generation of regular expressions from text in python, using the DEAP library. 
This required essentialy the writing of a sublanguage with objects representing the operations that can be done to restrict the sequences that can be generated.

But despite this and many other seeming improvements, the search spacehas so far proved too large for the randomization that occurs in the genetic algorithm's learning process to 
produce sensible expressions to evcen evaluate for precision or accuracy.

Possible improvements to match the paper performance, based on a thorough reproduction of the paper in Java that was done from their repo, that DID result in good performance, if after a long time,
are:

1. use a distributed process to slit the population and evaluate for best candidates of generation
2. constrain the possible sequences more with handcrafted rules and possible based on prior probabilities based on distributions of subsequences in text
