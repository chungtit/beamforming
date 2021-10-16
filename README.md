## DCA optimization algorithm to ensure information security in the network
Physical layer security technique takes advantage of the physical characteristic of the wireless channels to guarantee secrecy for messages being transmitted. In [A. Wyner, 1975](https://wiki.epfl.ch/edicpublic/documents/Candidacy%20exam/Wiretap_Wyner.pdf), Wyner developed a new direction for creating secure links without relying on the privacy cryptograph.

Applying the Wyner's idea, maximizing the secrecy rate with a certain level of power transmission and the appearance of an eavesdropper is a non-convex optimization problem. With a non-convex problem, we can use the [Difference of Convex functions Algorithms (DCA)](http://www.lita.univ-lorraine.fr/~lethi/index.php/dca.html) to solve it. 

This code implements 2 algorithms:

First,`dca_algorithm.py` was implemented by following the DCA scheme in paper [DC programming and DCA for Secure Guarantee with Null Space Beamforming in Two-Way Relay Networks](https://dl.acm.org/doi/10.1145/3316615.3316687). DCA constitutes the backbone of smooth/nonsmooth nonconvex programming and global optimization. Thus, it is not fast, but it is more accurate than another algorithm in `ssrm_algorithm.py`.

Second, `ssrm_algorithm.py`was implemented by following the Secrecy Sum Rate Maximization (SSRM) problem in paper [Algorithms for Secrecy Guarantee With Null Space Beamforming in Two-Way Relay Networks](https://ieeexplore.ieee.org/abstract/document/6730702).

There are many intermediate variables in these two papers. Some intermediate variables are kept names exactly the same as papers in this code. 

**Keywords:** Two-way relay network, transmitted power, secrecy sum rate (SSR), [Difference of Convex Algorithm](http://www.lita.univ-lorraine.fr/~lethi/index.php/en/dca.html).
