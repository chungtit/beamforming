# DC programming and DCA for Secure Guarantee with Null Space Beamforming in Two-Way Relay Networks
Coded by [Thuy-Chung Luong](https://github.com/ChungLuongThuy), Nguyen The Duy at FPT university 
# Introduction
Physical layer security technique takes advantage the physical characteristic of wireless channel to guarantee secrecy for message being transmitted. In [A. Wyner, 1975](https://wiki.epfl.ch/edicpublic/documents/Candidacy%20exam/Wiretap_Wyner.pdf), Wyner developed a new direction for creating secure links without relying on the privacy cryptograph. The notation ”secrecy capacity” was defined as the maximum rate received at the legitimate receiver, which still kept the eavesdropper completely ignorant of the transmitted messages. Therefore,
this method would help reduce the probability of interception.

We consider a two-way relay network  system which contains multiple cooperative relays transmitting information between two sources with attendance of an eavesdropper. A null space beamforming scheme is applied to ensure the secrecy in system. Our **goal** is achieving maximal secrecy sum rate with a certain level power transmit. In [Y. Yang, 2014](https://ieeexplore.ieee.org/abstract/document/6730702), authors propose an approach to solve Secrecy Sum Rate Maximization (SSRM) problem, whereas we use another approach base on Different of Convex Functions Algorithm (DCA).

**Keywords:** Two-way relay network, transmitted power, secrecy sum rate (SSR), [Difference of Convex Algorithm](http://www.lita.univ-lorraine.fr/~lethi/index.php/en/dca.html).
# Simulation result
![10tests](https://user-images.githubusercontent.com/36873488/60321627-92068600-99a7-11e9-960e-48e6b4fc32ba.png)

We compare performances of DC algorithm and Optimization of Beamforming Vector (OBV) algorithm. We consider the secrecy sum rate achieved by approaches with set of power Pt = [30 : 1 : 40] (dBW), the number of relays N ∈ {4, 6, 8}, P1 = P2 = 𝐏𝒕/𝟒(dBW),𝝈𝟏𝟐 = 𝝈𝟐𝟐 = 𝟏𝐝𝐁 . Each pair of Pt and N we use 10 sets of randomized data, then getting the average.

