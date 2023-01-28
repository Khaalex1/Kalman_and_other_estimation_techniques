# Kalman filter and other estimation techniques

Different labs for the engineering course of Localization and Estimation (specialization year at ENSTA Bretagne), rewritten on jupyter notebooks.
Exercises as well as some snippets used in the labs are from Dr Jaulin's course paper ["Mobile Robotics"](https://www.ensta-bretagne.fr/jaulin/ensi_isterobV2.pdf).

Lab 2 presents the Least Squares Estimator and shows application on small and elementary exercises.

Lab 3 highlights Monté-Carlo's search as well as the Simulated Annealing (SA) method, based on the same idea. SA is applied on a robot searching with LIDAR detectors for an ideal spot in a room.

Last labs are about the Kalman filter and its applications on elementary exercises. If linear-type relations exist between the states (or positions) and the measures; and a state and his previous one, the Kalman's filter can be a useful tool to estimate confidently a position. This filter is divided in 2 main parts, the Correction which can be applied alone to compute linear estimations, and the Prediction, which is the mode used when measurements are not available (problematic of GPS localization when an auto goes through a tunnel). When these two processes are associated, the filter can achieve great accuracy and confidence in its predictions.

SLAM applies the Kalman filter on the localization of an underwater robot. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/92320638/215238575-7df5ae81-b7f7-4beb-bf8a-218e730a6300.gif"/>
</p>

    Path est
$$Path estimation of an underwater robot$$
