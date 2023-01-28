# Kalman filter and other estimation techniques

Different labs for the engineering course of Localization and Estimation (specialization year at ENSTA Bretagne), rewritten on jupyter notebooks.
Exercises as well as some snippets used in the labs are from Dr Jaulin's course paper ["Mobile Robotics"](https://www.ensta-bretagne.fr/jaulin/ensi_isterobV2.pdf).

Lab 2 presents the Least Squares Estimator and shows application on small and elementary exercises.

Lab 3 highlights Monté-Carlo's search as well as the Simulated Annealing (SA) method, based on the same idea. SA is applied on a robot searching with LIDAR detectors for an ideal spot in a room.

Last labs are about the Kalman filter and its applications on elementary exercises. If linear-type relations exist between the states (or positions) and the measures; and a state and his previous one, in the form of 
$$   \begin{equation}
  X_{k+1} = A.X_{k} + u_{k} + \alpha_{k} \\
  Y_{k} = C.X_{k} + \beta_{k} \\ 
  \end{equation}$$
  , the Kalman's filter can be a useful tool to estimate confidently a position. This filter is the association of 2 main parts run iteratively, the Correction which can be applied alone to compute linear estimations, and the Prediction, which is the mode used when measurements are not available (problematic of GPS localization when an auto goes through a tunnel).

SLAM applies the Kalman filter on the localization of an underwater robot. The Kalman filter estimates the robot and landmarks' position as well as its confidence level, which can allow to plot more intuitive representations, such as :

<p align="center">
  <img src="https://user-images.githubusercontent.com/92320638/215238575-7df5ae81-b7f7-4beb-bf8a-218e730a6300.gif"/>
</p>

$$Confidence \ \ levels \ \ in \ \  position \ \  estimations \ \  of \ \  the \ \  robot \ \  (blue) \ \  and \ \ landmarks \ \  (red) $$

**Confidence ellipses at degree $\eta$ indicates that an estimated position is inside with proba $\eta$. The smaller they are, the smaller is the filter's margin of error (it is more accurate).
In this case, we see that the filter also exploits information from the environment, as the confidence of the filter increases when a landmark is detected.**
