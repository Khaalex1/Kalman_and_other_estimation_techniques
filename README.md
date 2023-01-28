# Kalman filter and other estimation techniques

Different labs for the engineering course of Localization and Estimation (specialization year at ENSTA Bretagne), rewritten on jupyter notebooks.
Exercises as well as some snippets used in the labs are from Dr Jaulin's course paper ["Mobile Robotics"](https://www.ensta-bretagne.fr/jaulin/ensi_isterobV2.pdf).

Lab 2 presents the Least Squares Estimator and shows application on small and elementary exercises.

Lab 3 highlights Monté-Carlo's search as well as the Simulated Annealing (SA) method, based on the same idea. SA is applied on a robot searching with LIDAR detectors for an ideal spot in a room.

Last labs are about the Kalman filter and its applications on elementary exercises. If linear-type relations exist between the states (or positions) and the measures; and a state and his previous one, in the form of 
$$X_{k+1} = A.X_{k} + u_{k} + \alpha_{k} $$
$$Y_{k} = C.X_{k} + \beta_{k}$$ 
  , the Kalman filter can estimate confidently a position. This filter is the association of 2 main parts run iteratively: the __Correction__ which can be applied alone to compute linear estimations, and the __Prediction__, which is the mode used when measurements are not available (e.g. problematic of GPS localization through a tunnel).

SLAM notebook applies the Kalman filter on the localization of an underwater robot. The Kalman filter estimates the robot and landmarks' position as well as its error variance, which can allow to plot more intuitive representations, such as :

<p align="center">
  <img src="https://user-images.githubusercontent.com/92320638/215238575-7df5ae81-b7f7-4beb-bf8a-218e730a6300.gif"/>
</p>

$$Confidence \ \ ellipses \ \ of \\ estimated \\ positions \\ (blue \\ : \\ robot, \\ red \\ : \\ landmarks) $$

*Confidence ellipses at degree $\eta$ (here $\eta$ = 0.9) indicates that an estimated position is inside with proba $\eta$. The smaller they are, the smaller is the filter's margin of error (it is more accurate).
In this case, we see that the filter also exploits information from the environment, as the confidence spreading of the filter decreases when a landmark is detected.*
