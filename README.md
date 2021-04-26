# DeepHEC

This project looks and implementing methods to solve a robotic grasping task in the PyBullet KukaCam Diverse Object Environment.
On-policy methods are implemented and then improved upon using techniques such as Hindsight Experience and Attention Mechanisms that improve performance and sample efficiency in the environment.
PPO was first implemented, which is a strictly on-policy method. IPG, a on-/off-policy hybrid was then implemented, using PPO policy loss to improve performance over standard PPO.
IPG with Hindsight Experience Replay was then implemented to again improve performance over standard IPG.
IPG using an Attention Mechanism was also implemented to improve performance over standard IPG.
Finally, IPG used with a combination of both HER and Attention was used to produce a significant improvement over all previous implementations in the environment.
Results can be seen below.
![results](https://user-images.githubusercontent.com/35261632/116062562-ea7a3d80-a67b-11eb-8bcd-621a30be12f7.png)
