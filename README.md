# On-particle-based-online-smoothing-and-parameter-inference-in-general-state-space-models

an Implementation of particle-based, rapid incremental smoother (PaRIS) [https://arxiv.org/abs/1412.7550]

the particle-based, rapid incremental smoother (PaRIS), is an algorithm to efficiently perform online approximation of smoothed expectations of additive state functionals in general hidden Markov model. The algorithm has, under weak assumptions, linear computational complexity and very limited memory requirements.

# particle filter

particle filter like Kalman filters, are a great way to track the state of a dynamic system for which you have a Bayesian model.  That means that if you have a model of how the system changes in time, possibly in response to inputs, and a model of what observations you should see in particular states, you can use particle filters to track your belief state.  
