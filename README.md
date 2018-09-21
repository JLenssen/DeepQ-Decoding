
###  _DeepQ Decoding_

This repository provides all the tools necessary to obtain and run surface code decoders for fault tolerant quantum computation, i.e. decoders capable of dealing with faulty syndrome measurements, via <a href="https://www.nature.com/articles/nature14236">DeepQ reinforcement learning</a>.

We begin with an introduction to the setting, strategy and techniques, before providing detailed working examples for:

<ul>
  <li>Training and evaulating new decoders from scratch.</li>
  <li>Loading and running pre-trained decoders.</li>
</ul> 

Finally, we provide some of the results obtained from the trained models that have been provided here.

Enjoy!

### 1) Setting

Topological quantum error correcting codes, and in particular the surface code, currently provide the most promising path to <a href="https://www.nature.com/articles/nature14236"> scalable fault tolerant quantum computation</a>. While a variety of decoders exist for such codes, such as <a href="https://arxiv.org/abs/1709.02154"> minimum weight perfect matching</a>, recently decoders obtained via  <a href="https://arxiv.org/abs/1802.06441"> machine learning techniques</a> have attracted attention due to both their potential flexibility, with respect to codes and noise models, and their potentially fast run times. However, up until this point it has not been demonstrated how one could use machine learning techniques to obtain decoders for the setting of fault tolerant quantum computation, in which both the physical qubits and syndrome extraction is assumed to be noisy. Here, we demonstrate how reinforcement learning techniques, and in particular deepQ learning, can be utilized to solve this problem and obtain such decoders.

#### 1a) The Surface Code

While the techniques presented here could be applied to any stabilizer code (work-in-progress) we focus on the surface code, as shown below: 

<p align="center">
<img src="https://user-images.githubusercontent.com/6330346/45884807-21499200-bdb5-11e8-97f7-60c9682c299e.png" width="75%" height="75%">
</p>

<ul>
  <li>(a) We consider a d by d lattice, with qubits on the vertices (referred to as physical qubits), and plaquette stabilizers.</li>
  <li>(b) Orange (blue) plaquettes indicate stabilizers which check the Z (X) parity of qubits on the vertices of the plaquette.</li>
  <li>(c) Using red circles to indicate violated stabilizers (anyons) we see here some basic examples of the anyons created from X, Y or Z Pauli flips on a given vertex qubit.</li>
   <li> (d) The X logical operator of the surface code we consider is given by any continuous string of X errors which connect the top and bottom boundaries of the code.
   <li> (e) Similarly, The Z logical operator is given by any continuous string of Z errors which connect the left and right boundaries of the code.
</ul> 

In order to get intuition for the decoding problem, which we will present in detail further down, it is useful to see some examples of the syndromes (anyon configurations) generated by various error configurations...

<p align="center">
<img src="https://user-images.githubusercontent.com/6330346/45884806-21499200-bdb5-11e8-9770-f7eaf37159fb.png" width="40%" height="40%">
</p>

In particular, it is very important to note that the map from syndromes to error configurations is _not_ one-to-one! For example, one can see that the error configurations given in the top-left and bottom-left codes both lead to the same syndrome. This ambiguity in the error configuration leading to a given syndrome gives rise to the decoding problem, which we describe below.

#### 1b) The Decoding Problem

Given the above introduction to the surface code it is now possible to understand the decoding problem, within the fault tolerant setting. Quite loosely, given a code in the eigenstate of one of the logical operators (or, more generally, given any state in the ground state space of the code), the key aim of decoding is keep the code in this given state by exploiting _faulty_ syndrome information to determine which corrections need to be applied to the code to compensate for continuous noise and errors.

<p align="center">
<img src="https://user-images.githubusercontent.com/6330346/45884801-20b0fb80-bdb5-11e8-957d-e84e99fee7d6.png" width="80%" height="80%">
</p>

To be more specific, lets consider the above illustration:

<ol>
  <li>In the top left, we start with a code in which each physical qubit is prepared in the |0\rangle state, as a result of which the logical qubit (given by the whole code) is in the eigenstate of the logical Z operator. Our goal is to maintain the logical qubit in this state. </li>
  <li>Now, while storing the logical qubit (between gates for instance) the physical qubits are subject to noise. We consider depolarizing noise here for simplicity, for which in each unit of time each physical qubit is subject to either a Pauli X, Y or Z flip with a given probability (the physical error rate). In the above illustration, we imagine an X flip occuring on the physical qubit in the third row and second column.  </li>
  <li>In order to maintain the code in the state it was given to us, we therefore need to perform a correction by applying an X gate to the qubit which was randomly flipped. To do this, we need perform a syndrome extraction, from which our decoding algorithm can attempt to diagnose the error configuration which gave rise to the received syndrome. However, as illustrated in the diagram, the syndrome extraction process is also noisy, and for each stabilizer there is a probability (the measurement error rate) that the measured stabilizer value is incorrect - i.e. that we see an anyone where there is not one, or no anyon where there actually is one.</li>
   <li> To deal with this situation, instead of providing a single syndrome to the decoder, we perform multiple (faulty) syndrome measurements, between which physical errors may also occur. We then provide as input to our decoder not a single syndrome, but a stacked volume of succesive syndrome slices.
   <li> From this syndrome volume the decoding algorithm needs to suggest corrections which when applied to the code lattice move the logical qubit back into the original state (in practice, these corrections are not actually implemented, but rather tracked through the computation, and applied in a single step at the end).
   <li> In the ideal case the decoder will be able to correctly diagnose a sufficient proportion of syndrome volumes, such that the probability of an error occuring on the logical qubit is lower than the physical error rate on a physical qubit.
</ol> 

### 1c) DeepQ Learning as a Tool for Obtaining Decoders

Given the problem as specified above, we utilize <a href="https://www.nature.com/articles/nature14236">DeepQ</a> <a href="http://incompleteideas.net/book/bookdraft2017nov5.pdf">reinforcement learning</a>, a technique which has been sucessfully used to obtain agents capable of super-human performance in domains such as  <a href="https://www.nature.com/articles/nature14236">Atari</a>, to obtain decoders which are capable of dealing with faulty measurements up to a threshold physical and measurement error rate. We will not go too deeply into the details and theory of Q-learning here, as an excellent introduction can be found in the fantastic textbook of <a href="http://incompleteideas.net/book/bookdraft2017nov5.pdf">Sutton and Barto</a>, which is strongly recommended.

However, to give a brief overview, the rough idea is that we will utilize a deep neural network (a convolutional neural network in our case) to parameterize the Q-function of a decoding agent, which interacts with the code lattice (the environment). This Q-function is a function which maps from states of the environment - syndrome volumes plus histories of previously applied corrections - to a Q-value for each available correction, where the Q-value of a given action, with respect to a particular environment state, encodes the expected long term benefit (not the exact technical definition!) to the agent of applying that correction when in that state. Given the Q-values corresponding to a given environment state, the optimal correction strategy then corresponds to applying the correction with the largest Q-value. Within this framework, the goal is then to obtain the optimal Q-function, which is done by letting the agent interact with the environment, during which the agents experiences are used to iteratively update the Q-function.

In order to present our approach it is therefore necessary to discuss:

<ul>
  <li> The manner in which we encode the environment state.</li>
  <li> The parameterization of our Q-function via a deep neural network.</li>
  <li> The procedure via which the agent interacts with the environment to gain experience, from which the Q-function can be updated.</li>
</ul> 

Let's begin with the manner in which the environment state is encoded. In particular, at any given time we consider the environment state to consist of:

<ol>
  <li> A representation of the most recently measured faulty syndrome volume.</li>
  <li> A representation of the actions which have been taken since receiving the most recent syndrome volume.</li>
</ol> 

Given a d by d surface code lattice, we encode a single syndrome slice in a (2d+1) by (2d + 1) binary matrix, as illustrated below:

<p align="center">
<img src="https://user-images.githubusercontent.com/6330346/45884802-20b0fb80-bdb5-11e8-906e-b758177a7c63.png" width="60%" height="60%">
</p>

Similarly, we encode the history of either X or Z Pauli corrections applied since the last syndrome volume was received in a (2d+1) by (2d + 1) binary matrix of the following form:

<p align="center">
<img src="https://user-images.githubusercontent.com/6330346/45884804-21499200-bdb5-11e8-86f6-f56a46c59567.png" width="60%" height="60%">
</p>

Finally, given these conventions for syndrome and action history slices we can construct the complete environment state by stacking syndrome slices on top of an action history slice for each allowed Pauli operator (in pratice we only need to allow for X and Z corrections). This gives us a total environment state in this form:

<p align="center">
<img src="https://user-images.githubusercontent.com/6330346/45884799-20b0fb80-bdb5-11e8-9313-0ffa63ae7382.png" width="80%" height="80%">
</p>

In the above image we have shown just three syndrome slices for simplicity, but as we will see later the depth of the syndrome volume (the number of slices) can be chosen at will.

Now that we know how the state of the environment is encoded at any given time step we can proceed to examine the way in which we choose to parameterize the Q-function of our agent via a deep convolutional neural network. For an introduction to such networks, see <a href="http://colah.github.io/posts/2014-07-Conv-Nets-Modular/">here</a> or <a href="https://github.com/R-Sweke/CrashCourseInNeuralNetworksWithKeras">here</a>.

<p align="center">
<img src="https://user-images.githubusercontent.com/6330346/45884803-20b0fb80-bdb5-11e8-8d20-c6c2fd337649.png" width="80%" height="80%">
</p>

As illustrated above, our deepQ network is given by a simple convolutional neural network, consisting of:

<ol>
  <li> A user-specified number of convolutional layers (a-b).</li>
  <li> A user specified number of feed-forward layers (c).</li>
  <li> A final layer providing Q-values for each available correction (d), with respect to the input state.</li>
</ol>  

Given these ingredients we can now examine in detail the training procedure, through which an optimal Q-function is updated via iterative updates from experience generated by interaction with the environment. As per the majority of reinforcement learning techniques, and illustrated below, this procedure involves a sequence of alternating steps in which:

<p align="center">
<img src="https://user-images.githubusercontent.com/6330346/45884797-20186500-bdb5-11e8-8c25-0dad87b46ce7.png" width="30%" height="30%">

<ol>
  <li> The environment provides a state to the agent.</li>
  <li> The agent uses its current strategy to choose an action, with which it acts on the environment.</li>
  <li> The environment updates its internal state appropriately, and responds to the agent by providing a new state along with a numerical reward and a binary signal which illustrates whether the agent is "dead" or "alive".</li>
    <li> If the agent hasn't "died", it can then use this reward signal to update its internal strategy before once again acting on the environment and starting another round of interaction. If it has died, a new episode is started. </li>  
</ol> 

From the agent's perspective the goal is to converge to a strategy which allows it to maximise the expected value of its (discounted) cumulative reward. In our particular context of the decoding problem, an episode works as illustrated below:

<p align="center">
<img src="./Images/https://user-images.githubusercontent.com/6330346/45884798-20b0fb80-bdb5-11e8-9ef3-23b55b52a498.png" width="80%" height="80%">
</p>

In particular:

<ol>
  <li> As illustrated and described in Section 1b. (the "Decoding Problem"), an episode starts with the extraction of a (faulty) syndrome volume from the code (a, b). If the syndrome volume is trivial, i.e. there is not a single anyon in the entire volume, then another syndrome volume is extracted.</li>
  <li> As a new syndrome volume has just been extracted, the action history is reset to all zeros (c).</li>
  <li> The just extracted syndrome volume is combined with the reset action history, as previously described in the "state construction" figure, and then provided to the agent as the initial state (d).</li>
    <li> Now the agent must choose an action (e). As per most RL algorithms it is helpful to balance a period of exploration, with a period of exploiting previously obtained knowledge. As such, with a given probability \epsilon, which is annealed during the course of training, the agent will choose an action at random, and with a probability 1-\epsilon the agent will choose the action with the maximal Q-value according to its current parameterization. In order to aid training, we restrict the agents random choice to actions which are either adjacent to anyons, or adjacent to previously acted on qubits.</li>
    <li> When the agents acts on the environment with the chosen action, provided the action is not the identity action, multiple things then happen simultaneously. Firstly, the action history slices of the visible state are updated to indicate the action that has been applied (f). Then, the action is actually applied to the code lattice, whose error configuration is updated accordingly (g). Then finally, in order to determine the reward, a "referee" decoder takes in the true non-faulty syndrome corresponding to the updated error configuration (h). If the referee decoder can succesfully decode the current syndrome, then the agent remains alive and the episode continues, if not then the agent dies and the episode ends. If the agent remains alive and its action has resulted in putting the code back into the desired initial state, the agent is giving a reward of 1, in any other case the agent is given a reward of 0.</li>
    <li> The reward and game over signal is then combined with the updated state (in which only the action history was updated) and provided to the agent (i,j). In addition, the tuple of (state, action, reward, new state, game_over) is added to an external memory which is used to update the parameterization of the agent via backpropogation. </li>
        <li> The procedure detailed above is then repeated (k-p) until the point at which the agent chooses to do the identity (q), which can be done explicitly, or by repeating an action. Conceptually, the identity action is meant as a tool for the agent to signal its belief that it has applied all the corrections necessary to return the code to the desired initial state. </li>    
    <li> Given the identity signal from the agent, the environment then provides a new faulty syndrome volume (s,t), the action history slices of the state are reset, the new visible state is constructed from the rest action history and the updated syndrome (u,t) and fed to the agent, from which the episode continues as per steps (4-7), until the agent dies. </li> 
</ol> 

What has not been specifically illustrated in the above diagram is the procedure via which the parameterization of the Q-function is updated from batches of experience tuples. We will not present the details here as this is done using the exact same Q learning methodoloy described in  <a href="https://www.nature.com/articles/nature14236">these</a>  <a href="https://arxiv.org/abs/1511.06581">two</a> landmark papers on deepQ learning.

At this point all that remains is to discuss how decoding is done in practice once training has been completed and the agent has converged to an optimal Q-function. As illustrated below, this is quite straightforward:

<p align="center">
<img src="https://user-images.githubusercontent.com/6330346/45884796-20186500-bdb5-11e8-96e1-36ef03160bcf.png" width="80%" height="80%">
</p>

Specifically, decoding proceeds as follows:

<ol>
  <li> Firstly, a syndrome volume is extracted from the code lattice and encododed as previously discussed (a,b). This encoded syndrome volume is then stacked with a blank action history to create the initial input state to the decoder (c, d).</li>
  <li> Given this input state, one forward pass of the neural network is executed and an argmax is taken over the output Q-values to obtain the first suggested correction. This suggested correction is then added to a memory (f) and used to update the action history slices of the visible state (e). These updated action history slices are then combined with the original syndrome volume (g) and passed to the decoder (h)</li>
  <li> Step 2 is then repeated (i,j,k,l) until the point at which the agent chooses the identity action (m).</li>
    <li> At this point, given that the agent has signalled that it belives it has supplied all the necessary corrections, the acummulated corrections are applied to the code lattice (n), or in practice, tracked through the computation. </li>
</ol> 


## 2) Training Decoders in Practice

Now that we have discussed the conceptual foundations, strategies and techniques involved, lets walk through a detailed example of how to train a decoder.

### 2a) Requirements

The following packages are required, and can be installed via PIP:

<ol>
  <li> Python 3</li>
  <li> tensorflow </li>
  <li> keras </li> 
  <li> gym </li> 
</ol> 

In addition, a modified version of the Keras-RL package is required, which should be installed from <a href="https://github.com/R-Sweke/keras-rl">this fork</a>.
