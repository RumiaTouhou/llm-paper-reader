# MARLUI: Multi-Agent Reinforcement Learning for Adaptive UIs

Fig. 1. We formulate online user interface adaptation as a multi-agent reinforcement learning problem. Our approach comprises a user- and interface agent. The user agent interacts with an application in order to reach a goal and the interface agent learns to assist it. In the depicted example the user agent interacts with a Virtual Reality toolbar, while the interface agent assigns relevant items for the user agent. The interface agent does not know the goal of the user agent. Crucially, our approach does not rely on labeled offline data or application-specific handcrafted heuristics.

Adaptive user interfaces (UIs) automatically change an interface to better support users' tasks. Recently, machine learning techniques have enabled the transition to more powerful and complex adaptive UIs. However, a core challenge for adaptive user interfaces is the reliance on high-quality user data that has to be collected offline for each task. We formulate UI adaptation as a multi-agent reinforcement learning problem to overcome this challenge. In our formulation, a user agent mimics a real user and learns to interact with a UI. Simultaneously, an interface agent learns UI adaptations to maximize the user agent's performance. The interface agent learns the task structure from the user agent's behavior and, based on that, can support the user agent in completing its task. Our method produces adaptation policies that are learned in simulation only and, therefore, does not need real user data. Our experiments show that learned policies generalize to real users and achieve on-par performance with data-driven supervised learning baselines.

CCS Concepts: Human-centered computing $\rightarrow$ Graphical user interfaces; User models; HCI theory, concepts and models.

Additional Key Words and Phrases: Multi-Agent Reinforcement Learning, Adaptive User Interfaces, Intelligent User Interfaces

## 1 INTRODUCTION
Many tasks require powerful User Interfaces (UIs). However, the more powerful the interface, the more complex it becomes. For instance, consider the plethora of factors (e.g., six degrees of freedom and real-world context) that influence the experience of Mixed Reality interfaces. One solution to the complexity problem is to dynamically adapt the UI to the user and their task by showing relevant information in a contextual and timely manner. Researchers demonstrated that adaptive UIs (AUIs) improve the usability over standard UIs in various use cases, including menus [3, 17, 25, 84], cooperative interfaces [75], and virtual reality interfaces [58]. Appropriate adaptations could help users complete their work, but designing useful adaptive UIs is challenging.

Modern AUIs [31, 88, 89, 96] predominantly leverage machine learning (ML) techniques that learn correlations between user input, user intentions, and adaptation. These approaches decrease development efforts and make AUIs more usable compared to their rule-based predecessors [32]. However, these methods rely on real user data, introducing three significant limitations. First, existing data is usually unavailable for the UIs of emerging technologies (data availability). Second, designers need to recollect data for any change or iteration in the UI design, which is an expensive and time-consuming process (design-specific data). Third, when collecting user data, it is non-trivial to ensure that the data is representative of the users' actual intentions (data quality).

We frame adapting UIs as a multi-agent reinforcement learning (MARL) problem to alleviate these challenges. In our formulation, a simulated user agent learns to interact with a UI to complete a task. Simultaneously, we train an interface agent that learns to adapt the same UI to help the user agent achieve the task more efficiently. By working in a simulated environment, our approach does not rely on tediously collected user data (data availability) since the user agent learns to use the interface through online interactions and thus generates unambiguous data (data quality). Finally, our method implements a general formulation of the interface adaptation problem (i.e., the method is general agnostic) and can learn meaningful policies for different interfaces and tasks (design-specific data).

Specifically, we model the user agent as a Hierarchical Reinforcement Learning (HRL) agent that learns to navigate an interface and complete its task by interacting with the UI. To achieve realistic behavior, we decompose highlevel decision-making (e.g., the decision to select a menu item) from motor control (e.g., moving the cursor to the corresponding menu slot). The interface agent is a single reinforcement learning (RL) policy. Its goal is to assist the user agent in completing the task more efficiently, for instance, by assigning the correct items to a toolbar with limited slots in a game character creation task (cf. Fig. 1). Crucially, the interface agent does not require access to the user agent's intent. Instead, the interface agent learns the underlying task structure. In our setting, both the user agent and the interface agent share the same reward for completing the task efficiently and accurately.

To demonstrate the feasibility of our method, we introduce four proof-of-concept use cases in VR: a game character design tool with an intelligent toolbar, an intelligent numeric keypad, a block tower building game, and an application that helps to select virtual objects that are out of reach.

Our main goal is to show that our method is competitive with baselines that require carefully collected real user data. Not relying on data would make our method suitable for a large variety of interfaces and tasks. To evaluate our goal, we perform both an evaluation with real users and in silico studies in the character creation task. Our study with real users compares our method against different data-driven baselines. We find that training the interface agent with our simulated user agent transfers well to humans and performs on par with previous data-driven methods regarding task completion time.

In summary, this paper makes four key contributions: (1) a MARL-based framework to adapt user interfaces online without relying on real user data; (2) a Hierarchical Reinforcement Learning-based, cognitively inspired user agent that can learn to operate a user interface and enables an interface agent to learn adaptations that are useful to real end-users; (3) a goal-agnostic interface agent that learns the underlying structure of the task purely by observing the action in the interface by the user agent; and (4) empirical results showing the effectiveness of our approach and four different usage scenarios to demonstrate its general applicability.

## 2 RELATED WORK

This paper proposes multi-agent RL as a framework for adaptive UIs. Our method features a user agent that models human interaction behavior and an interface agent that adapts the UI to support the user agent. Most related to our work is research on computational user modeling, methods for adaptive UIs, and (MA)RL.

### 2.1 Computational User Modelling

Computational user modeling has a long tradition in HCI . These models predict user performance and are essential for UI optimization [74]. Early work relies on heuristics [1, 10-12, 48] and on simple mathematical models [27, 38]. More recent work extends these models and, for instance, predicts the operating time for a linear menu [18], gaze patterns [82], or cognitive load [22].

Recently, reinforcement learning gained popularity within the research area of computational user models. This popularity is due to its neurological plausibility [7,29], allowing it to serve as a model of human cognitive functioning. The underlying assumption of RL in HCI is that users behave rationally within their bounded resources [34, 73]. There is evidence that humans use such strategy across domains, such as in causal reasoning [21] or perception [35]. In human-computer interaction, researchers have leveraged RL to automate the sequence of user actions in a KLM framework [55] or to predict fatigue in volumetric movements [13]. It was also used to explain search behavior in user interfaces [98] or menus [15] and as a model for multitasking [46]. Most similar to our work is research on hierarchical reinforcement learning for user modeling. Jokinen et al. [45] show that human-like typing can emerge with the help of Fitts' Law and a gaze model. Other works show that HRL can elicit human-like behavior in task interleaving [33] or touch interactions [45]. Inspired by this work, we design our user agent by decomposing high-level decision-making from motor control. Using two hierarchical levels yields simulated behavior.

### 2.2 Methods for Adaptive Uls

UI adaptation can either be offline, to computationally design an interface, or online, to adapt the UI according to users' goals. We will focus on online adaptive UIs and refer readers to [71, 72] for an overview of computational UI design.

### 2.2.1 Heuristics, Bayesian Networks \& Combinatorial Optimization. 

In early works, heuristic- or knowledge-based approaches are used to adapt the UI [9, 90, 92]. Similarly, multi-agent systems employ rule-based and message-passing approaches [79, 80, 99]. Another popular technique for AUIs is domain-expert-designed Bayesian networks [6, 39]. More recently, combinatorial optimization was used to adapt interfaces dynamically [58, 75]. The downside of these approaches is that experts need to specify user goals using complex rule-based systems or mathematical formulations. Creating them comprehensively and accurately requires developers to foresee all possible user states, which is tedious and requires expert knowledge. Commonly, these approaches also get into conflict when multiple rules or objectives apply. This conflict often results in unintuitive adaptations. In contrast, our method only requires the layout of the UI. From its representation as an RL environment, we learn policies that meaningfully adapt the UI and realistically reproduce user behavior.

### 2.2.2 Supervised Learning. 

Leveraging machine learning can overcome the limitations of heuristic-, network-, and optimization-based systems by learning appropriate UI adaptions from user data. Traditional machine learning approaches commonly learn a mapping from user input to UI adaptation. Algorithms like nearest neighbor [53, 64], Naïve Bayes [23, 65], perceptron [88, 89], support vector machines [5], or random forests [66, 77] are used and models are learned offline [5] and online [88]. Due to the problem setting, these approaches require users' input to be highly predictive of the most appropriate adaptation. Furthermore, it restricts the methods to work in use cases where myopic planning is sufficient, i.e., a single UI adaptation leads users to their goal. In contrast, our method considers multiple goals when selecting an adaptation and can lead users to their goal using sequences of adaptations.

More recent work overcomes the limitations stemming from simple input-to-adaptation mapping by following a two-step approach. They (1) infer users' intention based on observations and (2) choose an appropriate adaptation based on the inferred intent [74]. Such work uses neural networks, and user intention is modeled either explicitly [50, 91] or as a low-dimensional latent representation [81]. However, these approaches are still highly dependent on training data, which may not even be available for emerging technologies. In contrast, our method can learn supportive policies without pre-collected user data by just observing simulated user behavior.

### 2.2.3 Bandits \& Bayesian Optimization. 

Bandit systems are a probabilistic approach often used in recommender systems [36]. In a multi-armed bandit setting, each adaptation is modeled as an arm with a probability distribution describing the expected reward. The Bayes theorem updates the expectation, given a new observation and prior data. Related work leverages this approach for AUIs [47, 49, 60]. Bayesian optimization is a sample-efficient global optimization method that finds optimal solutions in multi-dimensional spaces by probing a black box function [86]. In the case of AUIs, it is used to find optimal UI adaptations by sampling users' preferences [51, 52]. Both approaches trade off exploration and exploitation when searching for appropriate adaptations (i.e., exploration finds entirely new solutions, and exploitation improves existing solutions), rendering them suitable approaches to the AUI problem.

However, such methods are not able to plan adaptations over a sequence of interaction steps, i.e., they plan myopic strategies. In addition, these approaches need to sample user feedback to learn or optimize for meaningful adaptations and, hence, also rely on high-fidelity user data. Furthermore, as users themselves learn during training or optimization, solutions can converge to sub-optimal user behavior as such methods reduce exploration with convergence. In contrast, our method can plan adaptations over a sequence of interaction steps learned from realistic, simulated user data.

### 2.2.4 Reinforcement Learning. 

Reinforcement learning is a natural approach to solving the AUI problem, as its underlying decision-making formalism implicitly captures the closed-loop iterative nature of HCI [41]. It is a generalization of bandits and learns policies for longer horizons, where current actions can influence future states. This generalization enables selecting UI adaptations according to user goals that require multiple interaction steps. Its capability makes RL a powerful approach for AUIs with applications in dialog systems [30, 93], crowdsourcing [19, 42], sequential recommendations [14, 57, 59], information filtering [85], personalized web page design [24], and mixed reality [31]. Similar to our work is a model-based RL method that optimizes menu adaptations [96].

Current RL methods sample predictive models [30, 42, 96] or logged user traces [31]. However, these predictive models and offline traces represent user interactions with non-adaptive interfaces. Introducing an adaptive interface will change user behavior; so-called co-adaptation [63]. Hence, it is unclear if the learned model can choose meaningful adaptations when user behavior changes significantly due to the model's introduction. In contrast, our user agent learns to interact with the adapted UI; hence, our interface agent learns on behavioral traces from the adapted setting.

### 2.2.5 Multi-Agent Reinforcement Learning. 

MARL is a generalization of RL in which multiple agents act, competitively or cooperatively, in a shared environment [102]. Multi-agent systems are common in games [4, 44], robotics [43, 70], or modeling of social dilemmas [54, 101]. MARL is challenging since multiple agents change their behavior as training progresses, making the learning problem non-stationary. Common techniques to address this issue is via implicit [95] or explicit [28] communication, centralized critic functions [62, 100], or curricula [61, 97]. We take inspiration from the latter and use a curriculum during the training of our agents.

Closest to our work is [20], which proposes a multi-agent system that maps 2D interface trajectories to actions for navigating 3D virtual environments. A user agent learns interactions on a 2D interface. A decoder that is trained on real user data maps the user agent's actions to 2D touch gestures. A second agent then translates these 2D touch gestures into 3D operations. In their setting, the interface agent does not observe the environment itself but receives the actions of the user agent as its state. Our work extends their setting to the case where the user agent and the interface agent observe and manipulate the same UI. Furthermore, we do not rely on real world user data.

## 3 MULTI-AGENT REINFORCEMENT LEARNING FOR ADAPTIVE UIS

Adaptive UIs aim to automatically and intelligently adapt a UI to guide the user toward completing their task. Thus, they usually require logged user data or online user feedback. This section explains how our proposed method, MARLUI, can create AUIs without needing data and provides an intuition on its learning procedure.

The MARLUI framework consists of three main components: a user agent, an interface agent, and a user interface as a shared environment (see Fig. 2). A simulated user agent learns to achieve a goal in a simulated UI environment while the interface agent, which is the AUI component, learns to adapt the interface to support the user better. We build on the idea that by providing a model of user and UI that are realistic enough given its real-world counterparts, the interface agent can provide meaningful support to real users in the real task. This becomes clearer if we consider MARLUI's learning procedure: Before starting with training, we specify the goal the user agent should achieve by interacting with the UI. The goal is randomly sampled and unknown to the interface agent. During training, the user agent tries to reach this state through trial and error. If the user agent behaves similarly enough to a human, then the interface agent gets meaningful observations to learn from. In the case of the toolbar in Figure 1, the user agent clicks through random sequences of menu items and then observes if the result matches the goal state. The user agent will explore paths in the UI and adapt its behavioral policy based on the degree to which the path leads to success. To learn user agent policies that exhibit human-like behavior, we consider task-relevant factors of human functioning in its model (e.g., human motor control).

At the same time and in the same environment (i.e., the same UI), we train an interface agent. While the goal of the user agent is to reach a goal state, the interface agent aims to change the UI such that the user agent can achieve its goal more efficiently. In the toolbar example, it would randomly assign items to menu slots and observe if this helps the user agent to make its desired changes on the game character. In an exploration-exploitation manner, the interface agent and user agent jointly explore the state space of the problem. From past observations of the user agent interactions with the UI, the interface agent will learn which goal the user agent attempts to achieve and how to guide it towards completing it. This avoids the need for the interface agent to have direct access to the goal state, i.e., it makes it goal-agnostic.

In the following, we formally introduce reinforcement learning and describe the details of our method. We then show the extent to which our approach generalizes to real users.

Fig. 2. Our interface agent and user agent act in the same environment. The user agent is modeled as a two-level hierarchy with a high-level decision-making policy $\pi_d$ and a low-level motor control policy $\pi_m$. The agent interacts with the UI. The high-level agent observes that state of the environment (Eq. 1) and chooses a specific menu slot as target accordingly (Eq. 2). The lower level receives this action and computes a movement (Sec. 5.2.2). The interface agent policy $\pi_I$ adapts the interface to assist the user agent in achieving its task more efficiently. It observes user actions in the UI (Eq. 6) and decides on adaptations. Note that the interface agent cannot access the goal, making the problem partially observable.

## 4 BACKGROUND

We briefly introduce (multi-agent) reinforcement learning and its underlying decision processes. Specifically, we assume that users behave according to Computational Rationality (CR) [73]. This allows us to frame user behavior as a Partially Observable Markov Decision Process (POMDP).

### 4.1 Partially Observable Markov Decision Processes

Partially Observable Markov Decision Processes (POMDP) is a mathematical framework for single-agent decisionmaking in stochastic partially observable environments [2], which is a generalization over Markov Decision Processes [40]. A POMDP is a seven-tuple ( $s, O, a, T, R, \gamma$ ), where $s$ is a set of states, $O$ is set of observations and $a$ a set of actions. In POMDPs, the exact states $(s \in s)$ of the evolving environment may or may not be captured fully. Therefore, observations $(o \in O)$ represent the observable states, which may differ from the exact state. $T: s \times a \times s \rightarrow[0,1]$ is a transition probability function, where $T\left(s^{\prime}, \mathrm{a}, \mathrm{s}\right)$ is the probability of the transition from state $s^{\prime}$ to $s$ after taking action a. Similarly, $F: s \times a \times O \rightarrow[0,1]$ is an observation probability function, where $F\left(\mathbf{o}, \mathrm{a}, \mathbf{s}^{\prime}\right)$ is the probability of observing $\mathbf{o}$ while transitioning to $s^{\prime}$ after taking action a. $R: s \times a \rightarrow \mathbb{R}$ is the reward function, discounted with factor $\gamma$.

### 4.2 Reinforcement Learning

Reinforcement Learning is a machine learning paradigm that rewards desired and penalizes undesired behavior. Generally, an agent observes an environment state and tries to take optimal actions to maximize a numerical reward signal. A key difference with supervised learning is that RL learns through exploration and exploitation rather than from an annotated dataset.

We follow the standard formulation of RL as an MDP [94], but use observations rather than states since the problem we are working on is partially observable. In our setting the observation space is a subspace of the state space (the interface agent does not have access to the internal state of the user agent), and observations are deterministic: $F\left(\mathbf{o}, \mathrm{a}, \mathbf{s}^{\prime}\right)=1$. The goal is to find an optimal policy $\pi: O \rightarrow a$, a mapping from observations to actions that maximizes the expected return: $\mathbb{E}\left[\sum_{t=0}^T \gamma^t R_i\left(\mathbf{o}_t, \mathbf{a}_t\right)\right]$. Since both observation- and action spaces can be high-dimensional, neural networks are used for policy learning (i.e., we approximate the policy as $\pi_\theta$, where $\theta$ are the learned parameters).

### 4.3 Multi-Agent Reinforcement Learning

Standard reinforcement learning formulations built upon MDPs or POMDPs assume a single policy. Stochastic games generalize MDPs for multiple policies [87]. When players do not have perfect information about the environment, stochastic games become partially observable stochastic games. A partially observable stochastic game is defined as an eight-tuple ( $N, \mathcal{S}, O, \mathcal{A}, T, \mathcal{F}, \mathcal{R}, \gamma)$, where $N$ is the number of policies. $\mathcal{S}=s_1 \times \ldots \times s_N$ is a finite set of state sets, and $O=O_1 \times \ldots \times O_N$ is a finite set of observation sets, with subscripts indicating different policies. $\mathcal{A}=a_1 \times \ldots \times a_N$ is a finite set of action sets. $T$ is the transition probability function. $\mathcal{F}=F_1 \times \ldots \times F_N$ defines a set of observation probability functions of different players. A set of reward functions is defined as $\mathcal{R}=R_1, \ldots R_N$. Furthermore, we define a set of policies as $\Pi=\pi_1, \ldots \pi_N$. Finally, $\gamma$ is the discount factor.

All policies have their individual actions, states, observations, and rewards. In this paper, we optimize each policy individually, while the observations are influenced by each other's actions. We use model-free RL (for comparison to model-based RL, see [78]). This set of algorithms is used in an environment where the underlying dynamics $T\left(\mathbf{s}^{\prime}, \mathbf{a}, \mathbf{s}\right)$ and $F\left(\mathbf{o}, \mathrm{a}, \mathbf{s}^{\prime}\right)$ are unknown, but it can be explored via trial-and-error. In the method section, we use the terms state and observation interchangeably.

## 5 METHOD

We present a general task description and outline the model of our user agent and interface agent (Fig. 2).

### 5.1 General Task Description

We model tasks to be completed if the user achieves their desired goal. For game character creation, a goal can be the desired configuration of a character with a certain shirt (red, green, blue) or backpack (pink, red, blue). We represent the goal as a one-hot vector encoding $g$ of these attributes. A one-hot vector can be denoted as $\mathbb{Z}_2^j$, where $j$ is the number of items. For the previous example, $\mathbf{g}$ would be in $\mathbb{Z}_2^6$ as it possesses six distinct items.

Furthermore, the user agent can access an input observation denoted by $\mathbf{x}$. For example, this can correspond to the current character configuration. The current input observation, $\mathbf{x}$, and the goal state $\mathbf{g}$ are identical in dimension and type.

The user agent interacts with the interface and attempts to make the input observation and goal state identical as fast as possible, such that $\mathbf{x}=\mathbf{g}$. Each interaction updates $\mathbf{x}$ accordingly, and a trial terminates once they are identical. In the character creation example, this would be the case if the shirt and backpack of the edited character are identical to the desired configuration. The interface agent makes online adaptations to the interface. It does not know the specific goal of a user. Instead, it needs to observe user interactions with the interface to learn the underlying task structure that will yield the optimal adaptations, e.g., the user likely wants to configure the backpack after configuring the shirt.

### 5.2 User Agent

First, we introduce the user agent, which interacts with an environment to achieve a certain goal (e.g., select the intended attributes of a character). The agent tries to accomplish this as fast and accurately as possible. Thus, the user agent first has to compare the goal state and input observation and then plan movements to reach the target. We model the user as a hierarchical agent with separate policies. Specifically, we introduce a two-level hierarchy: a high-level decision-making policy $\pi_d$ that computes a target for the agent (high-level decision-making), and a Fitts'-Law-based low-level motor policy $\pi_m$ that decides on a strategy to reach this target. We approximate visual cost with the help of existing literature. We now explain both policies in more detail.

5.2.1 High-level Decision-Making Policy. The high-level decision-making policy of the hierarchy is responsible to select the next target item in the interface. The overall goal of the policy is to complete a given task while being as fast as possible. Its actions are based on the current observation of the interface, the goal state, and the agent's current state. More specifically, the high-level state space $s_d$ is defined as:

Formula 1

$$
s_d=(\mathbf{p}, \mathbf{m}, \mathbf{x}, \mathbf{g}),
$$

which comprises: i) the current position of the user agent's end-effector normalized by the size of the UI, $\mathbf{p} \in I^n$ (where $n$ denotes the dimensions, e.g., 2 D vs 3 D ), ii) an encoding of the assignment of each item $\mathbf{m} \in \mathbb{Z}_2^{n_i \times n_s}$, with $n_i$ and $n_s$ being the number of menu items and environment locations, respectively, iii) the current input state $\mathbf{x} \in \mathbb{Z}_2^{n_i}$, and iv) the goal state $\mathbf{g} \in \mathbb{Z}_2^{n_i}$. Here, $I$ denotes the unit interval $[0,1]$, and $\mathbb{Z}_2^n$ is the previously described set of integers. The item-location encoding $m$ represents the current state of a UI. It can be used, for instance, to model item-to-slot assignments. The action space $a_D$ is defined as:

Formula 2

$$
a_d=\mathbf{t},
$$

which indicates the next target slot $\mathbf{t} \in \mathbb{N}_{n_s}$. The reward for the high-level decision-making policy consists of two weighted terms to trade-off between task completion accuracy and task completion time: i) how different the current input observation $\mathbf{x}$ is from the goal state $\mathbf{g}$, and ii) the time it takes to execute an action. Therefore, the high-level policy needs to learn how items correlate with the task goal as well as how to interact with any given interface. With this, we define the reward as follows:

Formula 3

$$
R_d=\alpha \underbrace{\mathcal{E}_{g d}}_{i)}-(1-\alpha) \underbrace{\left(T_D+T_M\right)}_{i i)}+\mathbb{1}_{\text {success }},
$$

where $\mathcal{E}_{g d}$ is the difference between the input observation and the goal state, $\alpha$ a weight term, $T_M$ the movement time as an output of the low-level policy, $T_D$ the decision time, and $\mathbb{1}_{\text {success }}$ an indicator function that is 1 if the task has been successfully completed and 0 otherwise.

In addition to movement time, we also need to determine the decision time $T_D$. To this end, we are inspired by the SDP model [18]. This model interpolates between an approximated linear visual search-time component $\left(T_s\right)$ and the Hick-Hyman decision time [38] ( $T_{h h}$ ), both are functions that take into account the number of menu items and user parameters. We refer to [18] for more details.

We define the difference $\mathcal{E}_{g d}$ between the input observation $\mathbf{x}$ and the goal state $\mathbf{g}$ as the number of mismatched attributes:

Formula 4

$$
\mathcal{E}_{g d}=-\sum_{x \in \mathbf{g}, y \in \mathbf{x}} \frac{\mathbb{1}_{x \neq y}}{n_{\text {attr }}},
$$

where $\mathbb{1}$ is an indicator function that is 1 if $x \neq y$ and else $0, x$ and $y$ are individual entries in the vectors $\mathbf{g}$ and $\mathbf{x}$ respectively, and $n_{\text {attr }}$ is the number of attributes (e.g., shirt, backpack, and glasses).

5.2.2 Low-Level Motor Control Policy. The low-level motor control policy is a non-learned controller for the end-effector movement. In particular, given a target, it selects the parameters of an endpoint distribution (mean $\mu_{\mathrm{p}}$ and standard deviation $\sigma_{\mathrm{p}}$ ). We set $\mu_{\mathrm{p}}$ to the center of the target. The target t is the action of the higher-level decision-making policy $\left(a_D\right)$. Following empirical results [27], we set $\sigma_{\mathrm{p}}$ to $1 / 6$ th of a menu slot width to reach a hitrate of $96 \%$.

Given the current position and the endpoint parameters (mean and standard deviation), we compute the predicted movement time using the WHo Model [37].

Formula 5

$$
T_M=\left(\frac{k}{\left(\sigma_{\mathbf{p}} / d_{\mathbf{p}}-y_0\right)^{1-\beta}}\right)^{1 / \beta}+T_M^{(0)}
$$

where $k$ and $\beta$ are parameters that describe a group of users, $T_M^{(0)}$ is the minimal movement time, and $y_0$ is equal to the minimum standard deviation. The term $d_{\mathrm{p}}$ indicates the traveled distance from the current position to the new target position $\mu_p$. We follow literature for the values of other parameters [37, 45]. We sample a new position from a normal distribution: $\mathbf{p} \sim \mathcal{N}\left(\mu_{\mathbf{p}}, \sigma_{\mathbf{p}}\right)$.

### 5.3 Interface Agent

The interface agent makes discrete changes to the UI to maximize the performance of the user agent. For instance, it assigns items to a toolbar to simplify their selection for the user agent. Unlike the user agent, we model the interface agent as a flat RL policy. The state space $s_I$ of the interface agent is defined as:

Formula 6

$$
s_I=(\mathbf{p}, \mathbf{x}, \mathbf{m}, \mathbf{o}),
$$

which includes: i) the position of the user $\mathbf{p} \in I^2$, ii) the input observation $\mathbf{x} \in \mathbb{Z}_2^{n_i}$, iii) the current state of the UI $\mathbf{m} \in \mathbb{Z}_2^{n_i \times n_s}$, and iv) a vector including the history of interface elements the user agent interacted with (commonly referred to as stacking). The action space $a_I \in \mathbb{Z}$ and its dimensionality is application-specific. The goal of the interface agent is to support the user agent. Therefore, the reward of the interface agent is directly coupled to the performance of the user agent. We define the reward of the interface agent to be equal to the reward of the user agent's high-level policy:

Formula 7

$$
R_I=R_D
$$

Note that the interface agent does not have access to the user agent's goal $\mathbf{g}$ or target $\mathbf{t}$. To accomplish its task, the interface agent needs to learn to help the user agent based on an implicit understanding of i) the objective of the user agent, and ii) the underlying task structure. Our setting allows the interface agent to gain this understanding solely by observing the changes in the interface as the result of the user agent's actions. This makes the problem more challenging but also more realistic.

Fig. 3. In our proposed task, the user (agent) matches a game character selection to a target state (1). The user operates a toolbar with three slots (2). The interface agent assigns the most relevant items to the available slots (3). This cycle continues till the two characters match (4).

## 6 IMPLEMENTATION

We train the user and interface agents' policies simultaneously. All policies receive an independent reward, and the actions of the policies influence a shared environment. We execute actions in the following order: (1) the interface agent's action, (2) the user agent's high-level action, followed by (3) the user agent's low-level motor action. The reward for the two learned policies is computed after the low-level motor action has been executed. The episode is terminated when the user agent has either completed the task or exceeded a time limit.

We implement our method in Python 3.8 using RLLIB [56] and Gym [8]. We use PPO [83] to train our policies. We use 3 cores on an Intel(R) Xeon(R) CPU @ 2.60GHz during training. Training takes $\sim 36$ hours. We utilize an NVIDIA TITAN Xp GPU for training. The user agent's high-level decision-making policy $\pi_d$ is a 3-layer MLP with 512 neurons per layer and ReLU activation functions. The interface agent's policy $\pi_I$ is a two-layer network with 256 neurons per layer and ReLU activation functions. We sample the full state initialization (including goal) from a uniform distribution. We use stochastic sampling for our exploration-exploitation trade-off. We use curriculum learning to increase the task difficulty and improve learnability; for more information, see Appendix A. The difference between agents of different applications is their respective state- and action spaces and the set of goals.

## 7 EVALUATION

MARLUI aims to learn UI adaptations from simulated users that can support real users in the same task. -Specifically, we want our method to produce AUIs that are competitive with baselines that require carefully collected real user data. In this section, we evaluate if our approach achieves this goal. Thus, we first conduct an in silico study to analyze how the interface agent and the user agent solve the UI adaptation problem in simulation. Then, we perform a user study to investigate if policies of the interface agent that were learned in simulation generalize to real users.

Fig. 4. We train our agent till convergence. Left: the fraction of successfully completed episodes per epoch. Ours and Static reach a $100 \%$ successful completion rate. Random does not converge. Right: The number of actions needed on average during a successful episode. Our method needs less actions compared to Static and Random.

### 7.1 Task \& Environment

To conduct the evaluation, we introduce the character-creation task (see Fig. 3). In this task, a user creates a virtual reality game character by changing its attributes. A character has five distinct attributes with three items per attribute: i) shoes (red, blue, white), ii) shirt (orange, red, blue), iii) glasses (reading, goggles, diving), iv) backpack (pink, blue, red), and v) dance (hip hop, break, silly). The characters' attribute states are limited to one per attribute, i.e., the character cannot be dancing hip hop and break simultaneously. This leads to a total of 15 attribute items and 243 character configurations. We sample uniformly from the different configurations.

The game character's attributes can be changed by selecting the corresponding items in a toolbar-like menu with three slots. The user can cycle through the items by selecting "Next." The static version of the interface has all items of an attribute assigned to the three slots, and every attribute has its own page (e.g., all shoes, if the user presses next, all backpacks). The character's attribute states correspond to the current input state $\mathbf{x}$ and the target state $\mathbf{g}$, where $\mathbf{g}$ is only known to the user agent. The goal of the interface agent to reduce the number of clicks necessary to change an attribute, by assigning the relevant items to the available menu slots. For the user agent, the higher level selects a target slot, and the lower level moves to the corresponding location.

### 7.2 In Silico

Training. We evaluate the training of our method against a static and a random interface. In the random interface, items are randomly assigned to the slots. Figure 4 shows the user agent's task completion rate and number of actions per task of all three interfaces during training. Ours and the static baseline converge, whereas the random baseline does not. Furthermore, the mean number of actions of ours is lower than the mean of the static interface.

Generalization to unseen goals. To understand how well our approach can generalize to unseen goals, we ablate the fraction of goals the agents have access to during training. We then evaluate the learned policies against the full set of goals, which is defined as all possible combinations of character attributes. The results are presented in Figure 5. We find that having access to roughly half of the goals is sufficient to not impact the results. This indicates that our approach generalizes to unseen goals of the same set.

Understanding policy behavior. We qualitatively analyzed the learned policies of our interface agent to understand how it supports the user agent in its task. In Figure 6, we show a snapshot of two sequences with identical initialization. To reach the target character configuration, the user agent can either select the blue bag or the purple glasses (both are needed). Depending on which item the user selects at this time step, the interface agent proposes different suggestions in subsequent steps. For instance, the blue backpack the user did not select initially (Figure 6, bottom) gets suggested again later. This behavior shows that the interface agent implicitly reasons about the attribute that the user intends to select based on previous interactions. In short, the interface agent learns to suggest items that the user is not wearing, or that the user has not interacted with.

Fig. 5. The fraction of successfully completed episodes as function of the fraction of the total number of goals. The graph shows that it is sufficient to see half of the goals to learn policies that generalize to all goals.

Fig. 6. With our method, multiple relevant items can be assigned simultaneously; yet the user can only select one (left). Depending on the user's action (top: select backpack, bottom: select glasses), other item gets assigned later (top: shoes, bottom: backpack). This shows that our method actively adapts to user input.

### 7.3 User Study

Our goal was to create a user agent whose behavior resembles that of real users, so the interface agent can support them in the same task. To this end, we evaluated the sim-to-real transfer capabilities of our framework by conducting a user study where the interface agent interacted with participants instead of the user agent.

7.3.1 Baselines. We compared our method to two supervised learning methods and the static interface (see Sec. 7.1). In line with previous work [31], we used a Support Vector Machine (SVM) with a Radial basis function (RBF) kernel as a baseline. We used the implementations of scikit-learn [76] and optimized the hyperparameters for performance. The feature vector of the baseline was identical to that of our method. The baseline learned the probability with which a user will select a certain character attribute next. We assigned the three attributes with the highest probability to the menu slots. Note that we did not consider "Next" to be an item.

Dataset. We collected data from 6 participants to train the supervised baselines. These participants did not take part in the user study. They interacted with the static interface, which resulted in a dataset with over 3000 logged interactions. We found that more data points did not improve the performance of the SVM classifier through k-fold cross-validation and reached around $91 \%$ top- 3 classification accuracy (ie., the percentage of how often the users' selected item was in the top three of the SVM output) on a test set. Furthermore, we found that the baseline generalize well to unseen participants (again through cross-validation).

Metrics. We used two metrics (dependent variables) to evaluate our approach.
(1) Number of Action: the number of clicks a user needed to complete a task, which is a direct measure of user efficiency [12].
(2) Task Completion Time: the total time a user needed to complete a task.

Fig. 7. The average number of actions (left) and the task completion time (right) participants needed to finish the tasks of our user study. We compare Ours against Static, an SVM, and a Bayes classifier. Our approach performs similar to the two data-driven approaches and uses significantly fewer actions than the static baseline. There is no significant effect on the task completion time.

7.3.2 Procedure. Participants interacted with the interface agent and the two baselines. The three settings were counterbalanced with a Latin square design, and the participants completed 30 trials per setting. In each condition, we discarded the first six trials for training. The participants were instructed to solve the task as fast as possible while reducing the number of redundant actions. They were allowed to rest in-between trials. We ensured that the number of initial attribute differences between the target and current character was uniformly distributed within the participant's trials. Participants used an Oculus Quest 2 with its controller.

We recruited 12 participants from staff and students of an institution of higher education ( 10 male, 2 female, aged between 23 and 33). All participants were right-handed and had a normal or correct-to-normal vision. On average, they needed between 35 to 40 minutes to complete the study.

7.3.3 Results. We present a summary of our results in Figure 7. We analyzed the effect of conditions on the performance of participants with respect to the number of actions and task completion time.

Participants needed on average $4.23 \pm 1.20$ actions to complete a task with our method, compared to $6.19 \pm 0.89$, and $4.40 \pm 0.78$ for the static, and SVM baselines respectively. To analyze the data, we performed a Friedman test as normality was violated (Shapiro-Wilk). We found a significant effect of the method on the number of actions $\left(\chi^2(2)=39.94, p<.001\right)$. Conover's posthoc test revealed significant differences between the static interface and the two other methods (all $p<0.001$ ). We also found an significant effect between our method and the SVM $p=0.020$.

When looking at the task completion time, participants using our method needed $12.76 \pm 2.49$ seconds to complete the task. The completion time was $12.6 \pm 2.25$ for the static interface and $12.8 \pm 2.38$. The task completion time was normally distributed (Shapiro-Wilk $p>0.05$ ). We found no significant difference in overall task completion time with a Greenhouse-Geisser (for sphericity) corrected repeated-measures ANOVA $(F(1.36,14.96)=1.70, p=0.22)$.

### 7.4 Discussion

To analyze if our multi-agent method is competitive with a baseline that requires carefully collected real user data, we compared it against a supervised SVM. We did not find significant differences between the two methods in the task performance metrics of the number of actions and task completion time. This suggests that our approach is a competitive alternative to data-driven methods for creating adaptive user interfaces.

The adaptive methods significantly reduce the number of actions necessary to complete the task compared to the static interface. However, no significant differences in task completion times were found. We argue that this could be due to real users being more familiar with the ordering of items in the static interface that is kept constant across trials. This familiarity is not captured by our current cognitive model or incentivized in the reward function. In future work, we will model familiarity and investigate its effect on task completion time.

We have shown qualitatively that our interface agent learns to take previous user actions into account. This characteristic is core to meaningful adaptations. At the moment, our agent's capabilities are limited by the size of the stack $\mathbf{o}$. In the future, recurrent methods such as LSTM could be investigated to overcome this limitation.

Furthermore, we presented evidence that our method can generalize to goals that were not seen during training. It is important to mention that the results of this experiment are subject to its task and that seen and unseen goals are from the same distribution. Nevertheless, the study provides first indications that our approach generalizes to real-world applications where users' goals might not always be encountered during training.

## 8 ADDITIONAL USECASES

We introduce three additional use cases in VR to demonstrate how our approach generalizes to different scenarios. Note that our method only requires minimal adaptations across tasks. Please refer to our supplementary video for visual demonstrations of the tasks. Because of the different nature of the tasks, we will report either number of clicks or task completion time. In Appendix C, we demonstrate in another use case that our method can also support users in 2D settings.

Number Entry

We introduce a price entry task on an adaptive keypad to show that our approach can support applications requiring users to issue command sequences and provide meaningful help given users' progress in the task (see Fig. 8). The task assumes a setting where the simulated user must enter a product price between 10.00 and 99.99. To complete the task, the user agent has to enter the first two digits, the decimal point, the second two digits, and then press enter. The interface agent can select one of three different interface layouts: i) a standard keypad, ii) a keypad with only digits and iii) a widget with only the decimal point and the enter key (see Fig. 8.2).

The goal difference penalty (Eq. 3) in this case is based on whether the current price $\mathbf{x}$ matches the target price $\mathbf{g}$ :

Formula 8

$$
\mathcal{E}_{g d}=-\sum_t \mathbb{1}_{\mathbf{x}_t \neq \mathbf{g}_t},
$$

where $\mathbb{1}$ is an indicator that is 1 if $\mathbf{x}_t \neq \mathbf{g}_t$ and 0 otherwise, and $t$ is the current timestep. Every time a button is hit, $t$ increases by 1 . This is similar to the penalty in all other tasks. However, it considers that the order of the entries matters. This task converges in 20 hours and 1500 iterations. On average, the user agent needs 4.0 seconds to complete the task in cooperation with the interface agent, compared to 4.9 seconds when using a static keypad. The number of clicks is identical, since the full task can be solved on the standard keypad.

Qualitative Policy Inspection. We observe that the interface agent learns to select the UI that has the biggest buttons for an expected number entry. From this we can conclude that the interface agent implicitly learns the concept of Speed-Accuracy trade off.

Fig. 8. Adaptive keypad: the user agent is asked to enter a randomly initialized price by using a keypad (1). The interface agent can select from three different widgets (2-3): i) a normal keypad, ii) a digits-only keypad, and iii) a non-digits-only keypad. The user agent selects a button of the chosen widget. The task ends when the user agent presses enter (4).

Block Building

The second scenario is a block-building task (Fig. 9) where the user agent constructs various castle-like structures from blocks. It can choose between 4 blocks (wall, gate, tower, roof) and a delete button. The agent needs to move the hand to a staging place for the blocks (see Figure 9) and then place the block in the corresponding location. The block cannot be placed in the air, i.e., it always needs another block on the floor below. The interface agent suggests a next block every time the user agent places a block. However, the user agent can put the block down, in case it is unsuitable. An action is picking or placing a block.

This task represents a subset of tasks that do not have a Heads-Up-Display-like UI to interact with, but are situated directly in the virtual world. This is a common interactive experience of AR/VR systems. This task takes 3000 iterations for both agents to converge. The user agent needs on average 1.1 actions with our method, compared to 2.0 actions without the interface agent. Thus 1.1 indicates that the interface agent suggests the correct next block, most of the time.

Qualitative Policy Inspection. We observe that the policy learns to always suggest a block that is usable given the current state of the tower. This indicates that the policy has an implicit understanding of the order of blocks and can distinguish between those belonging to the foundation versus the upper parts of a tower.

Fig. 9. Block Building: The user agent is building a castle from blocks (1). The user agent places the first block (2). The interface agent suggests a next block to place (3). This is repeated till the castle is built (4).

Fig. 10. Out-of-reach object grabbing: the user agent attempts to grab a specific object, that is initially out of reach, in a space containing multiple objects (1). The user agent learned to move towards an object to indicate its intention to grab it (2). Based on that, the interface agent learned to move the intended object within the user agent's reach (3). The user agent then grabs the object to finish the task (4).

Out-of-reach Item Grabbing

In the third usage scenario, the user needs to grab an object that is initially out of reach. Thus, the interface agent needs to move an object within reach of the user agent, which can then grab it. The interface agent observes the location of the users end-effector. The task environment includes several objects such that user agent and interface agent need to collaborate to select the correct target object and complete the task (see Figure 10). This scenario represents tasks that users cannot solve without the help of an adaptive component. Such a setting commonly arises in UIs of emerging technologies [32]. Our method needed 150 iterations to converge and training ran ca. an hour.

In this use case, we changed the lower level of our user agent to learn motor control with RL instead of using the Fitts-Law-based motor controller. This highlights the modularity of our approach and can be useful in scenarios where existing models, such as Fitt's Law, are not sufficient. We introduce the reinforcement-learning-based motor controller in Appendix B.

Qualitative Policy Inspection. We qualitatively evaluate the learned policy. We find that the the policy selects objects positioned in the direction of the users' arm movement, rather than the closest ones. This indicates that the policy implicitly learns about the correlation between directionality and intent.

## 9 POLICY BEHAVIOR

We have introduced a method that can be applied to vastly different scenarios, from a toolbar in Virtual Reality, to building blocks in a 3D setting, to handing out of reach items. For all these different scenarios the policy learned different behavior. For the toolbar scenario, the policy learned to suggest items the user has not interacted with before or has not already selected. For the number entry, the interface agent learned an implicit understanding of speed-accuracy trade-off. For block building it learned the concept of ordering suggestions in the right sequence. For out-of-reach-item grabbing the policy learned to link intent with directionality.

What all of these components have in common is that the interface agent learned to support the user, without having access to data. We enabled the interface agent to learn the underlying task structure, by treating human-computer interaction as multiplayer cooperative game. The wide variety of use cases we have shown, with the same method formulation, is a strong indicator that this is a first step towards methods that are not application specific handcrafted or rely on offline collected user data.

## 10 LIMITATIONS \& FUTURE WORK

MARLUI is a novel and general approach that offers exciting possibilities for adaptive user interfaces. It models humancomputer interaction as a multiplayer cooperative game by teaching a simulated user agent and an interface agent to cooperate. Learned policies of the interface agent have shown their capability to effectively assist real users. The successful demonstrations of our approach in a wide variety of use cases pose a first step towards general methods that are not tied to specific applications nor dependent on manually crafted rules or offline user data collection. However, there are limitations that require further research.

In our evaluation, we have compared against a supervised learning baseline. We did not find any differences in terms of task completion time or of number of actions necessary. This indicated that our data-free method is on par with data driven methods. This is important, since data can be hard time obtain, unavaible, or of low quality. However, there is room to outperform learned approaches. Since we do not rely on data, we could, for instance, investigate using multiple user models with different levels of skill more easily. Also, investigating continuous learning schemes could enable UIs that develop with the user. Crucially, and similar to our method, this would remain a form of cooperative game.

From an end-user perspective, we assume that the user knows what they are trying to achieve and reach a specific goal state. Removing the dependency on an explicit goal would open up more application domains, such as opened-ended creativity tasks like drawing. This presents an interesting future research direction, where we could learn to infer the user's intended goal from interactions with the interface. Having an informed goal, would enable a goal-oriented RL, which we use in our paper. To achieve this, we could use inverse RL [69] to learn a reward function of the user agent from unlabeled user data, which is cheap to collect. Investigating to what extent these approaches produce reward functions that generalize to unseen user goals remains an open question.

Moreover, user goals can change dynamically during system usage in HCI, particularly in creative tasks where users may have a broad range of objectives. This presents challenges for standard RL approaches, which assume that user goals remain stationary. Future research on MARL for AUIs could focus on finding strategies to easily adapt trained interfaces to new user goals. This would yield more robust and flexible adaptive interfaces.

We have demonstrated that our formulation can solve problems with up to 5 billion possible states (character creation application Sec. 7.1). However, the complexity of the problem grows exponentially with the number of states. This makes it challenging for MARLUI to scale to interfaces with even larger state spaces. To overcome this, we could explore different input modalities, such as representing the state of the UI as an image instead of using a one-hot encoding. This is similar to work on RL agents playing video games [67], which showed that image representations can be effective ways to deal with large state spaces.

We have shown that the simulated user agent's behavior was sufficiently human-like to enable the interface agent to learn helpful policies that transfer to real users. The interface agent's performance is inherently limited by the user agent. Therefore, increasing realism in the model of the simulated user is an interesting future research direction, for instance, achieving human-like gaze patterns [15] or motor control using a biomechanical model [26].

Our method has theoretical appeal because it provides a plausible model of the bilateral nature of AUIs: the adaptation depends on the user, whereas also the user action depends on the adaptation. Modeling this unilaterally as in supervised learning does not reflect reality well. Treating Adaptive UIs as MARL enables a better understanding of how users interact with a UI and how AUIs need to be adapted. Our setup has the potential to scale to multiple users with different skills and intentions. This can lead to bespoke assistive UIs for users with specific needs or UIs for users with specific expertise levels. In line with current research [68], we believe that future work can leverage our method to gain a better theoretical understanding of how users interact with a UI.

In summary, MARLUI is a promising approach that opens up a range of exciting research directions for adaptive UIs that leverage RL. While some limitations exist, such as the complexity challenge and the non-stationarity of user goals, future work can build on our approach to make further strides in this field.

## 11 CONCLUSION

We have taken a first step towards a general reinforcement learning-based framework for adaptive UIs. We introduced a multi-agent reinforcement learning approach that does not rely on any pre-collected user data or task-specific knowledge, while performing on par with data-driven methods.. Our method features a user agent and an interface agent. The user agent tries to achieve a task-dependent goal as fast as possible, while the interface agent learns the underlying task structure by observing the interactions between the user agent and the UI. Since the user agent is RL-based and therefore learns through trial-and-error interactions with the interface, it does not require real user data. We have evaluated our approach in simulation and with humans in a variety of different tasks. Results have shown that our method performs on par with data-driven baselines that rely on task and interface-specific data. This indicates that MARLUI poses a first step towards general methods for adaptive interfaces that are not tied to specific applications nor dependent on offline user data collection.

## A CURRICULUM LEARNING

We use curriculum learning for all settings. Specifically, we adjust the difficulty level every time a criteria has been met by increasing the mean number of initial attribute differences. More initial attribute differences result in longer action sequences and are therefore more complex to learn. We increase the mean by 0.01 every time the successful completion rate is above $90 \%$ and the last level up was at least 10 epochs away.

We randomly sample the number of attribute differences from a normal distribution with standard deviation 1, normalize the sampled number into the range $\left[1, n_a\right]$ and round it to the nearest integer, where $n_a$ is the number of attributes of a setting (in the case of game character $n_a=5$ ).

## B LEARNED LOWER LEVEL

The low-level motor control policy controls the end-effector movement. In particular, given a target slot and a speedaccuracy trade-off weight, the policy selects the parameters of an endpoint distribution. Given the current position and the endpoint parameters (mean and standard deviation), we compute the predicted movement time using the WHo Model [37]. The low-level policy needs to learn i) the coordinates and dimensions of menu slots, ii) an optimal speed-accuracy trade-off given a target slot, and its current position.

To prevent the low-level motor control policy from correcting wrong high-level decisions and to increase general performance, we limit the state space $s_M$ to strictly necessary elements with respect to the motor control task [16]:

Formula 9

$$
s_M=(\mathbf{p}, \mathbf{t}),
$$

with the current position $\mathbf{p} \in I^2$, the target slot $\mathbf{t} \in \mathbb{Z}_2^{n_s}$. The action space $a_M$ is defined as follows:

Formula 10

$$
a_M=\left(\mu_{\mathbf{p}}, \sigma_{\mathbf{p}}\right)
$$

It consists of $\mu_{\mathrm{p}} \in I^2$ and $\sigma_{\mathrm{p}} \in I$, i.e., the mean and standard deviation which describes the endpoint distribution in the unit interval. We scale the standard deviation linearly between a min and max value where the minimum value is the size of normalized pixel width and the max value is empirically chosen to be $15 \%$ of the screen width. Once an action is taken, we sample a new end-effector position from a normal distribution: $\mathbf{p} \sim \mathcal{N}\left(\mu_{\mathbf{p}}, \sigma_{\mathbf{p}}\right)$.

Given the predicted actions, we compute the expected movement time via the WHo model [37], similar to our non-learned low-level motor control policy in the main paper.

The reward for the low-level motor control policy is based on the motoric speed-accuracy trade-off. Specifically, we penalize: i) missing the target supplied by the high-level ( $\neg h$ ), and ii) the movement time ( $T_M$ ). Furthermore, we add a penalty iii) which amounts to the squared Euclidean distance between the center of the target $\mathbf{t}$ and $\mu_{\mathrm{p}}$. This incentivizes the policy to hit the desired target in the correct location. Since the penalty only considers the desired point $\mu_{\mathrm{p}}$, it will not impact the speed-accuracy trade-off (which is a function of $\sigma_{\mathrm{p}}$ ). The total reward is defined as follows:

Formula 11

$$
R_M=\underbrace{\lambda(\neg h)}_{i)}-\underbrace{(1-\lambda) T_M}_{i i)}-\underbrace{\beta\left\|\mu_{\mathrm{p}}-\mu_{\mathrm{t}}\right\|_2^2}_{i i i)},
$$

where $\neg h$ equals 0 when the target button is hit and -1 on a miss. A hit occurs when the newly sampled user position $\mathbf{p}$ is within the target $\mathbf{t}$, while a miss happens if the user position is outside of the target. $\lambda$ is a speed-accuracy trade-off weight and $\beta$ is a small scalar weight to help with learning.

Fig. 11. We introduce a photo editing task where (1) a user matches a photo to a target by operating a hierarchical menu. (2) The user selects the submenu 'size'. (3) The user then selects the attribute 'small', which alters the image. (4) After the user has changed an attribute, the interface observes the new state of the photo and finds the most likely submenu for the next user action. (5) The user clicks on an item in the submenu to complete the task.

## C 2D HIERARCHICAL MENU

In this task, a user edits a photo by changing its attributes. A photo has five distinct attributes with three states per attribute: i) filter (color, sepia, gray), ii) text (none, Lorem, Ipsum), iii) sticker (none, unicorn, cactus), iv) size (small, medium large), and v) orientation (original, flipped horizontal, and vertical). The photo's attribute states are limited to one per attribute, i.e., the photo cannot be in grayscale and color simultaneously. This leads to a total of 15 attribute states and 243 photo configurations.

The graphical interface is a hierarchical menu, where each attribute is a top-level menu entry, and each attribute state is in the corresponding submenu. By clicking a top-level menu, the submenu expands and thus becomes visible and selectable. Only one menu can be expanded at any given time.

The photo attribute states correspond to the current input state $\mathbf{x}$ and the target state $\mathbf{g}$, where $\mathbf{g}$ is only known to the user agent. The interface agent selects an attribute menu to open. Its goal is to reduce the number of clicks necessary to change an attribute, e.g., from two user interactions (filter->color) to one (color). For the user agent, the higher level selects a target slot, and the lower level moves to the corresponding location.