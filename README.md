# roulette

## Simple Q-Learning agent for OpenAI Gym Roulette-v0 environment.

This is a simulated casino Roulette environment from 1-36, including a 0,
where the agent can bet either even (actions 0, 2, 4, ..., 34),
odd (actions 1, 3, 5, ..., 35), 0 (action 36), or
walk away without betting anything (action 37)

The reward for correctly betting either even or odd is +1, and -1 for incorrect betting.
The reward for correctly betting 0 is +35, and -1 for incorrect betting.
The reward for walking away is 0.

As you can see from the final value of the q-table, the agent quickly learns that the best
result is to walk away: any other action results in a net negative reward,
just like at real casinos.

This is because of the "house advantage." The game of Roulette seemingly rewards bets appropriately
based on their risk: betting all even numbers from 1-36 pays you double your bet if successful,
but you will lose your money about half the time.

However, Roulette often has a 37th value, the number 0, which does not count towards any other bets,
except for a direct bet on 0, which gives a reward of 35.

Thus, if you were to bet on even, you will lose money 19/37 times and double your money 18/37 times,
ultimately resulting in a net loss of money, or "reward."

The code is pretty bare and simple, and there are many ways to improve upon it, such as:

Better visualization

Simulate a testing agent

Tweak hyperparameters


Source code for the Roulette-v0 environment itself can be found here: https://github.com/openai/gym/blob/master/gym/envs/toy_text/roulette.py


Credit to https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
for providing base code for the Q-Learning portion.
