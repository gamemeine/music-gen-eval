# Can a music generator grade itself? A model's own loss and hidden states as an evaluator, across models and under acceleration

## About
A music generator is usually judged from outside: a separate model rates its outputs, or a distance between generated and reference audio is computed. A 2026 paper showed that a generator's own intrinsic signals, the loss it assigns to a clip and the activity of its hidden layers, correlate with human ratings of that clip. This project asks whether such a free evaluator can be trusted: whether a model rates other models' outputs fairly or prefers its own, whether the correlation survives a prompt-only floor (the agreement a scorer reaches from the text prompt alone, without the audio), and whether accelerated models, which sample in a few steps and no longer compute a proper likelihood, keep a usable signal.

The desing proposal can be found in [DESIGN_PROPOSAL.md](DESIGN_PROPOSAL.md).

