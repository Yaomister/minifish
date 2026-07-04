![Alt text](/text-art.png)

I love chess, but I'm terrible at it.

minifish is a lightweight chess evaluation network. It is a chess engine powered by an efficient update neural network (NNEU), the same core idea behind Stockfish's evaluation, scaled down many times.

## Design notes

minifish is built on one idea: **the neural network's first layer and the engine's accumulator are the same thing.** Both take the position's feature indicies and su, the matching weight columns (simple matrix multiplication). Training does it with a `EmbeddingBag`; the engine does it incrementally, adding and subtracting tensors as the board changes.

A few points should be noted:

- **The dataset stores indices, not tensors.** A position is ~30 integers (one per non-king piece), not a dense board grid (roughtly 17x smaller and independent of the neural network's width.)
- **Moving the king triggers a full accumulator reset.** The king's square is baked into every feature index, so when the king moves, the accumulator rebuilds.
- **The score is trained in tanh space.** Stockfish evalulation scores are unbounded and mate is infinite; `tanh(eval/4)` compresses blowouts.

## File structure

```
.
├── README.md
├── requirements.txt
├── chess.py            # The chess engine
├── accumulator.py      # The accumulator
├── nneu.py             # The network architecture
├── training.py         # Training loop
├── evaluator.py        # Board evaluation with adversarial search (alpha-beta pruning minimax)
├── main.py             # Entry point
├── data
│   └── parser.py       # Streams lichess.zst -> training/dataset_*.npz
├── scripts
│   ├── parse.sh        # SLURM job: parsing
│   └── train.sh        # SLURM job: training
├── training/           # Parsed dataset chunks (generated)
└── weights/            # Trained weights (generated)
```

## Acknowledgements

The NNUE architecture originates with Yu Nasu's 2018 paper and its adoption into [Stockfish](https://github.com/official-stockfish/Stockfish). The [nnue-pytorch docs](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md) are the best explanation of it anywhere.

Training data comes from the wonderful open [Lichess database](https://database.lichess.org/).

## License

MIT
