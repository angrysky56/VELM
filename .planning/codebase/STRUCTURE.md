# Structure

```
.
├── src/                    # Core implementation
│   ├── model/              # Architecture primitives
│   │   ├── autoencoder.py  # CALM implementation
│   │   ├── energy_head.py  # Prediction head
│   │   └── miras_backbone.py # Core backbone
│   ├── training/           # Optimization and data
│   │   ├── eggroll.py      # Main optimizer
│   │   ├── fitness.py      # Training objectives
│   │   └── data_loader.py  # Streaming pipelines
│   ├── evolution/          # Population strategies
│   └── inference/          # Prediction and decoding
├── notebooks/              # Active Colab development
├── tests/                  # Verification suite
├── scripts/                # Standalone entry points
├── docs/                   # Documentation and research
├── backups/                # Persistent script backups
└── checkpoints/            # Serialized weights and metadata
```
