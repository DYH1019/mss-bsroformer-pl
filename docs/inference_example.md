### Inference example

```bash
python inference.py \
    --model_type bs_roformer \
    --config_path /23SA01/codes/Music-Source-Separation-BSRoFormer-pl/configs/train_config_bs_roformer.yaml \
    --start_check_point /23SA01/codes/Music-Source-Separation-BSRoFormer-pl/checkpoints/bs_roformer_vocal_wavebleeding-epoch=985-sdr=10.917.ckpt \
    --input_folder /23SA01/datasets/test \
    --extract_instrumental \
    --store_dir separation_results/ \
    --device_ids 3 \
```

See more parameters in inference.py
