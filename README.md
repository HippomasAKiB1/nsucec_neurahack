# NSUCEC NeuraHack 2026 — Mock Preliminary Round

Solution for **TransitText: Public Transport Feedback Classification** (6-class classification across English, Bengali, Banglish, and mixed-language passenger messages evaluated by **Macro F1-score**).

## Classes
- `0` **Delay**: Late buses, traffic delays, long waiting times, missed schedules, breakdowns
- `1` **Fare**: Ticket prices, overcharging, fare disputes, missing change, student fare refusal, payment errors
- `2` **Safety**: Harassment, reckless/drunk driving, accidents, unsafe conditions, pickpocketing/snatching
- `3` **Cleanliness**: Dirty seats/windows/toilets, waste/garbage, spit, vomit, bad smells, pests
- `4` **Service**: Staff behavior, overcrowding, poor service quality, AC off, skipping stops, reserved seats
- `5` **Other**: Inquiries, lost & found, app/website issues or praise, missing signs, route/feature suggestions, form tests

## Approach & Validation Progression
| Stage | Method | Validation (`sample.csv`) |
|---|---|---|
| Baseline | `TF-IDF` + `LogReg` (raw) | `0.2333` LOO / `0.3333` 10-Fold |
| Multilingual Encoder | `BAAI/bge-m3` + `LinearSVC` (raw) | `0.6833` LOO / `0.7000` 10-Fold |
| Normalization + Denoising | Wrapper stripping, typo repair, Banglish/Bengali normalization ([text_clean.py](experiments/text_clean.py)) + `BAAI/bge-m3` + `Ridge` | `0.9167` LOO / `0.9172` Macro-F1 |
| LLM Few-Shot | `Qwen/Qwen2.5-1.5B-Instruct` KV-cached single-token logit classifier ([step5c_qwen_kvcached.py](experiments/step5c_qwen_kvcached.py)) | `0.9000` 2-Fold / `0.9667` 24-shot LOO |
| Stage-2 Ensemble (v1) | `Qwen2.5-1.5B` + `BGE-M3` + `Nomic-v1.5` + `mE5-base` + `TF-IDF` + OOF-verified self-training + cluster voting ([step6_final_ensemble.py](experiments/step6_final_ensemble.py)) | `0.9500` OOF Acc / `0.9496` Macro-F1 |
| Synthetic-Augmented Ensemble (v2) | Rule-5 synthetic training examples ([synthetic_aug.py](experiments/synthetic_aug.py)) + `BGE-Large-EN` + `BGE-M3` + `Nomic-v1.5` + `mE5-base` + `Qwen2.5-1.5B` + Stage-2 self-training ([step9b_synth_and_ensemble.py](experiments/step9b_synth_and_ensemble.py)) | **`0.9833` LOO Acc / `0.9833` Macro-F1** |

## Repository Structure
- `mock/`: Competition overview, rules, and dataset (`sample.csv`, `test.csv`, `sample_submission.csv`)
- `experiments/`: Iterative EDA, normalization, CV benchmarks, LLM few-shot inference, synthetic augmentation, and ensembling scripts
- `submission.csv`: Final verified submission (`300` rows)

## Public Pretrained Models Used (Rule 5 / Section 6 Disclosure)
- `Qwen/Qwen2.5-1.5B-Instruct`
- `BAAI/bge-large-en-v1.5`
- `BAAI/bge-m3`
- `intfloat/multilingual-e5-base`
- `nomic-ai/nomic-embed-text-v1.5`
- *(Evaluated in ablations only)*: `intfloat/multilingual-e5-small`, `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli`, `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, `facebook/nllb-200-distilled-600M`
