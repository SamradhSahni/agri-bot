# KisanMitra AI — Fine-tuned mT5-base

## Model Details
- **Base model**: google/mt5-base (580M parameters)
- **Fine-tuning**: QLoRA (4-bit NF4, r=16, alpha=32)
- **Trainable params**: ~6.7M (1.15% of total)
- **Training data**: 20,000 Hindi-Hindi agricultural QA pairs
- **Best val loss**: 1.9585 (epoch 3)

## Task
Seq2Seq instruction-following for Hindi agricultural advisory.
Input: Hindi instruction with state, crop, intent, query.
Output: Hindi advisory answer.

## Intents Supported
weather_sowing, crop_advisory, pest_id, disease,
nutrient_management, msp_price, government_scheme,
horticulture, soil_water, animal_husbandry, equipment_machinery

## Dataset
- Source: KCC (Kisan Call Centre) helpline logs
- States: UP, Rajasthan, Haryana, Bihar, MP, Chhattisgarh,
          Himachal Pradesh, Jharkhand, Uttarakhand
- Language: Hindi (Devanagari)
- Queries translated from English using IndicTrans2
