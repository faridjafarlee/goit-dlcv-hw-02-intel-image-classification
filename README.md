# ДЗ 2 (тема 6): класифікація зображень Intel Image Classification

Здавати: `dz_topic_6_Джафарлі_Фарід.ipynb`. У полі здачі написати `1` (перший підхід).

Модель: ResNet18 (transfer learning), доналаштована вся мережа.

| Модель | val F1 | test accuracy | test F1 |
|---|---|---|---|
| основна | 0.933 | 0.928 | 0.929 |
| з аугментацією (фінальна) | 0.935 | 0.930 | 0.931 |
| заморожена, навчається лише останній шар | 0.909 | 0.906 | 0.907 |
| проста CNN з нуля (варіант A) | 0.847 | 0.836 | 0.836 |

Орієнтир з умови: для transfer learning accuracy від 90 до 97% і F1 від 0.89 до 0.96, для простої CNN accuracy від 75 до 85% і F1 від 0.73 до 0.83.

Інші файли:
- `build_notebook.py` збирає ноутбук;
- `data` посилання на набір даних у кеші kagglehub (якщо його немає, ноутбук завантажує дані з Kaggle);
- `checkpoints/` збережені найкращі моделі, здавати їх не потрібно;
- `dump_outputs.py` перевіряє, що в ноутбуці немає помилок;
- `old_v1/` попередня, розширена версія.

Перезапуск (близько 20 хвилин на GPU Apple M2 Max):

```bash
../.venv/bin/python build_notebook.py
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 dz_topic_6_Джафарлі_Фарід.ipynb
```
