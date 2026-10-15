"""Проверка подключения к модели и пробный набор белорусских запросов.

  python test_llm.py --list-free   список бесплатных моделей OpenRouter
  python test_llm.py               один тестовый запрос
  python test_llm.py --stream      то же, но потоком
  python test_llm.py --bench       10 запросов, результат в bench_results.jsonl
"""
import argparse
import json
import time

from llm_client import chat, chat_stream, list_free_models

PROMPTS = [
    "Прывітанне! Як справы?",
    "Як даехаць да цэнтра горада на аўтобусе?",
    "Раскажы коратка пра свята Купалле.",
    "Привет! Расскажи, что такое Каляды.",
    "Што такое Нёман і дзе ён цячэ?",
    "Дапамажы напісаць заяву ў жыллёва-камунальную службу.",
    "Як сказаць па-беларуску: зарплата, магазин, продавец, библиотека?",
    "Я пайшоў да дома ўчора. Ці правільна так сказаць?",
    "Раскажы пра Янку Купалу ў двух сказах.",
    "Які сёння курс долара?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-free", action="store_true")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()

    if a.list_free:
        for m in list_free_models():
            print(m)
        return

    if a.bench:
        with open("bench_results.jsonl", "a", encoding="utf-8") as f:
            for p in PROMPTS:
                t = time.time()
                try:
                    ans = chat([{"role": "user", "content": p}])
                except Exception as e:  # noqa: BLE001
                    ans = f"ОШИБКА: {e}"
                dt = round(time.time() - t, 1)
                f.write(json.dumps({"prompt": p, "answer": ans, "sec": dt}, ensure_ascii=False) + "\n")
                print(f"[{dt}s] {p}\n  -> {ans}\n")
        print("Результаты в bench_results.jsonl, отдай их на вычитку лингвисту.")
        return

    q = PROMPTS[0]
    if a.stream:
        for piece in chat_stream([{"role": "user", "content": q}]):
            print(piece, end="", flush=True)
        print()
    else:
        print(chat([{"role": "user", "content": q}]))


if __name__ == "__main__":
    main()
