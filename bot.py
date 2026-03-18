"""
Скрипт 1 — Прямой запрос к OpenAI через официальную библиотеку.

Установка зависимостей:
    pip install openai python-dotenv

Запуск:
    python bot.py

Настройка модели — в файле .env:
    OPENAI_MODEL=gpt-4o          # любая модель OpenAI
    OPENAI_MODEL=gpt-3.5-turbo   # по умолчанию, если не задано

Заготовленные запросы — в файле prompts.json:
    Каждый промпт содержит поля: role, context, question, format, test_input
"""

import os
import sys
import json
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    print("Ошибка: библиотека openai не установлена.")
    print("Выполните: pip install openai")
    sys.exit(1)


DEFAULT_MODEL = "gpt-3.5-turbo"
PROMPTS_FILE = os.path.join(os.path.dirname(__file__), "prompts.json")


def load_prompts() -> list[dict]:
    if not os.path.exists(PROMPTS_FILE):
        print(f"Файл с промптами не найден: {PROMPTS_FILE}")
        return []
    try:
        with open(PROMPTS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("prompts", [])
    except (json.JSONDecodeError, OSError) as e:
        print(f"Ошибка чтения {PROMPTS_FILE}: {e}")
        return []


def build_system_message(prompt: dict) -> str:
    parts = [
        "СТРОГОЕ ОГРАНИЧЕНИЕ: Ты являешься виртуальным менеджером автосервиса по удалению вмятин без покраски (PDR). "
        "Ты НЕ являешься универсальным ассистентом и НЕ умеешь делать ничего, кроме консультации по услугам PDR. "
        "На любой вопрос вне темы услуг отвечай только: что занимаешься исключительно удалением вмятин без покраски. "
        "Никогда не перечисляй свои возможности как ИИ и не обсуждай темы вне автосервиса."
    ]
    if prompt.get("role"):
        parts.append(prompt["role"] + ".")
    if prompt.get("context"):
        parts.append("Контекст: " + prompt["context"] + ".")
    if prompt.get("question"):
        parts.append("Задача: " + prompt["question"] + ".")
    if prompt.get("format"):
        parts.append("Формат ответа: " + prompt["format"] + ".")
    return "\n".join(parts)


def select_prompt(prompts: list[dict]) -> dict | None:
    print("\nДоступные заготовленные запросы:")
    print("-" * 40)
    for p in prompts:
        print(f"  [{p['id']}] {p['name']}")
    print("  [0] Ввести запрос вручную")
    print("-" * 40)

    while True:
        raw = input("Выберите номер запроса: ").strip()
        if raw == "0":
            return None
        try:
            choice = int(raw)
            for p in prompts:
                if p["id"] == choice:
                    return p
            print(f"  Нет запроса с номером {choice}. Попробуйте снова.")
        except ValueError:
            print("  Введите число.")


def get_float_input(prompt: str, default: float, min_val: float, max_val: float) -> float:
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return default
        try:
            value = float(raw)
            if min_val <= value <= max_val:
                return value
            print(f"  Введите число от {min_val} до {max_val}.")
        except ValueError:
            print("  Неверный формат. Введите число.")


def get_int_input(prompt: str, default: int, min_val: int, max_val: int) -> int:
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return default
        try:
            value = int(raw)
            if min_val <= value <= max_val:
                return value
            print(f"  Введите целое число от {min_val} до {max_val}.")
        except ValueError:
            print("  Неверный формат. Введите целое число.")


def send_message(
    client: OpenAI,
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    frequency_penalty: float,
) -> tuple[str, object]:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
    )
    return response.choices[0].message.content, response.usage


def setup_session(prompts: list[dict]) -> tuple[str, str]:
    """Выбор промпта и первого сообщения. Возвращает (system_message, first_user_input)."""
    system_message = ""
    user_input = ""

    if prompts:
        selected = select_prompt(prompts)
        if selected:
            print(f"\nВыбран промпт: «{selected['name']}»")
            print(f"  Роль:     {selected.get('role', '—')}")
            print(f"  Контекст: {selected.get('context', '—')}")
            print(f"  Задача:   {selected.get('question', '—')}")
            print(f"  Формат:   {selected.get('format', '—')}")

            system_message = build_system_message(selected)

            test_input = selected.get("test_input", "")
            if test_input:
                print(f"\nТестовый ввод:\n  {test_input}")
                use_test = input("\nИспользовать тестовый ввод? (Enter — да / введите свой): ").strip()
                user_input = use_test if use_test else test_input
            else:
                user_input = input("\nВаш первый вопрос: ").strip()
        else:
            system_message = input("\nSystem message (Enter — пропустить): ").strip()
            user_input = input("Ваш первый вопрос: ").strip()
    else:
        system_message = input("\nSystem message (Enter — пропустить): ").strip()
        user_input = input("Ваш первый вопрос: ").strip()

    return system_message, user_input


def main() -> None:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Ошибка: переменная OPENAI_API_KEY не найдена.")
        print("Создайте файл .env и добавьте: OPENAI_API_KEY=sk-...")
        sys.exit(1)

    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    prompts = load_prompts()

    print("=" * 50)
    print(f"  OpenAI — {model}  |  Режим диалога")
    print("=" * 50)
    print("  Команды в диалоге:")
    print("    /new   — начать новый диалог (сбросить контекст)")
    print("    /exit  — выйти из программы")
    print("=" * 50)

    temperature = get_float_input("\nТемпература [0.0–2.0, по умолчанию 0.7]: ", 0.7, 0.0, 2.0)
    max_tokens = get_int_input("max_completion_tokens [1–32000, по умолчанию 1024]: ", 1024, 1, 32000)
    frequency_penalty = get_float_input("frequency_penalty [0.0–2.0, по умолчанию 0.5]: ", 0.5, 0.0, 2.0)

    client = OpenAI(api_key=api_key)
    total_tokens_session = 0

    while True:
        system_message, first_input = setup_session(prompts)

        if not first_input:
            print("Запрос не может быть пустым.")
            continue

        messages: list[dict] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})

        turn = 0

        current_input = first_input

        while True:
            if current_input.strip().lower() == "/exit":
                print(f"\nСессия завершена. Всего токенов за сессию: {total_tokens_session}")
                sys.exit(0)

            if current_input.strip().lower() == "/new":
                print("\n" + "=" * 50)
                print("  Начинаем новый диалог...")
                print("=" * 50)
                break

            messages.append({"role": "user", "content": current_input})
            turn += 1

            print(f"\n--- Отправка запроса (сообщение #{turn})... ---\n")

            try:
                answer, usage = send_message(
                    client, messages, model, temperature, max_tokens, frequency_penalty
                )
            except Exception as exc:
                print(f"Ошибка API: {exc}")
                messages.pop()
                current_input = input("\nВаш вопрос: ").strip()
                continue

            messages.append({"role": "assistant", "content": answer})
            total_tokens_session += usage.total_tokens

            print("=" * 50)
            print(f"ОТВЕТ  [сообщение #{turn}]:")
            print("=" * 50)
            print(answer)
            print("=" * 50)
            print(
                f"Токены: запрос={usage.prompt_tokens}, "
                f"ответ={usage.completion_tokens}, "
                f"итого={usage.total_tokens}  |  "
                f"всего в сессии={total_tokens_session}"
            )
            print(f"История: {len([m for m in messages if m['role'] != 'system'])} сообщений")
            print("-" * 50)

            current_input = input("Ваш вопрос (/new — новый диалог, /exit — выход): ").strip()


if __name__ == "__main__":
    main()
