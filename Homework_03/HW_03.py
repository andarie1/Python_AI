from google import generativeai as genai
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import time


console = Console()

def main():
    load_dotenv(override=True)

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        console.print(
            Panel.fit(
                "[bold red]Ошибка: Не найден API-ключ Gemini[/bold red]\n"
                "Проверьте:\n"
                "1. Файл .env существует и содержит 'GEMINI_API_KEY=ваш_ключ'\n"
                "2. Ключ не содержит лишних пробелов/кавычек\n"
                "3. Скрипт запускается из той же папки, где лежит .env",
                title="Ошибка конфигурации",
                border_style="red"
            )
        )
        return

    console.print("[green]API-ключ Gemini успешно загружен![/green]")

    try:
        genai.configure(api_key=GEMINI_API_KEY)

        model = genai.GenerativeModel('gemini-pro')

        console.print(
            Panel.fit(
                "[bold green]🤖 Gemini AI Ассистент активирован[/bold green]",
                subtitle="Подключение к Google AI Studio",
                border_style="green"
            )
        )

        with console.status("[bold cyan]Инициализация модели...[/bold cyan]") as status:
            time.sleep(2)

        prompt = "Расскажи о себе в формате дружелюбного ИИ-ассистента."

        console.print(
            Panel.fit(
                f"[bold yellow]Ваш запрос:[/bold yellow] [white]{prompt}[/white]",
                title="📝 Ввод",
                border_style="yellow"
            )
        )

        response = model.generate_content(prompt)

        console.print(
            Panel.fit(
                Text.from_markup(response.text),
                title="💡 Ответ Gemini",
                border_style="blue",
                subtitle="Ответ сгенерирован",
                padding=(1, 4)
            )
        )
        console.print(
            Panel.fit(
                "[italic]Вы можете продолжить диалог, изменив переменную 'prompt'[/italic]",
                border_style="dim"
            )
        )

    except Exception as e:
        console.print(
            Panel.fit(
                f"[bold red]Произошла ошибка:[/bold red]\n{str(e)}",
                title="Ошибка",
                border_style="red"
            )
        )


if __name__ == "__main__":
    main()
