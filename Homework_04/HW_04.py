from google import generativeai as genai
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import logging
from google.api_core.exceptions import (
    ResourceExhausted,
    ServiceUnavailable,
    DeadlineExceeded,
    InternalServerError
)

logger = logging.getLogger(__name__)

console = Console()

RETRY_CONFIG = {
    "stop": stop_after_attempt(5),
    "wait": wait_exponential(multiplier=1, min=2, max=30),
    "retry": retry_if_exception_type(
        (ResourceExhausted, ServiceUnavailable, DeadlineExceeded, InternalServerError)
    ),
    "before_sleep": before_sleep_log(logger, logging.WARNING),
    "reraise": True
}

@retry(**RETRY_CONFIG)
def generate_content_with_retry(model, prompt, timeout=30):
    try:
        response = model.generate_content(prompt, timeout=timeout)
        return response
    except Exception as e:
        console.print(Panel.fit(f"[yellow]Попытка не удалась. Ошибка: {str(e)}[/yellow]", border_style="yellow"))
        raise

def main():
    load_dotenv(override=True)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        console.print(Panel.fit(
            "[bold red]Ошибка: Не найден API-ключ Gemini[/bold red]\n"
            "Проверьте:\n"
            "1. Файл .env существует и содержит 'GEMINI_API_KEY=ваш_ключ'\n"
            "2. Ключ не содержит лишних пробелов/кавычек\n"
            "3. Скрипт запускается из той же папки, где лежит .env",
            title="Ошибка конфигурации",
            border_style="red"))
        return

    console.print("[green]API-ключ Gemini успешно загружен![/green]")

    try:
        genai.configure(
            api_key=GEMINI_API_KEY,
            transport="rest",
            client_options={"api_endpoint": "generativelanguage.googleapis.com"}
        )

        model = genai.GenerativeModel('gemini-pro')
        console.print(Panel.fit(
            "[bold green]🤖 Gemini AI Ассистент активирован[/bold green]",
            subtitle="Подключение к Google AI Studio",
            border_style="green"))

        with console.status("[bold cyan]Инициализация модели...[/bold cyan]") as status:
            time.sleep(2)

        prompt = "Расскажи о себе в формате дружелюбного ИИ-ассистента."
        console.print(Panel.fit(
            f"[bold yellow]Ваш запрос:[/bold yellow] [white]{prompt}[/white]",
            title="📝 Ввод",
            border_style="yellow"))

        time.sleep(1)
        response = generate_content_with_retry(model, prompt, timeout=20)

        console.print(Panel.fit(
            Text.from_markup(response.text),
            title="💡 Ответ Gemini",
            border_style="blue",
            subtitle="Ответ сгенерирован",
            padding=(1, 4)))

        console.print(Panel.fit(
            "[italic]Вы можете продолжить диалог, изменив переменную 'prompt'[/italic]",
            border_style="dim"))

    except Exception as e:
        console.print(Panel.fit(
            f"[bold red]Критическая ошибка после всех попыток:[/bold red]\n{str(e)}",
            title="Ошибка",
            border_style="red"))

if __name__ == "__main__":
    main()